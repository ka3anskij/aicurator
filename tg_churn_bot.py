import asyncio
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    constants,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# -------------------- Settings --------------------

TOKEN = os.getenv("TG_TOKEN")
if not TOKEN:
    raise RuntimeError("Set TG_TOKEN env var.")

UPLOAD_DIR = Path(os.getenv("DATA_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH", "prod_lms_dropout_model.pkl")
CONFIG_PATH = os.getenv("CONFIG_PATH", "prod_lms_dropout_config.json")

DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.35"))
DEFAULT_TOPN = int(os.getenv("TOP_N", "20"))

# включить, если хочешь дергать внешний раннер вместо встроенного эвристического
USE_EXTERNAL_RUNNER = False  # <-- можно переключить на True, если есть CLI/модуль

# -------------------- Logging --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# -------------------- Small helpers --------------------


def kb_home() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🚀 Запустить /run", callback_data="RUN"),
                InlineKeyboardButton("📊 Статус", callback_data="STATUS"),
            ],
            [
                InlineKeyboardButton("⚙️ Порог риска", callback_data="SET_THR"),
                InlineKeyboardButton("🔝 Top-N", callback_data="SET_TOPN"),
            ],
            [
                InlineKeyboardButton("🧹 Очистить выгрузки", callback_data="CLEAR"),
                InlineKeyboardButton("ℹ️ Помощь", callback_data="HELP"),
            ],
        ]
    )


def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="HOME")]])


def fmt_ok(text: str) -> str:
    return f"✅ <b>{text}</b>"


def fmt_warn(text: str) -> str:
    return f"⚠️ <b>{text}</b>"


def fmt_err(text: str) -> str:
    return f"❌ <b>{text}</b>"


def human_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def latest_file_in_uploads() -> Path | None:
    files = sorted(
        [p for p in UPLOAD_DIR.glob("*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def parse_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


# -------------------- UX: progress / errors --------------------


async def run_with_progress(
    update: Update,
    context: CallbackContext,
    title: str,
    coro,
) -> None:
    """
    Показывает анимированный прогресс, параллельно исполняя coro().
    Любые исключения ловим и печатаем пользователю.
    """
    chat_id = update.effective_chat.id
    frames = ["⏳", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚"]
    i = 0
    started = datetime.now()

    msg = await context.bot.send_message(
        chat_id=chat_id,
        text=f"{frames[i]} <b>{title}</b>\n<i>идёт обработка…</i>",
        parse_mode=constants.ParseMode.HTML,
        reply_markup=kb_back(),
    )

    async def spinner():
        nonlocal i
        while True:
            i = (i + 1) % len(frames)
            elapsed = (datetime.now() - started).seconds
            try:
                await msg.edit_text(
                    f"{frames[i]} <b>{title}</b>\n<i>идёт обработка… {elapsed}s</i>",
                    parse_mode=constants.ParseMode.HTML,
                    reply_markup=kb_back(),
                )
            except Exception:
                pass
            await asyncio.sleep(1.2)

    spin_task = asyncio.create_task(spinner())
    try:
        result_text = await coro()
        await msg.edit_text(
            result_text, parse_mode=constants.ParseMode.HTML, reply_markup=kb_home()
        )
    except Exception as e:
        log.exception("Run failed")
        await msg.edit_text(
            fmt_err("Ошибка во время обработки")
            + "\n\n"
            + f"<code>{type(e).__name__}: {e}</code>",
            parse_mode=constants.ParseMode.HTML,
            reply_markup=kb_home(),
        )
    finally:
        spin_task.cancel()


# -------------------- Domain: scoring --------------------

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # пытаемся угадать колонки
    cols = {c.lower(): c for c in df.columns}
    # email
    email_col = next((cols[k] for k in cols if "email" in k or "mail" in k), None)
    # дни без активности
    days_col = next(
        (cols[k] for k in cols if "days" in k and ("last" in k or "inactive" in k)), None
    )
    if not days_col:
        days_col = next((cols[k] for k in cols if "days_since" in k), None)
    # события за 28 дней
    ev28_col = next(
        (cols[k] for k in cols if ("28" in k and "event" in k) or "events_28" in k), None
    )
    # средний квиз
    quiz_col = next(
        (cols[k] for k in cols if ("quiz" in k or "score" in k) and "avg" in k), None
    )

    # fallback: популярные имена
    email_col = email_col or cols.get("email") or cols.get("user_email")
    days_col = days_col or cols.get("days_since_last_activity") or cols.get("days_since")
    ev28_col = ev28_col or cols.get("events_28d") or cols.get("events_last_28d")
    quiz_col = quiz_col or cols.get("quiz_avg") or cols.get("avg_quiz")

    missing = [n for n in ("email", "days", "events28", "quiz") if locals()[f"{n}_col"] is None]
    if missing:
        raise ValueError(
            f"Не нашёл необходимые колонки: {', '.join(missing)}. "
            f"Колонки в файле: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "email": df[email_col].astype(str),
            "days_since": pd.to_numeric(df[days_col], errors="coerce").fillna(0).astype(int),
            "events_28d": pd.to_numeric(df[ev28_col], errors="coerce").fillna(0).astype(int),
            "quiz_avg": pd.to_numeric(df[quiz_col], errors="coerce").fillna(0.0),
        }
    )
    return out


def _rule_score(row) -> float:
    # простая интерпретируемая эвристика
    s = 0.0
    # давность
    if row.days_since >= 60:
        s += 0.65
    elif row.days_since >= 40:
        s += 0.5
    elif row.days_since >= 20:
        s += 0.35
    # события
    if row.events_28d <= 0:
        s += 0.25
    elif row.events_28d <= 2:
        s += 0.15
    elif row.events_28d <= 5:
        s += 0.1
    # квизы
    if row.quiz_avg < 40:
        s += 0.2
    elif 40 <= row.quiz_avg < 60:
        s += 0.1
    else:
        s -= 0.05
    return max(0.0, min(1.0, s))


def _reasons(row) -> List[str]:
    rs = []
    if row.days_since >= 1:
        rs.append(f"нет активности {row.days_since}дн")
    if row.events_28d <= 5:
        rs.append(f"мало событий за 28д ({row.events_28d})")
    if row.quiz_avg <= 60:
        rs.append(f"низкие квизы (avg ~{int(round(row.quiz_avg))})")
    return rs


def run_builtin_scoring(threshold: float, top_n: int) -> Tuple[str, int]:
    path = latest_file_in_uploads()
    if not path:
        raise FileNotFoundError("Не найдено ни одной выгрузки в /uploads")

    df = _normalize_cols(_read_any(path))
    df["p"] = df.apply(_rule_score, axis=1)
    df.sort_values("p", ascending=False, inplace=True)

    # алёрты
    alerted = df[df["p"] >= threshold].head(top_n)
    total = len(df)
    lines = [f"🔔 Всего алёртов: <b>{len(alerted)}/{total}</b>"]
    idx = 1
    for _, r in alerted.iterrows():
        reasons = "; ".join(_reasons(r))
        lines.append(f"{idx}. <code>{r.email}</code> — p={r.p:.2f} — {reasons}")
        idx += 1

    if len(alerted) == 0:
        lines.append("Порог слишком высокий? Попробуй понизить командой <code>/set_threshold 0.3</code>")

    header = f"<b>Файл:</b> {path.name} ({human_ts(path.stat().st_mtime)})"
    return header + "\n\n" + "\n".join(lines), len(alerted)


async def run_external_runner(threshold: float, top_n: int) -> Tuple[str, int]:
    """
    Заглушка на случай, если хочешь использовать ваш продовый раннер.
    Тут можно сделать subprocess или импорт модуля.
    Пока возвращаем None -> чтобы не ломать бот, используем builtin.
    """
    raise NotImplementedError


# -------------------- Bot state --------------------

STATE = {
    "threshold": DEFAULT_THRESHOLD,
    "top_n": DEFAULT_TOPN,
}

SCHED = AsyncIOScheduler()


# -------------------- Handlers --------------------

async def on_start(update: Update, context: CallbackContext):
    await update.message.reply_html(
        "Привет! Я бот алёртов по оттоку LMS.\n"
        "Пришли CSV/XLSX выгрузки, я их сохраню.\n\n"
        "Нажимай кнопки ниже или используй команды.\n",
        reply_markup=kb_home(),
    )


async def on_help(update: Update, context: CallbackContext):
    await update.effective_message.reply_html(
        "<b>Команды</b>\n"
        "/run — запустить скоринг по последним выгрузкам\n"
        "/status — конфиг и текущая папка\n"
        "/set_threshold &lt;0..1&gt;\n"
        "/set_notify_top_n &lt;N&gt;\n"
        "/schedule_every &lt;days&gt; &lt;HH:MM&gt;\n"
        "/cancel_schedule — отменить расписание\n"
        "/clear_uploads — удалить старые выгрузки\n"
        "/version — версия бота",
        reply_markup=kb_home(),
    )


async def on_version(update: Update, context: CallbackContext):
    await update.effective_message.reply_html(
        "tg-bot: <b>progress-enabled v2</b>", reply_markup=kb_home()
    )


async def on_status(update: Update, context: CallbackContext):
    lf = latest_file_in_uploads()
    text = [
        "<b>Статус:</b>",
        f"Model: <code>{MODEL_PATH}</code>",
        f"Config: <code>{CONFIG_PATH}</code> (ok)",
        f"Exports: <code>{UPLOAD_DIR}</code>",
        f"Threshold: <b>{STATE['threshold']:.2f}</b>",
        f"Top-N: <b>{STATE['top_n']}</b>",
    ]
    if lf:
        text.append(f"Last file: <code>{lf.name}</code> ({human_ts(lf.stat().st_mtime)})")
    else:
        text.append("Last file: —")
    await update.effective_message.reply_html("\n".join(text), reply_markup=kb_home())


async def on_set_threshold(update: Update, context: CallbackContext):
    val = None
    if context.args:
        val = parse_float(context.args[0])
    if val is None:
        await update.effective_message.reply_html(
            "Пришли число от 0 до 1, например: <code>/set_threshold 0.35</code>",
            reply_markup=kb_home(),
        )
        return
    val = max(0.0, min(1.0, val))
    STATE["threshold"] = val
    await update.effective_message.reply_html(fmt_ok(f"Threshold = {val:.2f}"), reply_markup=kb_home())


async def on_set_topn(update: Update, context: CallbackContext):
    n = None
    if context.args:
        try:
            n = int(context.args[0])
        except Exception:
            n = None
    if n is None or n <= 0:
        await update.effective_message.reply_html(
            "Пришли положительное число, например: <code>/set_notify_top_n 20</code>",
            reply_markup=kb_home(),
        )
        return
    STATE["top_n"] = n
    await update.effective_message.reply_html(fmt_ok(f"Top-N = {n}"), reply_markup=kb_home())


async def on_schedule(update: Update, context: CallbackContext):
    if len(context.args) != 2:
        await update.effective_message.reply_html(
            "Формат: <code>/schedule_every &lt;days&gt; &lt;HH:MM&gt;</code>",
            reply_markup=kb_home(),
        )
        return
    try:
        days = int(context.args[0])
        hh, mm = map(int, context.args[1].split(":"))
    except Exception:
        await update.effective_message.reply_html(
            "Формат: <code>/schedule_every 7 10:00</code>",
            reply_markup=kb_home(),
        )
        return

    if SCHED.running:
        SCHED.remove_all_jobs()
    else:
        SCHED.start()

    async def job():
        try:
            await _run_core(update, context, silent=True)
        except Exception:
            log.exception("Scheduled job failed")

    SCHED.add_job(job, "interval", days=days, next_run_time=None, hour=hh, minute=mm)
    await update.effective_message.reply_html(
        fmt_ok(f"Периодический запуск раз в {days} дн., в {hh:02d}:{mm:02d}"),
        reply_markup=kb_home(),
    )


async def on_cancel_schedule(update: Update, context: CallbackContext):
    if SCHED.running:
        SCHED.remove_all_jobs()
    await update.effective_message.reply_html(fmt_ok("Расписание выключено"), reply_markup=kb_home())


async def on_clear_uploads(update: Update, context: CallbackContext):
    cnt = 0
    for p in UPLOAD_DIR.glob("*"):
        try:
            p.unlink()
            cnt += 1
        except Exception:
            pass
    await update.effective_message.reply_html(
        fmt_ok(f"Удалено файлов: {cnt}"), reply_markup=kb_home()
    )


async def on_document(update: Update, context: CallbackContext):
    doc = update.message.document
    if not doc:
        return
    if not any(doc.file_name.lower().endswith(ext) for ext in (".csv", ".xlsx", ".xls")):
        await update.message.reply_html(
            fmt_warn("Поддерживаются только CSV/XLSX"), reply_markup=kb_home()
        )
        return

    f = await doc.get_file()
    ts_name = f"{datetime.utcnow():%Y-%m-%dT%H-%M-%S}_{doc.file_name}"
    dest = UPLOAD_DIR / ts_name
    await f.download_to_drive(str(dest))
    await update.message.reply_html(
        fmt_ok("Файл сохранён:")
        + f"\n<code>{dest}</code>"
        + "\n\nМожешь нажать «Запустить /run».",
        reply_markup=kb_home(),
    )


async def _run_core(update: Update, context: CallbackContext, silent: bool = False):
    thr = STATE["threshold"]
    topn = STATE["top_n"]

    async def work():
        if USE_EXTERNAL_RUNNER:
            # подключить внешний раннер при необходимости
            res, _ = await run_external_runner(thr, topn)
        else:
            res, _ = run_builtin_scoring(thr, topn)
        return res

    if silent:
        # для расписания без спиннера
        try:
            res = await work()
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=res,
                parse_mode=constants.ParseMode.HTML,
                reply_markup=kb_home(),
            )
        except Exception as e:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=fmt_err("Ошибка в расписании") + f"\n<code>{e}</code>",
                parse_mode=constants.ParseMode.HTML,
                reply_markup=kb_home(),
            )
        return

    await run_with_progress(update, context, "Запускаю скоринг", work)


async def on_run(update: Update, context: CallbackContext):
    await _run_core(update, context, silent=False)


# -------------------- Inline buttons --------------------

async def on_cb(update: Update, context: CallbackContext):
    q = update.callback_query
    await q.answer()
    data = q.data

    # «домой»
    if data == "HOME":
        await q.edit_message_text(
            "Выбирай действие:", parse_mode=constants.ParseMode.HTML, reply_markup=kb_home()
        )
        return

    if data == "RUN":
        # запускаем обычный /run
        fake_update = Update(update.update_id, message=q.message)  # reuse chat
        await on_run(fake_update, context)
        return

    if data == "STATUS":
        fake_update = Update(update.update_id, message=q.message)
        await on_status(fake_update, context)
        return

    if data == "HELP":
        fake_update = Update(update.update_id, message=q.message)
        await on_help(fake_update, context)
        return

    if data == "CLEAR":
        fake_update = Update(update.update_id, message=q.message)
        await on_clear_uploads(fake_update, context)
        return

    if data == "SET_THR":
        await q.edit_message_text(
            "Введи команду, например: <code>/set_threshold 0.30</code>",
            parse_mode=constants.ParseMode.HTML,
            reply_markup=kb_back(),
        )
        return

    if data == "SET_TOPN":
        await q.edit_message_text(
            "Введи команду, например: <code>/set_notify_top_n 20</code>",
            parse_mode=constants.ParseMode.HTML,
            reply_markup=kb_back(),
        )
        return


# -------------------- Errors --------------------

async def on_error(update: object, context: CallbackContext) -> None:
    log.exception("Unhandled error")
    try:
        chat_id = None
        if isinstance(update, Update) and update.effective_chat:
            chat_id = update.effective_chat.id
        if chat_id:
            await context.bot.send_message(
                chat_id=chat_id,
                text=fmt_err("Непредвиденная ошибка")
                + "\n\n"
                + f"<code>{type(context.error).__name__}: {context.error}</code>",
                parse_mode=constants.ParseMode.HTML,
                reply_markup=kb_home(),
            )
    except Exception:
        pass


# -------------------- Main --------------------

def main():
    log.info("Bot started.")
    app: Application = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("version", on_version))
    app.add_handler(CommandHandler("status", on_status))
    app.add_handler(CommandHandler("run", on_run))
    app.add_handler(CommandHandler("set_threshold", on_set_threshold))
    app.add_handler(CommandHandler("set_notify_top_n", on_set_topn))
    app.add_handler(CommandHandler("schedule_every", on_schedule))
    app.add_handler(CommandHandler("cancel_schedule", on_cancel_schedule))
    app.add_handler(CommandHandler("clear_uploads", on_clear_uploads))

    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(CallbackQueryHandler(on_cb))

    app.add_error_handler(on_error)

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
