import os
import io
import re
import csv
import sys
import json
import math
import zipfile
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_numeric_dtype

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# =========================
# CONFIG & GLOBALS
# =========================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

TG_TOKEN = os.getenv("TG_TOKEN")
if not TG_TOKEN:
    raise RuntimeError("Set TG_TOKEN env var.")

UPLOADS_DIR = os.getenv("DATA_DIR", "/tmp/uploads")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/prod_lms_dropout_model.pkl")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/prod_lms_dropout_config.json")

# runtime конфиг (хранится в памяти процесса)
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.35"))
NOTIFY_TOP_N = int(os.getenv("NOTIFY_TOP_N", "20"))

os.makedirs(UPLOADS_DIR, exist_ok=True)

EMAIL_PAT = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", re.I)

COL_SYNONYMS = {
    "email": ["email", "e-mail", "mail", "почта", "user_email", "login_email", "username"],
    "user_id": ["user_id", "userid", "id", "uid", "account_id"],
    "last_event_at": [
        "last_event_at", "last_activity", "last_seen", "updated_at",
        "last_event_time", "последняя_активность", "last_login"
    ],
    "events_28d": ["events_28d", "events28", "events_last_28d", "activity_28d", "count_28d", "activity_count_28d"],
    "quiz_avg": ["quiz_avg", "avg_quiz", "quiz_score", "score_avg", "avg_score", "mean_quiz"],
}

REQUIRED_ROLES = ["email", "last_event_at"]
OPTIONAL_ROLES = ["events_28d", "quiz_avg", "user_id"]


# =========================
# UTILS: FILES & DATA
# =========================

def has_data() -> bool:
    if not os.path.isdir(UPLOADS_DIR):
        return False
    for _, _, files in os.walk(UPLOADS_DIR):
        for f in files:
            if f.lower().endswith((".csv", ".xlsx", ".xls", ".zip")):
                return True
    return False


def _normalize_columns(cols):
    norm = []
    for c in cols:
        x = str(c).strip().lower()
        x = x.replace(" ", "_").replace("-", "_")
        norm.append(x)
    return norm


def read_any_table(path: str) -> List[pd.DataFrame]:
    """
    Возвращает список DF из файла (CSV/XLSX/ZIP).
    """
    dfs: List[pd.DataFrame] = []

    def _read_csv_bytes(b: bytes) -> Optional[pd.DataFrame]:
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            for sep in (",", ";", "\t", "|"):
                try:
                    df = pd.read_csv(io.BytesIO(b), encoding=enc, sep=sep, engine="python")
                    if df.shape[1] > 0:
                        return df
                except Exception:
                    continue
        return None

    low = path.lower()

    if low.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df.columns = _normalize_columns(df.columns)
            dfs.append(df)
        return dfs

    if low.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                nlow = name.lower()
                if not nlow.endswith((".csv", ".xlsx", ".xls")):
                    continue
                data = z.read(name)
                if nlow.endswith((".xlsx", ".xls")):
                    bio = io.BytesIO(data)
                    xls = pd.ExcelFile(bio)
                    for sheet in xls.sheet_names:
                        df = xls.parse(sheet)
                        df.columns = _normalize_columns(df.columns)
                        dfs.append(df)
                else:
                    df = _read_csv_bytes(data)
                    if df is not None:
                        df.columns = _normalize_columns(df.columns)
                        dfs.append(df)
        return dfs

    # csv
    with open(path, "rb") as f:
        b = f.read()
    df = _read_csv_bytes(b)
    if df is not None:
        df.columns = _normalize_columns(df.columns)
        dfs.append(df)
    return dfs


def _guess_col_by_synonyms(df: pd.DataFrame, role: str) -> Optional[str]:
    syns = COL_SYNONYMS.get(role, [])
    for s in syns:
        if s in df.columns:
            return s
    return None


def _guess_email_col(df: pd.DataFrame) -> Optional[str]:
    col = _guess_col_by_synonyms(df, "email")
    if col:
        return col
    for c in df.columns:
        series = df[c].astype(str).str.strip()
        sample = series.dropna().head(200)
        if not len(sample):
            continue
        m = (sample.str.match(EMAIL_PAT, na=False)).mean()
        if m >= 0.5:
            return c
    return None


def _coerce_datetime(series: pd.Series) -> pd.Series:
    try:
        s = pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)
        return s
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def _guess_datetime_col(df: pd.DataFrame) -> Optional[str]:
    col = _guess_col_by_synonyms(df, "last_event_at")
    if col:
        return col
    best, best_rate = None, 0
    for c in df.columns:
        s = _coerce_datetime(df[c])
        rate = s.notna().mean()
        if rate > 0.5 and rate > best_rate:
            best, best_rate = c, rate
    return best


def _guess_numeric_col(df: pd.DataFrame, role: str) -> Optional[str]:
    col = _guess_col_by_synonyms(df, role)
    if col:
        return col
    best, best_rate = None, 0
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        rate = s.notna().mean()
        if rate > 0.5 and rate > best_rate:
            best, best_rate = c, rate
    return best


def infer_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {
        "email": _guess_email_col(df),
        "last_event_at": _guess_datetime_col(df),
        "events_28d": _guess_numeric_col(df, "events_28d"),
        "quiz_avg": _guess_numeric_col(df, "quiz_avg"),
        "user_id": _guess_numeric_col(df, "user_id") or _guess_col_by_synonyms(df, "user_id"),
    }
    return mapping


def validate_mapping(mapping: Dict[str, Optional[str]]):
    missing = [r for r in REQUIRED_ROLES if not mapping.get(r)]
    return (len(missing) == 0, missing)


def normalize_df(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["email"] = df[mapping["email"]].astype(str).str.strip()
    out["last_event_at"] = _coerce_datetime(df[mapping["last_event_at"]])
    if mapping.get("events_28d"):
        out["events_28d"] = pd.to_numeric(df[mapping["events_28d"]], errors="coerce").fillna(0)
    else:
        out["events_28d"] = 0
    if mapping.get("quiz_avg"):
        q = pd.to_numeric(df[mapping["quiz_avg"]], errors="coerce")
        out["quiz_avg"] = q.apply(lambda x: x/100 if 1 < x <= 1000 else x).fillna(0).clip(0, 100)
    else:
        out["quiz_avg"] = 0
    if mapping.get("user_id"):
        out["user_id"] = df[mapping["user_id"]].astype(str)
    else:
        out["user_id"] = out["email"]

    # фильтры
    out = out[out["email"].str.match(EMAIL_PAT, na=False)]
    out = out[out["last_event_at"].notna()]
    out = out[out["last_event_at"] <= pd.Timestamp.now(tz=None)]
    return out


def load_current_batch(upload_dir: str) -> Optional[pd.DataFrame]:
    all_paths = []
    for root, _, files in os.walk(upload_dir):
        for f in files:
            if f.lower().endswith((".csv", ".xlsx", ".xls", ".zip")):
                all_paths.append(os.path.join(root, f))
    if not all_paths:
        return None

    parts: List[pd.DataFrame] = []
    for p in sorted(all_paths):
        try:
            for df in read_any_table(p):
                mapping = infer_schema(df)
                ok, missing = validate_mapping(mapping)
                if not ok:
                    continue
                df_norm = normalize_df(df, mapping)
                if len(df_norm):
                    parts.append(df_norm)
        except Exception as e:
            log.exception("Failed to read %s: %s", p, e)
            continue

    if not parts:
        return None

    big = pd.concat(parts, ignore_index=True)
    agg = (
        big.sort_values("last_event_at")
           .groupby("email", as_index=False)
           .agg({
               "user_id": "last",
               "last_event_at": "max",
               "events_28d": "max",
               "quiz_avg": "mean",
           })
    )
    return agg


# =========================
# SCORING (fallback + модель)
# =========================

_model = None

def try_load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.isfile(MODEL_PATH):
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
            log.info("Model loaded: %s", MODEL_PATH)
            return _model
        except Exception as e:
            log.warning("Can't load model %s: %s. Using heuristic fallback.", MODEL_PATH, e)
    return None


def heuristic_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    days_since -> чем больше, тем риск выше.
    мало events_28d -> риск выше.
    низкий quiz_avg -> риск выше.
    """
    now = pd.Timestamp.now(tz=None)
    days = (now - df["last_event_at"]).dt.days.clip(lower=0).astype(float)

    # нормировки
    d = (days / 60.0).clip(0, 1)                  # 60+ дней = max
    e = (1.0 - (df["events_28d"] / 20.0).clip(0, 1))  # 0 событий = 1 (плохо), 20+ = 0
    q = (1.0 - (df["quiz_avg"] / 100.0).clip(0, 1))   # 0% = 1 (плохо), 100% = 0

    risk = 0.6 * d + 0.25 * e + 0.15 * q
    out = df.copy()
    out["risk"] = risk.clip(0, 1)

    out["why"] = []
    msgs = []
    for i in range(len(out)):
        reasons = []
        if days.iloc[i] >= 20:
            reasons.append(f"нет активности {int(days.iloc[i])}дн")
        if df["events_28d"].iloc[i] <= 2:
            reasons.append(f"мало событий за 28д ({int(df['events_28d'].iloc[i])})")
        if df["quiz_avg"].iloc[i] <= 60:
            reasons.append(f"низкие квизы (avg ~{int(df['quiz_avg'].iloc[i])})")
        msgs.append("; ".join(reasons) if reasons else "сигналов немного")
    out["why"] = msgs
    return out


def score_students(df: pd.DataFrame) -> pd.DataFrame:
    mdl = try_load_model()
    if mdl is None:
        return heuristic_score(df)
    # если есть модель, подготовим feature matrix
    now = pd.Timestamp.now(tz=None)
    X = pd.DataFrame({
        "days_since": (now - df["last_event_at"]).dt.days.clip(lower=0).astype(float),
        "events_28d": df["events_28d"].astype(float),
        "quiz_avg": df["quiz_avg"].astype(float),
    })
    try:
        p = mdl.predict_proba(X)[:, 1]
    except Exception as e:
        log.warning("Model failed, fallback to heuristic: %s", e)
        return heuristic_score(df)

    out = df.copy()
    out["risk"] = pd.Series(p).clip(0, 1)
    # краткое объяснение всё равно сгенерим из правил
    return heuristic_score(out) if "why" not in out.columns else out


# =========================
# UX: KEYBOARD & SPINNER
# =========================

def build_menu() -> InlineKeyboardMarkup:
    rows = []
    if has_data():
        rows.append([
            InlineKeyboardButton("▶️ Запустить /run", callback_data="run"),
            InlineKeyboardButton("📊 Статус", callback_data="status"),
        ])
        rows.append([
            InlineKeyboardButton("⚙️ Порог риска", callback_data="threshold"),
            InlineKeyboardButton("🏅 Top-N", callback_data="topn"),
        ])
        rows.append([
            InlineKeyboardButton("🧹 Очистить выгрузки", callback_data="clear"),
            InlineKeyboardButton("🆘 Помощь", callback_data="help"),
        ])
    else:
        rows.append([
            InlineKeyboardButton("📊 Статус", callback_data="status"),
            InlineKeyboardButton("🆘 Помощь", callback_data="help"),
        ])
        rows.append([
            InlineKeyboardButton("🧹 Очистить выгрузки", callback_data="clear"),
        ])
    return InlineKeyboardMarkup(rows)


async def spinner(context: ContextTypes.DEFAULT_TYPE, message: Message, stop_event: asyncio.Event, base_text: str):
    frames = ["⏳", "⏳.", "⏳..", "⏳..."]
    i = 0
    while not stop_event.is_set():
        try:
            await message.edit_text(f"{base_text}\n{frames[i % len(frames)]}", reply_markup=build_menu())
        except Exception:
            pass
        i += 1
        await asyncio.sleep(1.0)
    # финальное погашение делает вызывающий код


# =========================
# HANDLERS
# =========================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Привет! Я бот алёртов по оттоку LMS.\n"
        "Пришли CSV/XLSX/ZIP выгрузки — я их сохраню и подскажу, что дальше.\n\n"
        "Пока нет файлов — кнопка «Запустить» скрыта."
        if not has_data()
        else "Файлы есть — можно запустить скоринг."
    )
    await update.message.reply_text(text, reply_markup=build_menu())


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_state = "ok" if os.path.isfile(MODEL_PATH) else "fallback"
    cfg_state = "ok" if os.path.isfile(CONFIG_PATH) else "missing"
    exports = UPLOADS_DIR
    schedule = "None"  # если подключишь APScheduler cron — выведем здесь

    text = (
        "🧾 Статус:\n"
        f"Model: {os.path.basename(MODEL_PATH)} ({model_state})\n"
        f"Config: {os.path.basename(CONFIG_PATH)} ({cfg_state})\n"
        f"Exports: {exports}\n"
        f"Threshold: {RISK_THRESHOLD:.2f} • Top-N: {NOTIFY_TOP_N}\n"
        f"Schedule: {schedule}"
    )
    await update.message.reply_text(text, reply_markup=build_menu())


async def cmd_set_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global RISK_THRESHOLD
    try:
        val = float(context.args[0])
        assert 0.0 <= val <= 1.0
        RISK_THRESHOLD = val
        await update.message.reply_text(f"OK. Новый порог риска: {RISK_THRESHOLD:.2f}", reply_markup=build_menu())
    except Exception:
        await update.message.reply_text("Использование: /set_threshold <0..1>", reply_markup=build_menu())


async def cmd_set_topn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NOTIFY_TOP_N
    try:
        val = int(context.args[0])
        assert val >= 1
        NOTIFY_TOP_N = val
        await update.message.reply_text(f"OK. Новый Top-N: {NOTIFY_TOP_N}", reply_markup=build_menu())
    except Exception:
        await update.message.reply_text("Использование: /set_notify_top_n <N>", reply_markup=build_menu())


async def clear_uploads_dir() -> int:
    cnt = 0
    if os.path.isdir(UPLOADS_DIR):
        for root, _, files in os.walk(UPLOADS_DIR):
            for f in files:
                if f.lower().endswith((".csv", ".xlsx", ".xls", ".zip")):
                    try:
                        os.remove(os.path.join(root, f))
                        cnt += 1
                    except Exception:
                        pass
    return cnt


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = await clear_uploads_dir()
    await update.message.reply_text(f"🧹 Очищено файлов: {n}\nПришли новые выгрузки.", reply_markup=build_menu())


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return
    doc = update.message.document
    name = doc.file_name or "file"
    if not name.lower().endswith((".csv", ".xlsx", ".xls", ".zip")):
        await update.message.reply_text("Поддерживаю CSV/XLSX/ZIP.", reply_markup=build_menu())
        return

    try:
        file = await context.bot.get_file(doc.file_id)
        dest = os.path.join(UPLOADS_DIR, f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{name}")
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        await file.download_to_drive(dest)
        await update.message.reply_text(
            f"✅ Файл сохранён:\n{dest}\n"
            f"{'Можешь запускать /run.' if has_data() else 'Пришли ещё нужные файлы.'}",
            reply_markup=build_menu(),
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при сохранении: {type(e).__name__}: {e}", reply_markup=build_menu())


def _format_alerts_table(df: pd.DataFrame, top_n: int) -> str:
    lines = []
    total = len(df)
    show = min(top_n, total)
    lines.append(f"🔔 Всего алёртов: {show}/{total}")
    for i, row in enumerate(df.head(show).itertuples(index=False), 1):
        p = f"{row.risk:.2f}"
        email = row.email
        why = getattr(row, "why", "")
        lines.append(f"{i}. {email} — p={p} — {why}")
    return "\n".join(lines)


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not has_data():
        await update.message.reply_text(
            "Пока нет выгрузок — пришли CSV/XLSX/ZIP.\n"
            "Достаточно одного файла с колонками `email` и `last_event_at`.",
            reply_markup=build_menu(),
        )
        return

    # спиннер
    status_msg = await update.message.reply_text("Запускаю скоринг…", reply_markup=build_menu())
    stop_event = asyncio.Event()
    spin_task = asyncio.create_task(spinner(context, status_msg, stop_event, "Запускаю скоринг"))

    try:
        df = load_current_batch(UPLOADS_DIR)
        if df is None or df.empty:
            stop_event.set()
            await asyncio.sleep(0.1)
            await status_msg.edit_text(
                "❌ Не удалось определить структуру выгрузок.\n"
                "Проверь, что есть колонки вроде `email` и `last_event_at`, "
                "или пришли другой файл.",
                reply_markup=build_menu(),
            )
            return

        scored = score_students(df)
        # фильтр по порогу и сортировка
        alerts = (
            scored[scored["risk"] >= RISK_THRESHOLD]
            .sort_values("risk", ascending=False)
            .reset_index(drop=True)
        )

        text = _format_alerts_table(alerts, NOTIFY_TOP_N) if not alerts.empty else "✅ Рисковых студентов не найдено."
        stop_event.set()
        await asyncio.sleep(0.1)
        await status_msg.edit_text(text, reply_markup=build_menu())

    except Exception as e:
        log.exception("Run failed: %s", e)
        stop_event.set()
        await asyncio.sleep(0.1)
        await status_msg.edit_text(
            f"❌ Ошибка во время обработки: {type(e).__name__}: {e}",
            reply_markup=build_menu(),
        )


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    fake_update = Update(update.update_id, message=q.message)

    if data == "run":
        await cmd_run(fake_update, context)
    elif data == "status":
        await cmd_status(fake_update, context)
    elif data == "clear":
        await cmd_clear(fake_update, context)
    elif data == "threshold":
        await q.message.edit_text(
            f"Текущий порог риска: {RISK_THRESHOLD:.2f}\n"
            f"Поменять: `/set_threshold 0.4`", parse_mode="Markdown", reply_markup=build_menu()
        )
    elif data == "topn":
        await q.message.edit_text(
            f"Текущий Top-N: {NOTIFY_TOP_N}\n"
            f"Поменять: `/set_notify_top_n 30`", parse_mode="Markdown", reply_markup=build_menu()
        )
    elif data == "help":
        await q.message.edit_text(
            "Команды:\n"
            "/run — запустить скоринг по последним выгрузкам\n"
            "/status — текущая конфигурация\n"
            "/set_threshold <0..1>\n"
            "/set_notify_top_n <N>\n"
            "/clear_uploads — удалить загруженные файлы",
            reply_markup=build_menu(),
        )
    else:
        await q.message.edit_text("Неизвестная команда.", reply_markup=build_menu())


# =========================
# BOOT
# =========================

def main():
    log.info("Bot started.")
    application = (
        ApplicationBuilder()
        .token(TG_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("run", cmd_run))
    application.add_handler(CommandHandler("set_threshold", cmd_set_threshold))
    application.add_handler(CommandHandler("set_notify_top_n", cmd_set_topn))
    application.add_handler(CommandHandler("clear_uploads", cmd_clear))
    application.add_handler(CallbackQueryHandler(on_button))

    application.add_handler(MessageHandler(filters.Document.ALL, on_document))

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
