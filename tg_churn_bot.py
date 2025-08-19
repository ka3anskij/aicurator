# tg_churn_bot.py
# Telegram-бот для алёртов по оттоку LMS.
# v2.2 — autodetect(schema), robust events→features, UX spinner & nice errors

import os
import io
import zipfile
import traceback
from datetime import datetime, timedelta

import pandas as pd
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackContext,
)

# ----------------------- CONFIG -----------------------
TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
UPLOAD_DIR = os.getenv("DATA_DIR", "./uploads")
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()  # опционально: путь к .pkl
CLEAR_ON_NEW_UPLOAD = os.getenv("CLEAR_ON_NEW_UPLOAD", "false").lower() in {"1", "true", "yes"}

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.35"))
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "20"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------- UX: Keyboards ----------------
def make_keyboard(has_files: bool) -> ReplyKeyboardMarkup:
    """
    Показываем 'Запустить /run' только если есть валидные выгрузки.
    """
    row1 = []
    if has_files:
        row1.append(KeyboardButton("🚀 Запустить /run"))
    row1.append(KeyboardButton("📊 Статус"))

    row2 = [KeyboardButton("⚙️ Порог риска"), KeyboardButton("🥇 Топ-N")]
    row3 = [KeyboardButton("🧹 Очистить выгрузки"), KeyboardButton("🆘 Помощь")]

    rows = []
    if row1:
        rows.append(row1)
    rows.append(row2)
    rows.append(row3)
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# ----------------------- Schema detection -------------
EMAIL_SYNONYMS = ["email", "user_id", "login", "mail", "e-mail", "user"]

def _find_email_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for key in EMAIL_SYNONYMS:
        if key in low:
            return low[key]
    # эвристика по содержимому
    for c in cols:
        try:
            if df[c].astype(str).str.contains("@").mean() > 0.5:
                return c
        except Exception:
            pass
    return None

def _find_ts_col(df: pd.DataFrame) -> str | None:
    candidates = ["timestamp", "time", "ts", "event_time", "created_at", "last_event_at"]
    low = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in low:
            return low[key]
    # эвристика по типу
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            pass
    return None

def detect_schema(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    # уже готовые признаки
    if {"email", "last_event_at", "events_28d", "quiz_avg"}.issubset(cols):
        return "features"
    # вероятный лог событий
    if _find_email_col(df) and _find_ts_col(df):
        return "events"
    return "unknown"

def build_features_from_events(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(dfs, ignore_index=True)

    email_col = _find_email_col(raw)
    ts_col    = _find_ts_col(raw)
    if not email_col or not ts_col:
        raise ValueError("Не нашёл колонки email/timestamp в событиях.")

    raw = raw.rename(columns={email_col: "email", ts_col: "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["email", "timestamp"])

    # референс — max(timestamp) в выгрузке (устойчиво для тестов/старых дампов)
    ref_now = raw["timestamp"].max()
    win_from = ref_now - timedelta(days=28)

    last_event = raw.groupby("email")["timestamp"].max().rename("last_event_at")
    events_28d = (
        raw[raw["timestamp"].between(win_from, ref_now)]
        .groupby("email")["timestamp"].count()
        .rename("events_28d")
    )

    # квизы (если есть)
    evt_col = next((c for c in ["event_type", "event", "type"] if c in raw.columns), None)
    score_col = next((c for c in ["score", "quiz_score"] if c in raw.columns), None)

    if evt_col and score_col:
        quiz_mask = raw[evt_col].astype(str).str.lower().isin(["quiz", "test", "exam"])
        quiz_avg = (
            raw[quiz_mask]
            .groupby("email")[score_col].mean()
            .round(2)
            .rename("quiz_avg")
        )
    else:
        quiz_avg = pd.Series(dtype=float, name="quiz_avg")

    # безопасная сборка
    all_emails = pd.Index(sorted(raw["email"].unique()), name="email")
    out = pd.concat([last_event, events_28d, quiz_avg], axis=1).reindex(all_emails)
    out = out.reset_index()

    out["events_28d"] = out["events_28d"].fillna(0).astype(int)
    out["quiz_avg"]   = out["quiz_avg"].fillna(0)
    out["last_event_at"] = pd.to_datetime(out["last_event_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out[["email", "last_event_at", "events_28d", "quiz_avg"]]


# ----------------------- File loading -----------------
def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

def _read_xlsx_bytes(b: bytes) -> list[pd.DataFrame]:
    xl = pd.ExcelFile(io.BytesIO(b))
    return [xl.parse(s) for s in xl.sheet_names]

def load_tables_from_path(path: str) -> list[pd.DataFrame]:
    """
    Поддержка CSV / XLSX / ZIP (внутри — csv/xlsx).
    Возвращает список датафреймов.
    """
    path_low = path.lower()
    dfs: list[pd.DataFrame] = []

    if path_low.endswith(".csv"):
        dfs.append(pd.read_csv(path))
    elif path_low.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(path)
        for s in xl.sheet_names:
            dfs.append(xl.parse(s))
    elif path_low.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                nlow = name.lower()
                with zf.open(name) as f:
                    data = f.read()
                if nlow.endswith(".csv"):
                    dfs.append(_read_csv_bytes(data))
                elif nlow.endswith((".xlsx", ".xls")):
                    dfs.extend(_read_xlsx_bytes(data))
    else:
        raise ValueError(f"Неизвестный формат файла: {os.path.basename(path)}")

    # нормализация заголовков
    norm = []
    for d in dfs:
        d = d.copy()
        d.columns = [str(c).strip() for c in d.columns]
        norm.append(d)
    return norm


# ----------------------- Scoring ----------------------
_model = None
def _load_model_if_any():
    global _model
    if _model is not None:
        return
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
        except Exception:
            _model = None

def _days_inactive(last_event_at: str, ref: datetime | None = None) -> int:
    try:
        dt = pd.to_datetime(last_event_at)
        ref = ref or datetime.utcnow()
        return max(0, (ref - dt.to_pydatetime()).days)
    except Exception:
        return 9999

def rule_based_score(df: pd.DataFrame) -> pd.Series:
    """
    Аккуратный бейзлайн, если нет модели.
    p ~ от 0 до 1. Учитываем:
      - дни без активности,
      - малое число событий за 28д,
      - низкий средний балл квизов.
    """
    ref_now = datetime.utcnow()
    days = df["last_event_at"].apply(lambda s: _days_inactive(s, ref_now))
    events = df.get("events_28d", pd.Series([0]*len(df)))
    quiz = df.get("quiz_avg", pd.Series([0]*len(df))).fillna(0)

    # нормировки / веса
    p = (
        (days.clip(0, 60) / 60) * 0.6 +
        ((10 - events.clip(0, 10)) / 10) * 0.3 +
        ((100 - quiz.clip(0, 100)) / 100) * 0.1
    )
    return p.clip(0, 1)

def predict_scores(features: pd.DataFrame) -> pd.Series:
    _load_model_if_any()
    if _model is not None:
        # ожидаем порядок: нужно подогнать к признакам модели, если используешь pkl
        # для универсальности — fallback к бейзлайну при ошибке
        try:
            X = features.copy()
            # возможные приведения типов
            X["events_28d"] = X["events_28d"].astype(float)
            X["quiz_avg"]   = X["quiz_avg"].astype(float)
            # last_event_at → days_inactive
            X["days_inactive"] = X["last_event_at"].apply(lambda s: _days_inactive(s))
            use_cols = [c for c in ["days_inactive", "events_28d", "quiz_avg"] if c in X.columns]
            p = pd.Series(_model.predict_proba(X[use_cols])[:, 1], index=features.index)
            return p.clip(0, 1)
        except Exception:
            pass
    return rule_based_score(features)


# ----------------------- Helpers ----------------------
def has_any_valid_upload() -> bool:
    files = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith(".")]
    return len(files) > 0

def list_uploaded_paths() -> list[str]:
    paths = []
    for f in sorted(os.listdir(UPLOAD_DIR)):
        if f.startswith("."):
            continue
        paths.append(os.path.join(UPLOAD_DIR, f))
    return paths

def clear_uploads():
    for p in list_uploaded_paths():
        try:
            os.remove(p)
        except Exception:
            pass

def short_trace(exc: Exception) -> str:
    tb = traceback.format_exc(limit=4)
    return "\n".join(tb.strip().splitlines()[-4:])


# ----------------------- Handlers ---------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    has_files = has_any_valid_upload()
    await update.message.reply_text(
        "Привет! Я бот алёртов по оттоку LMS.\n"
        "Пришли CSV/XLSX выгрузки, я их сохраню.\n\n"
        "Нажимай кнопки ниже или используй команды.",
        reply_markup=make_keyboard(has_files),
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Команды и кнопки:\n"
        "• 🚀 Запустить /run — скоринг по последним выгрузкам\n"
        "• 📊 Статус — конфиг и текущая папка\n"
        "• ⚙️ Порог риска — установить, например 0.35\n"
        "• 🥇 Топ-N — сколько верхних показывать\n"
        "• 🧹 Очистить выгрузки — убрать все загруженные файлы\n\n"
        "Форматы данных:\n"
        "A) features CSV/XLSX: email,last_event_at,events_28d,quiz_avg\n"
        "B) events CSV/XLSX/ZIP: user_id/email + timestamp + (event_type,score) — агрегирую сам."
    )
    await update.message.reply_text(txt, reply_markup=make_keyboard(has_any_valid_upload()))

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thresh = context.chat_data.get("threshold", DEFAULT_THRESHOLD)
    topn = context.chat_data.get("topn", DEFAULT_TOP_N)
    files = "\n".join([f"• {os.path.basename(p)}" for p in list_uploaded_paths()]) or "—"
    model = MODEL_PATH or "—"
    txt = (
        f"🔧 Статус:\n"
        f"Порог риска: {thresh}\nТоп-N: {topn}\n"
        f"Модель: {model}\n"
        f"Папка выгрузок: {UPLOAD_DIR}\n"
        f"Файлы:\n{files}"
    )
    await update.message.reply_text(txt, reply_markup=make_keyboard(has_any_valid_upload()))

async def set_threshold_btn(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Введи новое значение порога (например 0.35):",
        reply_markup=make_keyboard(has_any_valid_upload()),
    )
    context.chat_data["await_threshold"] = True

async def set_topn_btn(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Введи новое значение Top-N (например 20):",
        reply_markup=make_keyboard(has_any_valid_upload()),
    )
    context.chat_data["await_topn"] = True

async def clean_btn(update: Update, context: CallbackContext):
    clear_uploads()
    await update.message.reply_text("🧹 Очищено файлов: 1\nПришли новые выгрузки.", reply_markup=make_keyboard(False))

async def handle_text(update: Update, context: CallbackContext):
    text = (update.message.text or "").strip()

    # ожидание числовых значений
    if context.chat_data.pop("await_threshold", False):
        try:
            v = float(text.replace(",", "."))
            context.chat_data["threshold"] = v
            await update.message.reply_text(f"Ок, порог риска = {v}", reply_markup=make_keyboard(has_any_valid_upload()))
        except Exception:
            await update.message.reply_text("Не понял число. Пример: 0.35", reply_markup=make_keyboard(has_any_valid_upload()))
        return

    if context.chat_data.pop("await_topn", False):
        try:
            v = int(text)
            context.chat_data["topn"] = v
            await update.message.reply_text(f"Ок, Top-N = {v}", reply_markup=make_keyboard(has_any_valid_upload()))
        except Exception:
            await update.message.reply_text("Не понял число. Пример: 20", reply_markup=make_keyboard(has_any_valid_upload()))
        return

    # кнопки/команды
    low = text.lower()
    if "запустить" in low or low.startswith("/run"):
        await run_cmd(update, context)
    elif "статус" in low or low.startswith("/status"):
        await status_cmd(update, context)
    elif "порог" in low:
        await set_threshold_btn(update, context)
    elif "топ" in low:
        await set_topn_btn(update, context)
    elif "очист" in low:
        await clean_btn(update, context)
    elif "помощ" in low or low.startswith("/help"):
        await help_cmd(update, context)
    else:
        await update.message.reply_text("Не понял. Нажми кнопку или /help.", reply_markup=make_keyboard(has_any_valid_upload()))

async def handle_document(update: Update, context: CallbackContext):
    try:
        if CLEAR_ON_NEW_UPLOAD:
            clear_uploads()
        doc = update.message.document
        if not doc:
            await update.message.reply_text("Пришли CSV/XLSX/ZIP файл.")
            return
        file = await context.bot.get_file(doc.file_id)
        fn = doc.file_name or f"upload_{int(datetime.utcnow().timestamp())}"
        save_path = os.path.join(UPLOAD_DIR, fn)
        await file.download_to_drive(save_path)
        await update.message.reply_text(
            f"✅ Файл сохранён: {save_path}\n"
            f"Можешь нажать «Запустить /run».",
            reply_markup=make_keyboard(True),
        )
    except Exception as e:
        await update.message.replyText(f"🔴 Ошибка при сохранении: {e}\n{short_trace(e)}", reply_markup=make_keyboard(has_any_valid_upload()))

async def run_cmd(update: Update, context: CallbackContext):
    files = list_uploaded_paths()
    if not files:
        await update.message.reply_text("Сначала пришли выгрузку (CSV/XLSX/ZIP).", reply_markup=make_keyboard(False))
        return

    # спиннер
    msg = await update.message.reply_text("⏳ Запускаю скоринг…", reply_markup=make_keyboard(True))

    try:
        # читаем все таблицы
        all_dfs: list[pd.DataFrame] = []
        for p in files:
            all_dfs.extend(load_tables_from_path(p))

        # определяем схемы и готовим признаки
        schemas = [detect_schema(df) for df in all_dfs]
        if any(k == "features" for k in schemas):
            feats = [df for df, k in zip(all_dfs, schemas) if k == "features"]
            features = pd.concat(feats, ignore_index=True)
            await context.bot.edit_message_text(
                chat_id=msg.chat_id, message_id=msg.message_id,
                text="🔎 Найдены готовые признаки. Идёт скоринг…"
            )
        elif any(k == "events" for k in schemas):
            evts = [df for df, k in zip(all_dfs, schemas) if k == "events"]
            await context.bot.edit_message_text(
                chat_id=msg.chat_id, message_id=msg.message_id,
                text="🔨 Обнаружен лог событий. Строю признаки…"
            )
            features = build_features_from_events(evts)
        else:
            await context.bot.edit_message_text(
                chat_id=msg.chat_id, message_id=msg.message_id,
                text=("🔴 Не распознал формат выгрузки.\n"
                      "Нужен либо features-CSV: email,last_event_at,events_28d,quiz_avg;\n"
                      "либо events-лог: email/user_id + timestamp (+event_type,score).")
            )
            return

        # скоринг
        await context.bot.edit_message_text(
            chat_id=msg.chat_id, message_id=msg.message_id,
            text="🤖 Считаю вероятности…"
        )
        p = predict_scores(features)
        features = features.copy()
        features["p"] = p

        # параметры вывода
        threshold = context.chat_data.get("threshold", DEFAULT_THRESHOLD)
        topn = context.chat_data.get("topn", DEFAULT_TOP_N)

        risky = features.sort_values("p", ascending=False)
        top_rows = risky.head(topn)

        # форматируем вывод
        alerts = []
        cnt_alerts = 0
        ref_now = datetime.utcnow()
        for _, r in top_rows.iterrows():
            email = str(r.get("email", "—"))
            prob = float(r["p"])
            days = _days_inactive(r.get("last_event_at", ""), ref_now)
            events_28 = int(r.get("events_28d", 0))
            quiz = r.get("quiz_avg", 0)
            reasons = []
            if days >= 20: reasons.append(f"нет активности {days}дн")
            if events_28 <= 3: reasons.append(f"мало событий за 28д ({events_28})")
            if (isinstance(quiz, (int, float)) and quiz <= 60): reasons.append(f"низкие квизы (avg ~{int(quiz)})")
            line = f"{email} — p={prob:.2f} — " + ("; ".join(reasons) if reasons else "сигналов мало")
            if prob >= threshold:
                cnt_alerts += 1
                alerts.append(line)

        head = f"🔔 Всего алёртов: {cnt_alerts}/{len(top_rows)}"
        body = "\n".join([f"{i+1}. {s}" for i, s in enumerate(alerts)]) if alerts else "нет"
        final = f"{head}\n{body}"
        await context.bot.edit_message_text(chat_id=msg.chat_id, message_id=msg.message_id, text=final)

    except Exception as e:
        txt = f"🔴 Ошибка во время обработки: {e}\n{short_trace(e)}"
        try:
            await context.bot.edit_message_text(chat_id=msg.chat_id, message_id=msg.message_id, text=txt)
        except Exception:
            await update.message.reply_text(txt, reply_markup=make_keyboard(has_any_valid_upload()))


# ----------------------- App bootstrap ----------------
def main():
    if not TG_TOKEN:
        raise RuntimeError("Set TG_TOKEN env var.")

    app = ApplicationBuilder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("run", run_cmd))

    # кнопки
    app.add_handler(MessageHandler(filters.Regex("^⚙️ Порог риска$"), set_threshold_btn))
    app.add_handler(MessageHandler(filters.Regex("^🥇 Топ-N$"), set_topn_btn))
    app.add_handler(MessageHandler(filters.Regex("^🧹 Очистить выгрузки$"), clean_btn))
    app.add_handler(MessageHandler(filters.Regex("^🆘 Помощь$"), help_cmd))

    # документы и текст
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
