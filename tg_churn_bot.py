#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, logging, subprocess, asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

VERSION = "tg-bot: progress-enabled v2"

TG_TOKEN    = os.getenv("TG_TOKEN", "")
MODEL_PATH  = os.getenv("MODEL_PATH", "prod_lms_dropout_model.pkl")
CONFIG_PATH = os.getenv("CONFIG_PATH", "prod_lms_dropout_config.json")
DATA_DIR    = os.getenv("DATA_DIR", "./uploads")
RUNNER_PATH = os.getenv("RUNNER_PATH", "lms_churn_prod.py")
STATE_PATH  = os.getenv("STATE_PATH", "./bot_state.json")
LOCAL_TZ    = ZoneInfo("Europe/Amsterdam")

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"schedule": None}

def save_state(st):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def latest_exports_folder() -> str:
    if not os.path.isdir(DATA_DIR):
        return DATA_DIR
    subs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not subs:
        return DATA_DIR
    subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subs[0]

def run_scoring(input_folder: str, out_csv_path: str, timeout_sec: int = 900):
    cmd = [os.sys.executable, RUNNER_PATH, "--input_folder", input_folder, "--model", MODEL_PATH,
           "--config", CONFIG_PATH, "--out", out_csv_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return p.returncode, ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired as e:
        return -2, f"Timeout {timeout_sec}s\n{(e.stdout or '')}\n{(e.stderr or '')}".strip()
    except Exception as e:
        return -1, f"Subprocess error: {e}"

def preview_alerts(out_csv_path: str, limit=20) -> str:
    import pandas as pd
    if not os.path.exists(out_csv_path):
        return "alerts.csv не найден."
    try:
        df = pd.read_csv(out_csv_path)
    except Exception as e:
        return f"Не смог прочитать alerts.csv: {e}"
    if df.empty:
        return "алёртов нет (пусто)."
    if "prob_dropout_14d" not in df.columns:
        return "alerts.csv без prob_dropout_14d — проверь модель/конфиг."
    if "alert" in df.columns and df["alert"].sum() > 0:
        dfx = df[df["alert"]==1].copy(); title = f"🔔 Всего алёртов: {len(dfx)}/{len(df)}"
    else:
        dfx = df.copy(); title = f"ℹ️ По порогу нет — покажу топ-{limit} по вероятности"
    dfx = dfx.sort_values("prob_dropout_14d", ascending=False).head(limit)
    lines = [title]
    for i, r in enumerate(dfx.itertuples(index=False), start=1):
        user = getattr(r, "user_key", "?"); p = getattr(r, "prob_dropout_14d", 0.0)
        reason = getattr(r, "reason", "") if "reason" in df.columns else ""
        tail = f" — {reason}" if reason else ""
        lines.append(f"{i}. {user} — p={p:.2f}{tail}")
    return "\n".join(lines)

SPINNER = ["⏳","🕒","🕑","🕐","🕙","🕘","🕗","🕖","🕕","🕔","🕓"]
async def spinner(edit, title, stop_event: asyncio.Event, period=3.0):
    i=0
    while not stop_event.is_set():
        try: await edit(f"{SPINNER[i%len(SPINNER)]} {title} (идёт обработка…)")
        except Exception: pass
        await asyncio.sleep(period); i+=1

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот алёртов по оттоку LMS.\n"
        "Пришли CSV/XLSX выгрузки, я их сохраню.\n"
        "/run — скоринг по последним выгрузкам\n"
        "/status — конфиг и текущая папка\n"
        "/set_threshold <0.35>\n"
        "/set_notify_top_n <20>\n"
        "/schedule_every <days> <HH:MM>\n"
        "/cancel_schedule — отключить расписание\n"
        "/version — версия бота"
    )

async def version(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(VERSION)

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await start(update, ctx)

async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: cfg = json.load(f)
    st = load_state(); folder = latest_exports_folder()
    await update.message.reply_text("\n".join([
        "🧠 Статус:",
        f"Model: {MODEL_PATH}",
        f"Config: {CONFIG_PATH} {('(ok)' if cfg else '(not found)')}",
        f"Exports: {folder if folder else 'нет'}",
        f"Schedule: {st.get('schedule')}",
    ]))

async def set_threshold(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Использование: /set_threshold 0.35")
    try:
        thr = float(ctx.args[0])
        with open(CONFIG_PATH,"r",encoding="utf-8") as f: cfg=json.load(f)
        cfg["threshold"]=thr
        with open(CONFIG_PATH,"w",encoding="utf-8") as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
        await update.message.reply_text(f"✅ Порог обновлён: {thr:.3f}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def set_notify_top_n(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Использование: /set_notify_top_n 25")
    try:
        n=int(ctx.args[0])
        with open(CONFIG_PATH,"r",encoding="utf-8") as f: cfg=json.load(f)
        cfg["notify_top_n"]=n
        with open(CONFIG_PATH,"w",encoding="utf-8") as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
        await update.message.reply_text(f"✅ notify_top_n обновлён: {n}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message.document: return
    doc=update.message.document; name=doc.file_name or "upload.bin"
    ext=os.path.splitext(name)[1].lower()
    if ext not in [".csv",".xlsx"]:
        return await update.message.reply_text("Пришлите .csv или .xlsx файл.")
    day_folder=os.path.join(DATA_DIR, datetime.now(tz=LOCAL_TZ).strftime("%Y-%m-%d"))
    os.makedirs(day_folder, exist_ok=True)
    file_obj=await ctx.bot.get_file(doc.file_id)
    path=os.path.join(day_folder, name)
    await file_obj.download_to_drive(path)
    await update.message.reply_text(f"✅ Файл сохранён: {path}\nМожно запустить /run.")

async def run_now(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    folder=latest_exports_folder()
    if not os.path.exists(folder) or not os.listdir(folder):
        return await update.message.reply_text("Не нашёл выгрузок. Пришлите CSV/XLSX.")
    out_csv=os.path.join(folder,"alerts.csv")
    msg=await update.message.reply_text("⏳ Запускаю скоринг…")
    stop=asyncio.Event()
    async def edit(text):
        try: await msg.edit_text(text, parse_mode="HTML")
        except Exception: pass
    spin=asyncio.create_task(spinner(edit,"Скоринг",stop))
    try:
        rc,out=await asyncio.to_thread(run_scoring, folder, out_csv, 900)
        stop.set(); await asyncio.sleep(0.1)
        if rc!=0:
            snippet=(out[-1500:] if out else "без сообщения")
            return await edit(f"❌ Ошибка при скоринге (rc={rc}).\n<code>{snippet}</code>")
        preview=await asyncio.to_thread(preview_alerts,out_csv,20)
        await edit(preview)
    except Exception as e:
        stop.set(); await asyncio.sleep(0.1)
        await edit(f"❌ Непредвиденная ошибка: {e!r}")

scheduler=AsyncIOScheduler(timezone=str(LOCAL_TZ))

async def scheduled_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        folder=latest_exports_folder(); out_csv=os.path.join(folder,"alerts.csv")
        rc,out=await asyncio.to_thread(run_scoring, folder, out_csv, 900)
        chat_id=context.job.kwargs["chat_id"]
        if rc!=0:
            snip=(out[-1500:] if out else "без сообщения")
            return await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка планового запуска (rc={rc}).\n<code>{snip}</code>", parse_mode="HTML")
        preview=await asyncio.to_thread(preview_alerts,out_csv,20)
        await context.bot.send_message(chat_id=chat_id, text=preview)
    except Exception as e:
        chat_id=context.job.kwargs.get("chat_id")
        if chat_id: await context.bot.send_message(chat_id=chat_id, text=f"❌ Непредвиденная ошибка планировщика: {e!r}")

async def schedule_every(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args)!=2:
        return await update.message.reply_text("Использование: /schedule_every <days> <HH:MM>")
    try:
        days=int(ctx.args[0]); hh,mm=ctx.args[1].split(":"); hour=int(hh); minute=int(mm)
        st=load_state(); st["schedule"]={"days":days,"hour":hour,"minute":minute}; save_state(st)
        for j in scheduler.get_jobs(): j.remove()
        trigger=CronTrigger(hour=hour, minute=minute, timezone=str(LOCAL_TZ))
        scheduler.add_job(scheduled_job, trigger, kwargs={"chat_id": update.effective_chat.id}, id="churn_job")
        if not scheduler.running: scheduler.start()
        await update.message.reply_text(f"✅ Планировщик включён: каждые {days} дн. в {hour:02d}:{minute:02d} ({LOCAL_TZ.key})")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def cancel_schedule(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    for j in scheduler.get_jobs(): j.remove()
    st=load_state(); st["schedule"]=None; save_state(st)
    await update.message.reply_text("⏹️ Планировщик отключён.")

def main():
    token=TG_TOKEN or os.environ.get("TG_TOKEN")
    if not token: raise RuntimeError("Set TG_TOKEN env var.")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("set_threshold", set_threshold))
    app.add_handler(CommandHandler("set_notify_top_n", set_notify_top_n))
    app.add_handler(CommandHandler("run", run_now))
    app.add_handler(CommandHandler("schedule_every", schedule_every))
    app.add_handler(CommandHandler("cancel_schedule", cancel_schedule))
    app.add_handler(CommandHandler("version", version))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    logging.getLogger(__name__).info("Bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
