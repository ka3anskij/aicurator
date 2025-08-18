# LMS Churn Bot (Render Worker, spinner & errors)

## Files
- `Dockerfile` — container image
- `render.yaml` — Render blueprint (type: worker)
- `requirements.txt` — deps
- `tg_churn_bot.py` — Telegram bot with live spinner and rich error reporting
- `lms_churn_prod.py` — runner (features + model or heuristic fallback)
- `prod_lms_dropout_config.json` — thresholds & windows
- `lms_events_sample.csv` — sample export for quick test

## Deploy (Render)
1) Push these files to a repo.
2) On Render: **New → Blueprint** → select repo.
3) Set env var **TG_TOKEN** (BotFather token).
4) Deploy — service type will be **Worker** (no port binding).
5) In Telegram: `/start`, upload `lms_events_sample.csv`, then `/run`.

## Notes
- On free plan we write into `/tmp/uploads` (ephemeral); files are lost on re-deploy. For persistence use S3/Backblaze or paid Render Disk.
- If you already have a trained model, place it as `prod_lms_dropout_model.pkl` at project root; runner will use it automatically; otherwise it uses a reasonable heuristic.
