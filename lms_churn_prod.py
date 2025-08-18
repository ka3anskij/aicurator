#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder", required=True)
    ap.add_argument("--model", required=False, default="prod_lms_dropout_model.pkl")
    ap.add_argument("--config", required=False, default="prod_lms_dropout_config.json")
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def read_config(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"threshold": 0.30, "feature_window_days": 28, "horizon_days": 14, "notify_top_n": 20}

def load_tables(folder):
    dfs = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(folder, fn)))
        elif fn.lower().endswith(".xlsx"):
            try:
                dfs.append(pd.read_excel(os.path.join(folder, fn)))
            except Exception:
                pass
    if not dfs:
        raise RuntimeError("No CSV/XLSX files in input_folder")
    return dfs

def pick_col(cols, candidates):
    ll = [c.lower() for c in cols]
    for cand in candidates:
        if cand in ll:
            return cols[ll.index(cand)]
    return None

def build_features(dfs, feature_window_days=28):
    # concat and normalize
    df = pd.concat(dfs, ignore_index=True)
    # try to find user & timestamp
    user_col = pick_col(df.columns, ["user_id","user","email","login","username"])
    ts_col   = pick_col(df.columns, ["timestamp","ts","time","datetime","event_time"])
    if user_col is None or ts_col is None:
        raise RuntimeError("No event-like tables (need user & timestamp). Found columns: %s" % list(df.columns))
    # parse timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[ts_col])
    # filter last window
    now = pd.Timestamp.utcnow().tz_localize(None)
    start = now - pd.Timedelta(days=feature_window_days)
    dfw = df[df[ts_col] >= start].copy()

    # aggregates
    duration_col = pick_col(df.columns, ["duration_min","minutes","duration"])
    score_col    = pick_col(df.columns, ["score","quiz_score"])

    feats = []
    for user, g in df.groupby(user_col):
        g = g.sort_values(ts_col)
        last_ts = g[ts_col].max()
        days_since_last = (now - last_ts).days
        g28 = g[g[ts_col] >= start]
        events_28d = len(g28)
        unique_days_28d = g28[ts_col].dt.date.nunique()
        mean_score = g28[score_col].mean() if (score_col and score_col in g28) else np.nan
        total_dur = g28[duration_col].sum() if (duration_col and duration_col in g28) else np.nan
        feats.append({
            "user_key": user,
            "days_since_last": days_since_last,
            "events_28d": events_28d,
            "active_days_28d": unique_days_28d,
            "mean_score_28d": mean_score,
            "total_duration_28d": total_dur,
        })
    X = pd.DataFrame(feats).fillna(0)
    return X

def heuristic_prob(row):
    # combine inactivity & scarcity of events; clamp to [0,1]
    p = 0.0
    p += min(row["days_since_last"]/30.0, 1.2) * 0.6
    p += (1.0 - min(row["events_28d"]/10.0, 1.0)) * 0.25
    if "mean_score_28d" in row:
        low_score = max(0.0, (60 - row["mean_score_28d"]) / 60.0)  # if mean_score below 60 -> risk
        p += min(low_score, 1.0) * 0.15
    return float(max(0.0, min(p, 1.0)))

def explain(row):
    reasons = []
    if row["days_since_last"] >= 7:
        reasons.append(f"нет активности {int(row['days_since_last'])}дн")
    if row["events_28d"] <= 2:
        reasons.append(f"мало событий за 28д ({int(row['events_28d'])})")
    if row.get("mean_score_28d", 100) < 60:
        reasons.append(f"низкие квизы (avg ~{int(row['mean_score_28d'])})")
    if not reasons:
        reasons.append("паттерн близок к норме")
    return "; ".join(reasons)

def main():
    args = parse_args()
    cfg = read_config(args.config)
    dfs = load_tables(args.input_folder)
    feats = build_features(dfs, cfg.get("feature_window_days", 28))

    # predict
    probs = None
    if args.model and os.path.exists(args.model):
        try:
            model = joblib.load(args.model)
            # expect the model to have 'predict_proba' and the feature names
            # if mismatch, fall back to heuristic
            need_cols = getattr(model, "feature_names_in_", None)
            X = feats[need_cols] if need_cols is not None and set(need_cols).issubset(feats.columns) else feats
            probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"Model load/predict failed, fallback to heuristic: {e}", file=sys.stderr)
    if probs is None:
        probs = feats.apply(heuristic_prob, axis=1)

    out = feats.copy()
    out["prob_dropout_14d"] = probs
    thr = float(cfg.get("threshold", 0.30))
    out["alert"] = (out["prob_dropout_14d"] >= thr).astype(int)
    out["reason"] = out.apply(explain, axis=1)
    out = out.sort_values("prob_dropout_14d", ascending=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"OK. Saved {len(out)} rows to {args.out}. Alerts={out['alert'].sum()}")

if __name__ == "__main__":
    main()
