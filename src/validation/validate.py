"""Quarantine invalid rows; reasons are additive and checks never silently coerce IDs."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import load,save_csv,write_json

REQUIRED={"users":["user_id","age","region"],"items":["item_id","title","category"],
          "interactions":["user_id","item_id","rating","timestamp"]}

def validate_table(df,table,cfg,user_ids=None,item_ids=None):
    missing=set(REQUIRED[table])-set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing required columns {sorted(missing)}")
    df=df.copy();reasons=[[] for _ in range(len(df))]
    def flag(mask,label):
        for i,bad in enumerate(mask):
            if bad: reasons[i].append(label)
    ids=["user_id"] if table=="users" else ["item_id"] if table=="items" else ["user_id","item_id"]
    for col in ids:
        numeric=pd.to_numeric(df[col],errors="coerce")
        flag(numeric.isna() | ~np.isfinite(numeric) | (numeric % 1 != 0) | (numeric<=0),f"{col}_invalid_integer")
    flag(df.duplicated(),"duplicate_record")
    if table in ["users","items"]:
        numeric=pd.to_numeric(df[ids[0]],errors="coerce")
        flag(numeric.duplicated(keep="first"),"duplicate_primary_key")
    if table=="items":
        flag(df.title.isna() | df.title.fillna("").astype(str).str.strip().eq(""),"missing_title")
    if table=="users":
        age=pd.to_numeric(df.age,errors="coerce")
        # Missing ages are repairable; nonnumeric or impossible supplied ages are not.
        flag(df.age.notna() & (age.isna() | ~age.between(0,120)),"invalid_age")
    if table=="interactions":
        rating=pd.to_numeric(df.rating,errors="coerce")
        flag(df.rating.isna(),"missing_rating")
        flag(df.rating.notna() & rating.isna(),"rating_incorrect_type")
        flag(rating.notna() & ~rating.between(1,5),"rating_out_of_range")
        ts=pd.to_datetime(df.timestamp,errors="coerce",utc=True,format="mixed")
        flag(ts.isna(),"invalid_timestamp")
        flag(ts.notna() & ((ts<pd.Timestamp(cfg["start"])) | (ts>=pd.Timestamp(cfg["end"]))),"timestamp_out_of_window")
        flag(~pd.to_numeric(df.user_id,errors="coerce").isin(user_ids),"unknown_user")
        flag(~pd.to_numeric(df.item_id,errors="coerce").isin(item_ids),"unknown_item")
    valid=np.array([not row for row in reasons])
    clean=df.loc[valid].copy()
    for col in ids: clean[col]=pd.to_numeric(clean[col]).astype(int)
    if table=="interactions": clean["rating"]=pd.to_numeric(clean.rating).astype(float)
    errors=df.loc[~valid].copy()
    errors.insert(0,"source_csv_line",np.flatnonzero(~valid)+2)
    errors["error_reasons"]=[";".join(row) for row in reasons if row]
    counts={label:sum(label in row for row in reasons) for label in sorted({s for row in reasons for s in row})}
    return clean,errors,{"input":len(df),"valid":len(clean),"quarantined":len(errors),"reasons":counts}

def run_stage(run,cfg):
    run=Path(run);clean={};summary={}
    for table in ["users","items","interactions"]:
        accepted,rejected,stats=validate_table(load(run,"raw",table),table,cfg,
            set(clean["users"].user_id) if "users" in clean else None,
            set(clean["items"].item_id) if "items" in clean else None)
        if accepted.empty: raise ValueError(f"No valid {table} remain")
        clean[table]=accepted;summary[table]=stats
        save_csv(accepted,run/f"data/validated/{table}.csv")
        save_csv(rejected,run/f"data/errors/{table}.csv")
        print(f"{table}: input={stats['input']}, valid={stats['valid']}, quarantined={stats['quarantined']}")
        for reason,count in stats["reasons"].items(): print(f"  {reason}: {count}")
    summary["repairable_missing_age"]=int(clean["users"].age.isna().sum())
    summary["repairable_missing_category"]=int(clean["items"].category.isna().sum())
    write_json(run/"reports/validation.json",summary)
    print("Invalid rows and source line numbers saved to data/errors; optional missing fields queued for preparation")
