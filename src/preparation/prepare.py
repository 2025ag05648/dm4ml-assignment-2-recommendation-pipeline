"""Clean optional metadata and split globally at a configured UTC cutoff."""
from pathlib import Path
import pandas as pd
from src.common import load,save_csv,write_json

def run_stage(run,cfg):
    run=Path(run)
    users,items,events=[load(run,"validated",n) for n in ["users","items","interactions"]]
    age=pd.to_numeric(users.age)
    median=age.median() if age.notna().any() else 30
    users["age"]=age.fillna(median).astype(int)
    for df,col in [(users,"region"),(items,"category")]:
        df[col]=df[col].fillna("Unknown").astype(str).str.strip().replace("","Unknown")
    items["title"]=items.title.astype(str).str.strip()
    events["timestamp"]=pd.to_datetime(events.timestamp,utc=True,format="mixed")
    before=len(events)
    # A repeated user/item rating represents an update: retain the latest at each side of cutoff.
    cutoff=pd.Timestamp(cfg["cutoff"])
    train=events.loc[events.timestamp<cutoff].sort_values("timestamp").drop_duplicates(["user_id","item_id"],keep="last")
    test=events.loc[events.timestamp>=cutoff].sort_values("timestamp").drop_duplicates(["user_id","item_id"],keep="last")
    if train.empty or test.empty: raise ValueError("Time split must produce nonempty training and test sets")
    if not train.timestamp.max()<test.timestamp.min(): raise AssertionError("Temporal split leakage")
    for name,df in [("users",users),("items",items),("train",train),("test",test)]:
        save_csv(df,run/f"data/processed/{name}.csv")
    summary={"cutoff":cutoff.isoformat(),"train":len(train),"test":len(test),
             "repeated_pair_updates_removed":before-len(train)-len(test),
             "max_train_timestamp":train.timestamp.max().isoformat(),"min_test_timestamp":test.timestamp.min().isoformat(),
             "age_imputation":"median of static user metadata; never used by either ranker"}
    write_json(run/"reports/split.json",summary)
    print(f"Train {len(train)} rows; test {len(test)} rows; UTC cutoff {cutoff.isoformat()}")
    print(f"Max train {train.timestamp.max()}; min test {test.timestamp.min()}")
    print("Filled optional ages/categories; standardized strings and UTC timestamps; strict temporal ordering PASS")
