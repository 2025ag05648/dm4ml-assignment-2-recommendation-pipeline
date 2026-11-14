"""Generate repeatable fictional movie data, or ingest user-supplied CSVs."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from src.common import save_csv, sha256, write_json

CATEGORIES = ["Action", "Comedy", "Drama", "Science Fiction", "Thriller", "Animation"]

def generate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n, m = cfg["users"], cfg["items"]
    if cfg["interactions_per_user"] > m:
        raise ValueError("Interactions per user must not exceed catalog size")
    users = pd.DataFrame({"user_id": range(1,n+1), "age": rng.integers(18,70,n),
                          "region": rng.choice(["North","South","East","West"],n)})
    items = pd.DataFrame({"item_id": range(1,m+1),
        "title": [f"Fictional Movie {i:03d}" for i in range(1,m+1)],
        "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(m)]})
    rows = []
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    span = int((end-start).total_seconds())
    for uid in users.user_id:
        affinity = rng.dirichlet(np.full(len(CATEGORIES), 0.6))
        choice_weights = np.array([affinity[i % len(CATEGORIES)] for i in range(m)])
        choice_weights = 0.25/m + 0.75*choice_weights/choice_weights.sum()
        chosen = rng.choice(m, cfg["interactions_per_user"], replace=False, p=choice_weights)
        for item in chosen:
            score = 2.2 + 2.8*affinity[item % len(CATEGORIES)]/affinity.max() + rng.normal(0,0.65)
            rating = int(np.clip(np.rint(score),1,5))
            ts = start + pd.Timedelta(seconds=int(rng.integers(0,span)))
            rows.append([int(uid),int(item)+1,rating,ts.isoformat()])
    interactions = pd.DataFrame(rows,columns=["user_id","item_id","rating","timestamp"])
    # Deliberate independent faults make quarantine behavior demonstrable.
    faults = []
    for column,value in [("user_id",999999),("item_id",999999),("rating",8),
                         ("rating","bad"),("user_id","wrong-type"),("rating",None),
                         ("timestamp","invalid-date"),("timestamp","2035-01-01T00:00:00Z")]:
        row=interactions.iloc[0].to_dict();row[column]=value;faults.append(row)
    interactions=pd.concat([interactions,pd.DataFrame(faults),interactions.iloc[:2]],ignore_index=True)
    users.loc[0,"age"]=np.nan  # Optional field: impute after splitting is defined.
    items.loc[0,"category"]=None  # Optional metadata: explicit Unknown category.
    return users,items,interactions

def run_stage(run,cfg,source=None):
    run=Path(run); dest=run/"data/raw";dest.mkdir(parents=True,exist_ok=True)
    names=["users","items","interactions"]
    if source:
        for name in names:
            shutil.copyfile(Path(source)/f"{name}.csv",dest/f"{name}.csv")
        kind="supplied_csv"
    else:
        for name,df in zip(names,generate(cfg)):
            save_csv(df,dest/f"{name}.csv")
        kind="documented_synthetic"
    manifest={"source":kind,"seed":cfg["seed"],"files":{}}
    for name in names:
        path=dest/f"{name}.csv"
        count=len(pd.read_csv(path))
        manifest["files"][name]={"rows":count,"sha256":sha256(path)}
        print(f"Ingested {name}: {count} rows; SHA256 {sha256(path)[:16]}")
    write_json(run/"reports/ingestion.json",manifest)
    print(f"Source: {kind}; files preserved unchanged under data/raw")
