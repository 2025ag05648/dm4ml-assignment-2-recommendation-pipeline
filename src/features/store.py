"""Versioned custom feature layer shared by model fit and prediction."""
from pathlib import Path
import pandas as pd
from src.common import load,save_csv,write_json,sha256,read_json

CATALOG=[
 ["user","activity_count","int","Number of training interactions","Training history"],
 ["user","average_rating","float","Mean training rating","Training history"],
 ["user","days_since_last_interaction","float","Cutoff minus latest training timestamp in days","Training history"],
 ["user","preferred_category","string","Category with most positive training interactions; Unknown fallback","Training positives and item metadata"],
 ["user_category","preference","float","Smoothed category positive share; numerator count plus 1, denominator positives plus category count","Training positives only"],
 ["item","interaction_count","int","Number of training interactions","Training history"],
 ["item","average_rating","float","Mean training rating","Training history"],
 ["item","positive_count","int","Training ratings at or above positive threshold","Training history"],
 ["item","category","string","Cleaned item category","Static catalog metadata"],
 ["interaction","is_positive","int","Rating at or above fixed positive threshold","Training history"],
 ["interaction","days_before_cutoff","float","Cutoff minus interaction timestamp in days","Training history"],
]

class FeatureStore:
    def __init__(self,run): self.path=Path(run)/"data/features"
    def load(self):
        metadata=read_json(self.path/"metadata.json")
        for name,digest in metadata["sha256"].items():
            if sha256(self.path/name)!=digest: raise ValueError(f"Feature checksum mismatch: {name}")
        return {n:pd.read_csv(self.path/f"{n}.csv") for n in ["users","items","user_categories","interactions"]}

def run_stage(run,cfg):
    run=Path(run);path=run/"data/features";path.mkdir(parents=True,exist_ok=True)
    train=load(run,"processed","train");items=load(run,"processed","items")
    us=pd.read_csv(run/"data/transformed/user_activity.csv")
    it=pd.read_csv(run/"data/transformed/item_popularity.csv")
    cutoff=pd.Timestamp(cfg["cutoff"])
    us["days_since_last_interaction"]=(cutoff-pd.to_datetime(us.last_interaction,utc=True)).dt.total_seconds()/86400
    pos=train.loc[train.rating>=cfg["positive_rating"]].merge(items[["item_id","category"]],on="item_id",validate="many_to_one")
    categories=sorted(items.category.unique())
    counts=pos.groupby(["user_id","category"]).size().unstack(fill_value=0).reindex(index=us.user_id,columns=categories,fill_value=0)
    preferences=(counts+1).div(counts.sum(axis=1)+len(categories),axis=0)
    prefs=preferences.rename_axis(index="user_id",columns="category").stack().rename("preference").reset_index()
    preferred=counts.idxmax(axis=1).where(counts.sum(axis=1)>0,"Unknown")
    us["preferred_category"]=us.user_id.map(preferred)
    it=it.merge(items,on="item_id",validate="one_to_one")
    it["positive_count"]=it.item_id.map(pos.groupby("item_id").size()).fillna(0).astype(int)
    interactions=train.copy()
    interactions["is_positive"]=(train.rating>=cfg["positive_rating"]).astype(int)
    interactions["days_before_cutoff"]=(cutoff-pd.to_datetime(train.timestamp,utc=True)).dt.total_seconds()/86400
    for name,df in [("users",us),("items",it),("user_categories",prefs),("interactions",interactions)]:
        save_csv(df,path/f"{name}.csv")
    catalog=pd.DataFrame(CATALOG,columns=["entity","feature","type","definition","source"])
    save_csv(catalog,run/"reports/feature_catalog.csv")
    lines=["| Entity | Feature | Type | Definition | Source |","|---|---|---|---|---|"]
    lines += ["| "+" | ".join(row)+" |" for row in CATALOG]
    (run/"reports/feature_catalog.md").write_text("\n".join(lines)+"\n")
    metadata={"version":"v1","as_of":cfg["cutoff"],"source":"processed/train.csv only plus static item metadata", "sha256":{p.name:sha256(p) for p in sorted(path.glob("*.csv"))}}
    write_json(path/"metadata.json",metadata)
    print(f"Feature layer v1 as_of={cfg['cutoff']}: users={len(us)}, items={len(it)}, category preferences={len(prefs)}")
    print(f"{len(CATALOG)} documented features; checksums persisted; no test events used")
