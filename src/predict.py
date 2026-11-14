"""Load the shared feature snapshot and persisted model for inference."""
import argparse
from pathlib import Path
from src.features.store import FeatureStore
from src.training.train import load_model,recommend

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--user",type=int,required=True);p.add_argument("--k",type=int,default=5)
    a=p.parse_args();features=FeatureStore(a.run).load();model=load_model(a.run)
    print(f"Hybrid recommendations for user {a.user} (cold-user fallback: popularity)")
    ids=recommend(model,a.user,"hybrid",a.k)
    items=features["items"].set_index("item_id")
    for rank,i in enumerate(ids,1): print(f"{rank}. {i} | {items.loc[i,'title']} | {items.loc[i,'category']}")
if __name__=="__main__": main()
