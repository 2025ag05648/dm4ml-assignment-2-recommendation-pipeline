"""Derived summaries and sparse matrix use training interactions only."""
from pathlib import Path
import pandas as pd
from scipy.sparse import coo_matrix,save_npz
from src.common import load,save_csv,write_json

def run_stage(run,cfg):
    run=Path(run);train=load(run,"processed","train")
    users=train.groupby("user_id").agg(activity_count=("rating","size"),average_rating=("rating","mean"),last_interaction=("timestamp","max")).reset_index()
    items=train.groupby("item_id").agg(interaction_count=("rating","size"),average_rating=("rating","mean")).reset_index()
    ids_u=sorted(train.user_id.unique().tolist());ids_i=sorted(train.item_id.unique().tolist())
    um={u:i for i,u in enumerate(ids_u)};im={v:i for i,v in enumerate(ids_i)}
    matrix=coo_matrix((train.rating.to_numpy(),(train.user_id.map(um),train.item_id.map(im))),shape=(len(um),len(im))).tocsr()
    path=run/"data/transformed";path.mkdir(parents=True,exist_ok=True)
    save_csv(users,path/"user_activity.csv");save_csv(items,path/"item_popularity.csv")
    save_npz(path/"user_item_matrix.npz",matrix)
    write_json(path/"matrix_index.json",{"users":ids_u,"items":ids_i})
    print(f"Training-only summaries: {len(users)} users, {len(items)} items")
    print(f"CSR interaction matrix: {matrix.shape}, nonzero={matrix.nnz}, density={matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.3f}")
    print("Saved activity summary, item popularity, sparse matrix and stable ID mapping")
