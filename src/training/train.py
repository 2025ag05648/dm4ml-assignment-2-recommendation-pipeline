"""Popularity and item-neighborhood hybrid; artifacts contain scores, never test labels."""
from pathlib import Path
import numpy as np
from src.common import write_json,read_json
from src.features.store import FeatureStore

def fit(features,cfg):
    users=features["users"].sort_values("user_id");items=features["items"].sort_values("item_id")
    events=features["interactions"];prefs=features["user_categories"]
    uids=users.user_id.astype(int).tolist();iids=items.item_id.astype(int).tolist()
    um={u:i for i,u in enumerate(uids)};im={v:i for i,v in enumerate(iids)}
    positives=np.zeros((len(uids),len(iids)),dtype=float);seen=np.zeros_like(positives,dtype=bool)
    for row in events.itertuples(index=False):
        u,i=um[row.user_id],im[row.item_id];seen[u,i]=True
        positives[u,i]=row.is_positive
    norms=np.sqrt((positives**2).sum(axis=0))
    sim=(positives.T@positives)/np.maximum(norms[:,None]*norms[None,:],1e-12)
    np.fill_diagonal(sim,0)
    cf=positives@sim
    cf=cf/np.maximum(cf.max(axis=1,keepdims=True),1e-12)
    lookup={(int(r.user_id),r.category):r.preference for r in prefs.itertuples(index=False)}
    content=np.array([[lookup.get((u,c),0) for c in items.category] for u in uids])
    content=content/np.maximum(content.max(axis=1,keepdims=True),1e-12)
    pop=items.positive_count.to_numpy(dtype=float)
    pop=pop/max(float(pop.max()),1.0)
    hybrid=cfg["cf_weight"]*cf+cfg["content_weight"]*content+cfg["popularity_weight"]*pop
    return {"users":np.array(uids),"items":np.array(iids),"seen":seen,"popularity":pop,"hybrid":hybrid,"similarity":sim}

def recommend(model,user_id,kind,k):
    matches=np.where(model["users"]==user_id)[0]
    if kind not in ["popularity","hybrid"]: raise ValueError("Unknown model")
    if len(matches):
        u=int(matches[0]);scores=model["popularity"].copy() if kind=="popularity" else model["hybrid"][u].copy()
        scores[model["seen"][u]]=-np.inf
    else: scores=model["popularity"].copy()  # Cold users get a deterministic baseline.
    order=np.lexsort((model["items"],-scores))
    return [int(model["items"][i]) for i in order if np.isfinite(scores[i])][:k]

def load_model(run):
    with np.load(Path(run)/"models/recommenders.npz",allow_pickle=False) as data:
        return {k:data[k] for k in data.files}

def run_stage(run,cfg):
    run=Path(run);features=FeatureStore(run).load();model=fit(features,cfg)
    (run/"models").mkdir(parents=True,exist_ok=True)
    np.savez_compressed(run/"models/recommenders.npz",**model)
    write_json(run/"models/parameters.json",{k:cfg[k] for k in ["positive_rating","cf_weight","content_weight","popularity_weight","seed"]})
    print(f"Fitted popularity baseline and personalized item-CF/category hybrid on {len(features['interactions'])} training rows")
    print(f"Users={len(model['users'])}, candidate items={len(model['items'])}, item similarity shape={model['similarity'].shape}")
    print(f"Fixed hybrid weights: CF={cfg['cf_weight']}, content={cfg['content_weight']}, popularity={cfg['popularity_weight']}")
    print("Saved non-pickle NPZ artifact and parameters; MLflow logging occurs after evaluation")
