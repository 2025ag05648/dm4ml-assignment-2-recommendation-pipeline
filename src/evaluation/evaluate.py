"""Full-catalog offline ranking; user macro averages and category-based diversity."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import load,save_csv,write_json
from src.training.train import load_model,recommend

def rank_metrics(ranked,relevant,k):
    hits=np.array([int(item in relevant) for item in ranked],dtype=float)
    precision=float(hits.sum()/k)
    recall=float(hits.sum()/len(relevant))
    dcg=float((hits/np.log2(np.arange(2,len(hits)+2))).sum())
    ideal=float((1/np.log2(np.arange(2,min(k,len(relevant))+2))).sum())
    return precision,recall,dcg/ideal if ideal else 0.0

def run_stage(run,cfg):
    run=Path(run);model=load_model(run);test=load(run,"processed","test")
    items=load(run,"processed","items");categories=items.set_index("item_id").category.to_dict()
    known_users=set(model["users"].tolist());catalog=set(model["items"].tolist());k=cfg["k"]
    eligible={};cold=set();unrecommendable=0
    for uid,group in test.groupby("user_id"):
        if uid not in known_users: cold.add(int(uid));continue
        u=int(np.where(model["users"]==uid)[0][0]);seen=set(model["items"][model["seen"][u]].tolist())
        positives=set(group.loc[group.rating>=cfg["positive_rating"],"item_id"].astype(int))
        allowed=(positives & catalog)-seen
        unrecommendable+=len(positives-allowed)
        if allowed: eligible[int(uid)]=allowed
    if not eligible: raise ValueError("No eligible test users with positive unseen items")
    results=[];records=[]
    for kind in ["popularity","hybrid"]:
        metrics=[];recommended=set();diversities=[]
        for uid,relevant in eligible.items():
            ranked=recommend(model,uid,kind,k)
            metrics.append(rank_metrics(ranked,relevant,k));recommended.update(ranked)
            pairs=list(combinations(ranked,2))
            diversity=sum(categories[a]!=categories[b] for a,b in pairs)/len(pairs) if pairs else 0.0
            diversities.append(diversity)
            for rank,item in enumerate(ranked,1):
                records.append({"model":kind,"user_id":uid,"rank":rank,"item_id":item,"category":categories[item],"relevant":item in relevant})
        avg=np.mean(metrics,axis=0)
        result={"model":kind,f"Precision@{k}":float(avg[0]),f"Recall@{k}":float(avg[1]),f"NDCG@{k}":float(avg[2]),"Coverage":len(recommended)/len(catalog),"Diversity":float(np.mean(diversities))}
        results.append(result)
    table=pd.DataFrame(results);save_csv(table,run/"reports/evaluation.csv")
    save_csv(pd.DataFrame(records),run/"reports/recommendations.csv")
    write_json(run/"reports/evaluation_protocol.json",{"k":k,"positive_rating":cfg["positive_rating"],"eligible_users":len(eligible),"test_users":int(test.user_id.nunique()),"cold_users_excluded":len(cold),"users_without_eligible_positives":int(test.user_id.nunique())-len(cold)-len(eligible),"unrecommendable_positive_pairs":unrecommendable,"candidate_catalog_size":len(catalog),"ranking":"full training catalog excluding each user training-seen items; deterministic item-ID ties","aggregation":"macro average over users with at least one eligible positive","coverage":"distinct recommended items divided by training catalog size","diversity":"mean fraction of recommendation pairs with different categories","rmse_mae":"Not applicable: models rank items and do not predict explicit ratings","weights":"Fixed in advance, no tuning on test"})
    print(table.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"Eligible evaluation users={len(eligible)}; cold users excluded={len(cold)}; relevance=rating>={cfg['positive_rating']}")
    print("Metrics use unseen train-catalog candidates; RMSE/MAE not applicable to rank-only models")
