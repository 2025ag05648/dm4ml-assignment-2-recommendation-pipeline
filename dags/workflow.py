"""Explicit dependency graph; run with python -m dags.workflow."""
import argparse
import contextlib
from datetime import datetime,timezone
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import mlflow
import pandas as pd
from src.common import read_json,write_json,sha256
from src.ingestion import ingest
from src.validation import validate
from src.preparation import prepare
from src.transformation import transform
from src.features import store
from src.training import train
from src.evaluation import evaluate

STAGES=[("ingestion",[],ingest.run_stage),("validation",["ingestion"],validate.run_stage),
 ("preparation",["validation"],prepare.run_stage),("transformation",["preparation"],transform.run_stage),
 ("feature_management",["transformation"],store.run_stage),("model_training",["feature_management"],train.run_stage),
 ("model_evaluation",["model_training"],evaluate.run_stage)]

class Tee(io.TextIOBase):
    def __init__(self,other): self.other=other;self.buffer=io.StringIO()
    def write(self,text): self.other.write(text);self.buffer.write(text);return len(text)
    def flush(self): self.other.flush()

def git_revision():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): return "not_in_git"

def track(run,cfg,state):
    # Absolute local file URI isolates experiments from environment settings.
    mlflow.set_tracking_uri((run/"mlruns").resolve().as_uri())
    mlflow.set_experiment("dm4ml-recommendation")
    table=pd.read_csv(run/"reports/evaluation.csv")
    for row in table.to_dict("records"):
        with mlflow.start_run(run_name=row["model"]) as active:
            mlflow.log_params(cfg)
            mlflow.set_tags({"pipeline_run":state["run_id"],"execution_environment":state["environment"],"source_revision":state["git_revision"]})
            mlflow.log_metrics({key.replace("@","_at_"):value for key,value in row.items() if key!="model"})
            mlflow.log_artifacts(str(run/"models"),artifact_path="models")
            mlflow.log_artifacts(str(run/"data/features"),artifact_path="features")
            for name in ["evaluation.csv","evaluation_protocol.json","feature_catalog.csv","validation.json"]:
                mlflow.log_artifact(str(run/"reports"/name),artifact_path="reports")
            state.setdefault("mlflow_runs",{})[row["model"]]=active.info.run_id
            print(f"MLflow {row['model']}: run_id={active.info.run_id}; params, metrics, model and feature artifacts logged")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/pipeline.json")
    p.add_argument("--output",default="runs")
    p.add_argument("--source",help="Optional directory containing users.csv, items.csv, interactions.csv")
    p.add_argument("--environment",default="Local execution; BITS Lab execution not verified")
    a=p.parse_args();cfg=read_json(a.config)
    if cfg["k"] not in [5,10]: raise ValueError("Assignment requires K=5 or K=10")
    if not pd.Timestamp(cfg["start"])<pd.Timestamp(cfg["cutoff"])<pd.Timestamp(cfg["end"]): raise ValueError("Invalid time boundaries")
    if abs(sum(cfg[k] for k in ["cf_weight","content_weight","popularity_weight"])-1)>1e-9: raise ValueError("Hybrid weights must sum to one")
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]
    run=Path(a.output).resolve()/run_id
    (run/"logs").mkdir(parents=True);(run/"reports").mkdir()
    write_json(run/"config.json",cfg)
    state={"run_id":run_id,"environment":a.environment,"hostname":platform.node(),"platform":platform.platform(),"python":sys.version,"git_revision":git_revision(),"started_utc":datetime.now(timezone.utc).isoformat(),"status":"RUNNING","stages":[],"dependencies":{n:d for n,d,_ in STAGES}}
    state["packages"]={name:importlib.metadata.version(name) for name in ["numpy","pandas","scipy","mlflow-skinny","reportlab"]}
    write_json(run/"run.json",state)
    done=set()
    try:
        for name,deps,fn in STAGES+[("orchestration",["model_evaluation"],None)]:
            if not set(deps)<=done: raise RuntimeError(f"Unsatisfied dependencies: {name}")
            record={"stage":name,"status":"RUNNING","started_utc":datetime.now(timezone.utc).isoformat()};state["stages"].append(record)
            tee=Tee(sys.stdout);started=time.perf_counter()
            try:
                with contextlib.redirect_stdout(tee):
                    print(f"STAGE {name} | run {run_id} | environment {a.environment}")
                    if name=="ingestion": fn(run,cfg,a.source)
                    elif name=="orchestration":
                        track(run,cfg,state)
                        for prev in state["stages"][:-1]: print(f"{prev['stage']}: {prev['status']}")
                        print("Dependencies satisfied; 8/8 stages completed; workflow SUCCESS")
                    else: fn(run,cfg)
                    print(f"SUCCESS {name}")
                record["status"]="SUCCESS";done.add(name)
            except Exception as exc:
                record.update(status="FAILED",error=repr(exc));raise
            finally:
                record["seconds"]=round(time.perf_counter()-started,4)
                (run/"logs"/f"{name}.log").write_text(tee.buffer.getvalue())
                write_json(run/"run.json",state)
        state["status"]="SUCCESS"
    except Exception as exc:
        state.update(status="FAILED",error=repr(exc));raise
    finally:
        state["finished_utc"]=datetime.now(timezone.utc).isoformat()
        state["artifact_sha256"]={str(path.relative_to(run)):sha256(path) for folder in ["data","models","reports","logs"] for path in sorted((run/folder).rglob("*")) if path.is_file()}
        write_json(run/"run.json",state)
        if state["status"]=="SUCCESS":
            (Path(a.output).resolve()/"LATEST").write_text(str(run)+"\n")
        print(f"RUN_DIRECTORY={run}")

if __name__=="__main__": main()
