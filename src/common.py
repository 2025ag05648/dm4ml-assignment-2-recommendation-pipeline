"""Shared disk formats: every stage uses the current run directory only."""
import hashlib
import json
from pathlib import Path
import pandas as pd

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")

def read_json(path):
    return json.loads(Path(path).read_text())

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load(run, folder, table):
    return pd.read_csv(Path(run) / "data" / folder / f"{table}.csv")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
