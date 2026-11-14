"""Contract tests: malformed data, hand-computed metrics, leakage and inference."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.common import read_json
from src.ingestion.ingest import generate
from src.validation.validate import validate_table
from src.evaluation.evaluate import rank_metrics
from src.training.train import fit,recommend,load_model
from src.features.store import FeatureStore

ROOT=Path(__file__).resolve().parents[1]

class DataContracts(unittest.TestCase):
    def setUp(self): self.cfg=read_json(ROOT/"configs/pipeline.json")
    def test_synthetic_deterministic(self):
        for first,second in zip(generate(self.cfg),generate(self.cfg)):
            pd.testing.assert_frame_equal(first,second)
    def test_faults_quarantined(self):
        users,items,events=generate(self.cfg)
        clean,rejected,report=validate_table(events,"interactions",self.cfg,set(users.user_id),set(items.item_id))
        self.assertEqual(len(clean),8400);self.assertEqual(len(rejected),10)
        for reason in ["missing_rating","duplicate_record","rating_incorrect_type","rating_out_of_range","invalid_timestamp","timestamp_out_of_window","user_id_invalid_integer","unknown_user","unknown_item"]:
            self.assertGreater(report["reasons"].get(reason,0),0)
        self.assertEqual(len(clean)+len(rejected),len(events))
    def test_schema_missing_column_fails(self):
        with self.assertRaisesRegex(ValueError,"missing required columns"):
            validate_table(pd.DataFrame({"user_id":[1]}),"interactions",self.cfg,{1},{1})
    def test_fractional_id_not_truncated(self):
        frame=pd.DataFrame({"user_id":[1.5],"item_id":[1],"rating":[4],"timestamp":["2025-02-01T00:00:00Z"]})
        accepted,rejected,_=validate_table(frame,"interactions",self.cfg,{1},{1})
        self.assertTrue(accepted.empty);self.assertIn("user_id_invalid_integer",rejected.iloc[0].error_reasons)
    def test_duplicate_primary_key_different_metadata(self):
        frame=pd.DataFrame({"user_id":[1,1],"age":[20,30],"region":["A","B"]})
        accepted,rejected,_=validate_table(frame,"users",self.cfg)
        self.assertEqual(len(accepted),1);self.assertIn("duplicate_primary_key",rejected.iloc[0].error_reasons)
    def test_hand_computed_metrics(self):
        p,r,n=rank_metrics([1,2,3,4,5],{2,4},5)
        self.assertAlmostEqual(p,0.4);self.assertAlmostEqual(r,1)
        expected=(1/np.log2(3)+1/np.log2(5))/(1+1/np.log2(3))
        self.assertAlmostEqual(n,expected)
        self.assertEqual(rank_metrics([6,7,8,9,10],{2,4},5),(0,0,0))

class ExecutedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        latest=ROOT/"runs/LATEST"
        if not latest.exists(): raise unittest.SkipTest("Run python -m dags.workflow first")
        cls.run_dir=Path(latest.read_text().strip());cls.cfg=read_json(cls.run_dir/"config.json")
    def test_success_and_mlflow(self):
        state=read_json(self.run_dir/"run.json")
        self.assertEqual(state["status"],"SUCCESS")
        self.assertEqual(len(state["stages"]),8)
        self.assertTrue(all(s["status"]=="SUCCESS" for s in state["stages"]))
        import mlflow
        client=mlflow.tracking.MlflowClient(tracking_uri=(self.run_dir/"mlruns").as_uri())
        for kind,runid in state["mlflow_runs"].items():
            tracked=client.get_run(runid)
            self.assertEqual(tracked.info.status,"FINISHED")
            self.assertIn("Precision_at_5",tracked.data.metrics)
            self.assertTrue(client.list_artifacts(runid,"models"))
    def test_global_split_and_feature_lineage(self):
        train=pd.read_csv(self.run_dir/"data/processed/train.csv")
        test=pd.read_csv(self.run_dir/"data/processed/test.csv")
        self.assertLess(pd.to_datetime(train.timestamp,utc=True).max(),pd.to_datetime(test.timestamp,utc=True).min())
        f=FeatureStore(self.run_dir).load()
        self.assertEqual(len(f["interactions"]),len(train))
        self.assertTrue((f["interactions"].days_before_cutoff>0).all())
        expected=train.groupby("user_id").rating.mean().sort_index()
        actual=f["users"].set_index("user_id").average_rating.sort_index()
        np.testing.assert_allclose(expected.to_numpy(),actual.to_numpy())
    def test_seen_items_excluded_and_unknown_user_fallback(self):
        model=load_model(self.run_dir)
        for uidx,uid in enumerate(model["users"]):
            seen=set(model["items"][model["seen"][uidx]])
            for kind in ["popularity","hybrid"]:
                ranked=recommend(model,int(uid),kind,5)
                self.assertEqual(len(ranked),5);self.assertEqual(len(set(ranked)),5)
                self.assertFalse(seen.intersection(ranked))
        self.assertEqual(recommend(model,999999,"hybrid",5),recommend(model,999999,"popularity",5))
    def test_retraining_reproduces_scores(self):
        model=load_model(self.run_dir);rerun=fit(FeatureStore(self.run_dir).load(),self.cfg)
        for key in model: np.testing.assert_allclose(model[key],rerun[key],atol=1e-12)
    def test_corrupt_feature_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data/features";shutil.copytree(self.run_dir/"data/features",p)
            with (p/"users.csv").open("a") as f: f.write("corrupted\n")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"): FeatureStore(tmp).load()

if __name__=="__main__": unittest.main(verbosity=2)
