"""Build a 3-page report plus 4 evidence pages. Final mode requires real repo URL and lab attestation."""
import argparse
from datetime import datetime
from itertools import islice
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph,Table,TableStyle
from src.common import read_json,sha256

W,H=595.28,841.89
M=44;CW=W-2*M
STAGES=["ingestion","validation","preparation","transformation","feature_management","model_training","model_evaluation","orchestration"]
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyCustom",fontName="Helvetica",fontSize=10,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name="SmallCustom",fontName="Helvetica",fontSize=8.4,leading=11))

class Report:
    def __init__(self,path,draft):
        self.c=canvas.Canvas(str(path),pagesize=(W,H));self.page=0;self.draft=draft;self.y=H-M
        self.c.setTitle("End to End Recommendation Pipeline")
        self.new()
    def new(self):
        if self.page:self.footer();self.c.showPage()
        self.page+=1;self.y=H-M
        self.c.setFont("Helvetica",8);self.c.setFillColor(colors.HexColor("#48566b"))
        self.c.drawString(M,self.y,"DATA MANAGEMENT FOR MACHINE LEARNING  |  ASSIGNMENT 2")
        self.y-=24
        if self.draft:
            self.c.setFont("Helvetica-Bold",10);self.c.setFillColor(colors.HexColor("#9b3d0c"))
            self.c.drawString(M,self.y,"PREVIEW  |  LAB RUN AND GITHUB URL STILL REQUIRED")
            self.y-=23
        self.c.setFillColor(colors.black)
    def footer(self):
        self.c.setFont("Helvetica",8);self.c.setFillColor(colors.HexColor("#48566b"))
        self.c.drawString(M,24,"Recommendation pipeline | "+("Local execution preview" if self.draft else "Submission report"))
        self.c.drawRightString(W-M,24,str(self.page));self.c.setFillColor(colors.black)
    def heading(self,text,size=14):
        self.y-=4;self.c.setFont("Helvetica-Bold",size);self.c.drawString(M,self.y,text);self.y-=20
    def text(self,text,small=False):
        p=Paragraph(text,styles["SmallCustom" if small else "BodyCustom"])
        _,height=p.wrap(CW,800)
        if self.y-height<50: raise ValueError(f"Report overflow page {self.page}: {text[:50]}")
        p.drawOn(self.c,M,self.y-height);self.y-=height+8
    def table(self,rows,widths,font=8.5):
        cells=[[Paragraph(escape(str(c)),styles["SmallCustom"]) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7edf4")),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#ced5de")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        _,h=t.wrap(CW,800)
        if self.y-h<50:raise ValueError("Table overflow")
        t.drawOn(self.c,M,self.y-h);self.y-=h+14
    def image(self,path,maxheight):
        with PILImage.open(path) as im: iw,ih=im.size
        scale=min(CW/iw,maxheight/ih);dw,dh=iw*scale,ih*scale
        self.c.drawImage(str(path),M+(CW-dw)/2,self.y-dh,width=dw,height=dh,preserveAspectRatio=True)
        self.y-=dh+10
    def close(self):self.footer();self.c.save()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True)
    p.add_argument("--out",default="submission/Recommendation_Pipeline_Preview.pdf")
    p.add_argument("--repo-url");p.add_argument("--student",default="Student name and ID to be added")
    p.add_argument("--final",action="store_true");p.add_argument("--confirm-lab-run",action="store_true",help="Explicitly attest that this run was executed on the required BITS Lab infrastructure")
    p.add_argument("--screenshots",help="Optional folder containing named PNGs for all 8 stages; otherwise run/evidence")
    a=p.parse_args();run=Path(a.run).resolve();state=read_json(run/"run.json");cfg=read_json(run/"config.json")
    if state["status"]!="SUCCESS":raise ValueError("Report requires a successful run")
    if a.final:
        if not a.repo_url or not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?",a.repo_url): raise ValueError("Final report requires an actual GitHub repository URL")
        if not a.confirm_lab_run:raise ValueError("Final report requires --confirm-lab-run after execution on BITS Lab")
        if "BITS" not in state["environment"] or "not BITS" in state["environment"]:raise ValueError("Current run environment is not labelled as a BITS Lab execution")
        if a.student=="Student name and ID to be added":raise ValueError("Provide --student for final report")
    evidence=Path(a.screenshots).resolve() if a.screenshots else run/"evidence"
    for stage in STAGES:
        if not (evidence/f"{stage}.png").exists(): raise ValueError(f"Missing screenshot: {stage}.png")
    if not a.screenshots:
        manifest=read_json(evidence/"manifest.json")
        if manifest["run_id"]!=state["run_id"]:raise ValueError("Screenshot/run mismatch")
        for stage in STAGES:
            rec=manifest["files"][stage]
            if sha256(evidence/f"{stage}.png")!=rec["png_sha256"] or sha256(run/"logs"/f"{stage}.log")!=rec["log_sha256"]: raise ValueError("Screenshot evidence checksum mismatch")
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    r=Report(out,not a.final)
    r.heading("End to End Recommendation Pipeline",18)
    r.text(escape(a.student))
    r.text("<b>Objective.</b> Build and automate a movie recommendation data pipeline with validated ingestion, time-aware preparation, reusable features, two ranking models, experiment tracking and offline evaluation.")
    repo=a.repo_url or "PENDING - create and upload the GitHub repository before submission"
    if a.repo_url:r.text(f'<b>GitHub repository.</b> <link href="{escape(repo)}" color="#164b87">{escape(repo)}</link>')
    else:r.text("<b>GitHub repository.</b> "+repo)
    r.text("<b>Execution provenance.</b> "+escape(state["environment"])+". Run "+escape(state["run_id"])+". Started "+escape(state["started_utc"])+". All eight stages completed successfully.")
    r.heading("Design and reproducibility")
    r.text("Python modules exchange versioned CSV and NPZ artifacts within an isolated run directory. A structured Python workflow enforces dependencies, stops on failure, records task durations, preserves raw inputs, hashes outputs and records the software environment. A cron example schedules the same entry point; a file lock prevents overlap.")
    r.table([["Order","Pipeline stage","Primary artifact"],
      ["1","Ingestion","Raw CSV files and source checksums"],
      ["2","Validation","Validated rows, quarantine and reason counts"],
      ["3","Preparation","Clean metadata, chronological train/test split"],
      ["4","Transformation","Activity, popularity and sparse rating matrix"],
      ["5","Feature management","Versioned features, catalog and checksums"],
      ["6","Model training","Popularity and personalized hybrid artifacts"],
      ["7","Evaluation","Five ranking metrics and recommendation lists"],
      ["8","Orchestration","MLflow runs, stage status and run manifest"]],[38,130,CW-168])
    r.heading("Dataset and scope")
    r.text(f"The documented synthetic dataset contains {cfg['users']} fictional users, {cfg['items']} fictional movies and {cfg['users']*cfg['interactions_per_user']:,} valid ratings from January to June 2025. Seed {cfg['seed']} controls generation. User category preferences influence sampling and ratings. Ten deliberately faulty interaction rows demonstrate validation; one missing age and one missing category demonstrate repair.")
    r.text("Synthetic data avoids external downloads and private user records. Metrics measure this generator and are not evidence of performance on a real platform. No BITS Lab execution is claimed by a local preview.",small=True)
    r.new();r.heading("Data quality and feature management",17)
    v=read_json(run/"reports/validation.json");split=read_json(run/"reports/split.json")
    r.text(f"<b>Validation result.</b> {v['interactions']['input']:,} interaction rows were ingested; {v['interactions']['valid']:,} were accepted and {v['interactions']['quarantined']} quarantined. Checks cover required columns, missing required values, integer IDs, duplicate records and primary keys, rating types/ranges, parseable in-window UTC timestamps and user/item foreign keys. A row may have several error reasons.")
    r.text("Rejected rows retain source CSV line numbers and reasons in data/errors. Missing optional age is filled with the static user median and is unused by the rankers; missing category becomes Unknown. Duplicate rating updates retain the latest event separately within each side of the cutoff. Empty valid datasets or an empty train/test partition fail the workflow.")
    r.text(f"<b>Temporal boundary.</b> Training has {split['train']:,} rows before {escape(cfg['cutoff'])}; test has {split['test']:,} rows at or after the cutoff. A strict ordering assertion is enforced. Test interactions never enter feature generation, popularity counts, similarities, user profiles or model fitting. Static item metadata is assumed available at the cutoff.")
    r.table([["Entity","Reusable features","Computation"],
      ["User","Activity count, mean rating, preferred category, days since last event","Training history before cutoff"],
      ["User and category","Smoothed positive category preference","(positive category count + 1) / (user positives + category count)"],
      ["Item","Interaction count, mean rating, positive count, category","Training events and static metadata"],
      ["Interaction","Positive indicator, days before cutoff","Rating threshold and cutoff timestamp"]],[76,214,CW-290])
    r.text("<b>Feature contracts.</b> The custom feature layer writes a version, as-of timestamp, provenance and SHA256 per feature file. Both training and inference load this layer and verify checksums. The full eleven-feature catalog is supplied as CSV and Markdown. The persisted training feature snapshot is logged with each MLflow experiment.")
    r.heading("Training and experiment tracking")
    r.text("The baseline ranks items by positive training interaction count. The personalized hybrid combines item-item cosine collaborative scores with the user's smoothed positive category profile and a popularity term. Weights are fixed at 0.65, 0.30 and 0.05 respectively; neither hyperparameters nor model selection are optimized on test results. Cold users fall back to popularity.")
    r.text("Both recommenders filter training-seen items, rank the training catalog, and use item ID to break score ties. MLflow stores a separate run for each model with configuration, ranking metrics, model artifacts, features and quality reports. The default uses a local file tracking store, so no server or external credentials are required.")
    r.new();r.heading("Evaluation results and limitations",17)
    table=pd.read_csv(run/"reports/evaluation.csv");protocol=read_json(run/"reports/evaluation_protocol.json")
    r.text(f"Evaluation uses K={cfg['k']} and rating >= {cfg['positive_rating']} as relevance. It scores {protocol['eligible_users']} users with at least one unseen positive test item in the training catalog. {protocol['cold_users_excluded']} cold users and {protocol['users_without_eligible_positives']} users without eligible positives were excluded. {protocol['unrecommendable_positive_pairs']} positive pairs were not recommendable under this protocol.")
    rows=[["Model",f"P@{cfg['k']}",f"R@{cfg['k']}",f"NDCG@{cfg['k']}","Coverage","Diversity"]]
    for rec in table.to_dict("records"): rows.append([rec["model"]]+[f"{rec[c]:.4f}" for c in table.columns[1:]])
    r.table(rows,[87,66,66,84,96,CW-399])
    r.text("<b>Definitions.</b> Precision@K is hits divided by K; Recall@K is hits divided by the user's eligible positives. NDCG@K discounts binary hits by log2(rank + 1) and normalizes by ideal DCG. These are macro averaged across eligible users. Coverage is distinct recommended items divided by training catalog size. Diversity is the mean fraction of pairs within a list that have different categories.")
    k = cfg["k"]
    base,hybrid=table.iloc[0],table.iloc[1]
    r.text(f"<b>Interpretation.</b> On this run, hybrid Precision@{cfg['k']} is {hybrid[f'Precision@{k}']:.4f} versus {base[f'Precision@{k}']:.4f} for popularity; NDCG@{cfg['k']} is {hybrid[f'NDCG@{k}']:.4f} versus {base[f'NDCG@{k}']:.4f}. Coverage rises to {hybrid['Coverage']:.4f}, while category diversity falls to {hybrid['Diversity']:.4f}. Personalization concentrates lists within favored categories; this is an accuracy-variety tradeoff rather than an improvement on every measure.")
    r.heading("Limitations and next steps")
    r.text("The synthetic generator favors category-based personalization and omits realistic exposure bias, arrivals and preference drift. The dense item-similarity matrix scales quadratically and is suitable only for this small assignment dataset. Cold items outside the training catalog are excluded, and cold users are omitted from the main table although inference supports a fallback. Category mismatch is a narrow diversity proxy; Unknown is treated as a category.")
    r.text("The models produce ranking scores, so RMSE and MAE do not apply. Future work should evaluate a real dataset, introduce an earlier validation window for tuning, measure cold-start performance, test sparse nearest-neighbor retrieval, and assess diversity-aware reranking on validation data. No online impact is inferred from this offline experiment.")
    r.heading("Execution and submission evidence")
    evidence_kind = "supplied stage screenshots" if a.screenshots else "browser screenshots of a viewer of saved execution logs"
    r.text("The appendix contains " + evidence_kind + ", one per stage. Each captured log identifies the actual execution environment. Local preview images show terminal windows displaying saved local execution logs. For submission, execute on BITS Lab, capture those outputs there, verify repository access, and rebuild the PDF with the actual repository URL and student details.")
    r.text("The repository includes unit and integration checks for fault quarantine, hand-computed ranking metrics, temporal separation, training-only features, deterministic fitting, seen-item filtering, unknown-user fallback, feature corruption and MLflow artifacts. Run the supplied unittest command after each lab execution.",small=True)
    for i in range(0,8,2):
        r.new();r.heading(f"Execution evidence {i+1} to {i+2}",16)
        for j in [i,i+1]:
            stage=STAGES[j]
            r.text(f"<b>{j+1}. {stage.replace('_',' ').title()}</b> - run {escape(state['run_id'])}",small=True)
            r.image(evidence/f"{stage}.png",maxheight=292)
            r.y-=7
    r.close();print(f"Created {out} (3 report pages + 4 evidence pages)")
if __name__=="__main__": main()
