"""Screenshot an HTML viewer of verbatim captured logs. This is not a terminal capture.
Run this inside BITS Lab for lab-generated evidence, or capture the lab terminal manually.
"""
import argparse
import hashlib
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import read_json,write_json

STAGES=["ingestion","validation","preparation","transformation","feature_management","model_training","model_evaluation","orchestration"]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);p.add_argument("--browser-executable",help="Optional absolute path to installed Chromium");a=p.parse_args()
    from playwright.sync_api import sync_playwright
    run=Path(a.run).resolve();state=read_json(run/"run.json")
    if state["status"]!="SUCCESS": raise ValueError("Cannot capture successful evidence from a failed workflow")
    out=run/"evidence";out.mkdir(exist_ok=True)
    manifest={"kind":"browser screenshots of verbatim captured execution logs, not terminal screenshots","run_id":state["run_id"],"environment":state["environment"],"files":{}}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,executable_path=a.browser_executable,args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1600,"height":900},device_scale_factor=1.5)
        for index,stage in enumerate(STAGES,1):
            text=(run/"logs"/f"{stage}.log").read_text()
            content=f"""<!doctype html><html><head><meta charset='utf-8'><style>
            *{{box-sizing:border-box}}body{{margin:0;background:#f1f4f8;color:#17223b;font:20px Arial;padding:32px}}
            h1{{font-size:30px;margin:0 0 10px}}.meta{{font-size:18px;color:#42516b;line-height:1.6}}
            pre{{background:#fff;border:1px solid #ced6e0;border-radius:8px;padding:24px;font:18px/1.6 monospace;white-space:pre-wrap;overflow-wrap:anywhere}}
            footer{{font-size:17px;color:#42516b;margin-top:16px}}.status{{color:#14723c;font-weight:bold}}
            </style></head><body><h1>{index:02d} {html.escape(stage.replace('_',' ').title())} <span class='status'>SUCCESS</span></h1>
            <div class='meta'>Execution log viewer | Run {html.escape(state['run_id'])}<br>
            Environment: {html.escape(state['environment'])}<br>Host: {html.escape(state['hostname'])} | UTC start: {html.escape(state['started_utc'])}</div>
            <pre>{html.escape(text)}</pre><footer>Screenshot of the saved execution log. No output text has been added or changed.<br>
            This image records the environment shown above and does not certify BITS Lab execution.</footer></body></html>"""
            (out/f"{stage}.html").write_text(content)
            page.set_content(content,wait_until="load")
            page.screenshot(path=str(out/f"{stage}.png"),full_page=True)
            manifest["files"][stage]={"file":f"{stage}.png","log_sha256":hashlib.sha256(text.encode()).hexdigest(),"png_sha256":hashlib.sha256((out/f"{stage}.png").read_bytes()).hexdigest()}
        browser.close()
    write_json(out/"manifest.json",manifest)
    print(f"Captured 8 log-viewer screenshots to {out}")
if __name__=="__main__": main()
