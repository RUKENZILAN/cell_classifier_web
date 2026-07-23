#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Cell Culture Classifier — Flask Web Interface (single file)

Run:
    pip install flask numpy scikit-learn pillow
    # Optional (for real backbones): pip install torch torchvision
    python app.py
    # Open http://127.0.0.1:5000

Place `agentic_cell_classifier.py` next to this file.
"""

import os
import sys
import json
import time
import queue
import shutil
import tempfile
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template_string, send_file, abort
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import io
import zipfile
import csv


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agentic_cell_classifier import (  # noqa: E402
    PipelineConfig, CentralMemory, CentralAgent,
)


# ---------------------------------------------------------------------------
# Shared run state
# ---------------------------------------------------------------------------
class CancelledError(Exception):
    """Raised inside the pipeline thread to abort a run cooperatively."""


class RunState:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.status = "idle"  # idle | running | completed | failed | cancelled
        self.log_queue = queue.Queue()
        self.log_history = []
        self.result = None
        self.error = None
        self.report_md = None
        self.started_at = None
        self.finished_at = None
        self.tmp_upload_dir = None
        self.cancel_flag = threading.Event()

    def push_log(self, line):
        with self.lock:
            self.log_history.append(line)
        self.log_queue.put(line)

    def reset(self):
        with self.lock:
            self.status = "idle"
            self.log_history = []
            self.result = None
            self.error = None
            self.report_md = None
            self.started_at = None
            self.finished_at = None
        self.cancel_flag.clear()
        try:
            while True:
                self.log_queue.get_nowait()
        except queue.Empty:
            pass


STATE = RunState()
UPLOAD_ROOT = os.path.join(tempfile.gettempdir(), "cell_classifier_uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)


class StreamingMemory(CentralMemory):
    def __init__(self, run_state: RunState):
        super().__init__()
        self._run_state = run_state

    def log(self, message):
        if self._run_state.cancel_flag.is_set():
            # Announce once, then raise to unwind the pipeline stack.
            raise CancelledError("Run cancelled by user.")
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self._lock:
            self.storage["logs"].append(line)
        self._run_state.push_log(line)


def _run_pipeline(config: PipelineConfig):
    STATE.status = "running"
    STATE.started_at = time.time()
    STATE.push_log(f"=== Pipeline started at {datetime.now().isoformat(timespec='seconds')} ===")
    STATE.push_log(f"Data folder: {config.data_dir}")
    STATE.push_log(f"Magnifications: {config.magnifications}")
    STATE.push_log(f"Backbone: {'dummy' if config.use_dummy_extractor else config.backbone}")

    memory = StreamingMemory(STATE)
    try:
        orchestrator = CentralAgent(memory, config)
        orchestrator.run_pipeline()
        metrics = memory.read("metrics", {})
        STATE.result = {
            "metrics": metrics,
            "best": memory.read("best_magnification"),
            "output_dir": os.path.abspath(config.output_dir),
            "significance": memory.read("significance"),
        }
        report_path = os.path.join(config.output_dir, "final_report.md")
        if os.path.exists(report_path):
            STATE.report_md = open(report_path, encoding="utf-8").read()
        STATE.status = "completed"
        STATE.push_log("=== Pipeline completed successfully ===")
    except CancelledError:
        STATE.status = "cancelled"
        STATE.error = "Cancelled by user"
        STATE.push_log("=== Pipeline cancelled by user ===")
    except Exception as e:
        STATE.error = f"{type(e).__name__}: {e}"
        STATE.status = "failed"
        STATE.push_log(f"!!! Pipeline failed: {STATE.error}")
        STATE.push_log(traceback.format_exc())
    finally:
        STATE.finished_at = time.time()
        STATE.push_log("__END__")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB uploads


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    return jsonify(ok=False, error="Upload is too large. Use the server path option instead, or reduce the folder size."), 413


@app.after_request
def add_local_ui_headers(response):
    # Allows the optional standalone index.html file to call this local Flask server.
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cell Culture Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0b0f16;
    --bg-2: #0f1520;
    --panel: #141b26;
    --panel-2: #1b2432;
    --panel-3: #212c3d;
    --border: #26324a;
    --border-2: #334263;
    --text: #eaf0f8;
    --muted: #8896ab;
    --muted-2: #5f6b80;
    --accent: #6ea8ff;
    --accent-2: #22d3a5;
    --accent-3: #b78bff;
    --danger: #ff6b7a;
    --warn: #f5c451;
    --grad: linear-gradient(135deg, #6ea8ff, #b78bff);
    --grad-2: linear-gradient(135deg, #22d3a5, #6ea8ff);
    --shadow: 0 8px 24px rgba(0,0,0,0.35), 0 2px 4px rgba(0,0,0,0.2);
    --radial-1: #182238;
    --radial-2: #1a1631;
    --header-bg: rgba(11,15,22,0.6);
    --code-bg: #06090f;
    --chart-grid: #26324a;
    --chart-tick: #8896ab;
    --chart-text: #eaf0f8;
  }
  html[data-theme="light"] {
    --bg: #f4f6fb;
    --bg-2: #eef1f8;
    --panel: #ffffff;
    --panel-2: #f3f5fa;
    --panel-3: #e9edf5;
    --border: #dde3ee;
    --border-2: #cbd4e6;
    --text: #131b2c;
    --muted: #5b6580;
    --muted-2: #808ba5;
    --accent: #3f7de0;
    --accent-2: #10a37f;
    --accent-3: #8a5cf6;
    --danger: #e0405a;
    --warn: #b8860b;
    --grad: linear-gradient(135deg, #3f7de0, #8a5cf6);
    --grad-2: linear-gradient(135deg, #10a37f, #3f7de0);
    --shadow: 0 8px 24px rgba(20,30,60,0.10), 0 2px 4px rgba(20,30,60,0.06);
    --radial-1: #dbe6ff;
    --radial-2: #e6ddff;
    --header-bg: rgba(255,255,255,0.75);
    --code-bg: #eef1f6;
    --chart-grid: #dde3ee;
    --chart-tick: #5b6580;
    --chart-text: #131b2c;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: radial-gradient(1200px 800px at 15% -10%, var(--radial-1) 0%, transparent 60%),
                radial-gradient(900px 700px at 110% 10%, var(--radial-2) 0%, transparent 55%),
                var(--bg);
    color: var(--text);
    min-height: 100vh;
    transition: background-color .2s ease, color .2s ease;
  }
  code, pre, .mono { font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; }

  header {
    padding: 22px 32px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(10px);
    background: var(--header-bg);
    position: sticky; top: 0; z-index: 20;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand .logo {
    width: 36px; height: 36px; border-radius: 10px;
    background: var(--grad); display: grid; place-items: center;
    font-size: 20px; box-shadow: var(--shadow);
  }
  .brand h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0.2px; }
  .brand p { margin: 2px 0 0; font-size: 12px; color: var(--muted); }

  .pill {
    padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 500;
    background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
    display: inline-flex; align-items: center; gap: 6px;
  }
  .pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
  .pill.running { color: var(--warn); border-color: rgba(245,196,81,0.3); }
  .pill.running .dot { background: var(--warn); animation: pulse 1s infinite; }
  .pill.completed { color: var(--accent-2); border-color: rgba(34,211,165,0.35); }
  .pill.completed .dot { background: var(--accent-2); }
  .pill.failed { color: var(--danger); border-color: rgba(255,107,122,0.35); }
  .pill.failed .dot { background: var(--danger); }
  .pill.cancelled { color: var(--warn); border-color: rgba(245,196,81,0.35); }
  .pill.cancelled .dot { background: var(--warn); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  .theme-toggle {
    width: 36px; height: 36px; padding: 0; margin: 0;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 10px; border: 1px solid var(--border);
    background: var(--panel-2); color: var(--text);
    font-size: 16px; line-height: 1; cursor: pointer;
    box-shadow: none; transition: background-color .15s, transform .1s, border-color .15s;
  }
  .theme-toggle:hover { border-color: var(--border-2); transform: translateY(-1px); }
  .theme-toggle:active { transform: translateY(0); }

  main {
    display: grid; grid-template-columns: 400px 1fr; gap: 22px;
    padding: 22px 32px 60px; max-width: 1600px; margin: 0 auto;
  }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; padding: 18px; } }

  .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent), var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 20px;
    box-shadow: var(--shadow);
  }
  .card h2 {
    margin: 0 0 16px; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--muted); font-weight: 600;
    display: flex; align-items: center; gap: 8px;
  }
  .card h2::before {
    content: ""; width: 3px; height: 12px; background: var(--grad); border-radius: 2px;
  }

  .sidebar { display: flex; flex-direction: column; gap: 18px; position: sticky; top: 90px; align-self: start; }
  @media (max-width: 1000px) { .sidebar { position: static; } }
  .content { display: flex; flex-direction: column; gap: 18px; }

  /* Folder picker */
  .dropzone {
    border: 2px dashed var(--border-2);
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background: var(--panel-2);
  }
  .dropzone:hover, .dropzone.hover {
    border-color: var(--accent);
    background: rgba(110,168,255,0.06);
  }
  .dropzone .icon { font-size: 32px; margin-bottom: 8px; }
  .dropzone .title { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
  .dropzone .hint { font-size: 12px; color: var(--muted); }
  .dropzone.filled { border-style: solid; border-color: var(--accent-2); background: rgba(34,211,165,0.05); }
  .folder-summary { margin-top: 10px; font-size: 12px; color: var(--muted); }
  .folder-summary strong { color: var(--text); }

  .divider {
    display: flex; align-items: center; gap: 10px;
    color: var(--muted-2); font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    margin: 16px 0;
  }
  .divider::before, .divider::after { content: ""; flex: 1; height: 1px; background: var(--border); }

  label { display: block; font-size: 12px; margin: 12px 0 6px; color: var(--muted); font-weight: 500; }
  input[type=text], input[type=number], select {
    width: 100%; padding: 10px 12px; border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--panel-2); color: var(--text);
    font-size: 13.5px; font-family: inherit;
    transition: border-color 0.15s;
  }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: 10px; }
  .row > * { flex: 1; }
  .checkbox { display: flex; align-items: center; gap: 8px; margin-top: 14px; font-size: 13px; color: var(--muted); cursor: pointer; }
  .checkbox input { accent-color: var(--accent); }

  button {
    background: var(--grad); color: #0b1220; border: 0; padding: 12px 16px;
    border-radius: 10px; font-weight: 600; cursor: pointer; font-size: 14px;
    width: 100%; margin-top: 18px; font-family: inherit;
    transition: transform 0.05s, opacity 0.15s;
    box-shadow: 0 6px 18px rgba(110,168,255,0.25);
  }
  button:hover:not(:disabled) { transform: translateY(-1px); }
  button:active:not(:disabled) { transform: translateY(0); }
  button:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none; }
  button.secondary {
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); box-shadow: none;
    margin-top: 8px;
  }
  button.secondary:hover:not(:disabled) { background: var(--panel-3); }

  .progress {
    height: 6px; background: var(--panel-3); border-radius: 3px; overflow: hidden; margin-top: 12px;
    display: none;
  }
  .progress.active { display: block; }
  .progress .bar { height: 100%; width: 0%; background: var(--grad-2); transition: width 0.2s; }

  #logbox {
    background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px; height: 360px; overflow-y: auto;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; line-height: 1.6; white-space: pre-wrap;
  }
  #logbox div { padding: 1px 0; }
  #logbox .err { color: var(--danger); }
  #logbox .ok { color: var(--accent-2); }
  #logbox .info { color: var(--accent); }
  #logbox .dim { color: var(--muted-2); }
  #logbox::-webkit-scrollbar { width: 8px; }
  #logbox::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 4px; }

  .empty {
    color: var(--muted); font-size: 13px; padding: 30px 16px; text-align: center;
    border: 1px dashed var(--border); border-radius: 8px;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; }
  td.num { font-variant-numeric: tabular-nums; text-align: right; }
  tr.best td:first-child { color: var(--accent-2); font-weight: 600; }
  tr.best { background: rgba(34,211,165,0.04); }

  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
  .chart-wrap { background: var(--panel-2); border-radius: 10px; padding: 14px; border: 1px solid var(--border); }
  .chart-wrap.white { background: #ffffff; }
  .chart-wrap.white canvas { background: #ffffff; }

  .cm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
  .cm-block { background: var(--panel-2); border-radius: 10px; padding: 14px; border: 1px solid var(--border); }
  .cm-block.white { background: #ffffff; color: #111; }
  .cm-block.white h3, .cm-block.white .sub { color: #333; }
  .cm-block h3 { margin: 0 0 10px; font-size: 13px; color: var(--text); font-weight: 600; }
  .cm-block .sub { font-size: 11px; color: var(--muted); margin-bottom: 10px; }
  .cm-table td, .cm-table th { text-align: center; padding: 8px 10px; font-size: 13px; }
  .cm-table td.diag { background: rgba(34,211,165,0.18); color: var(--accent-2); font-weight: 600; }
  .cm-block.white .cm-table td, .cm-block.white .cm-table th { border-color: #ddd; color: #111; }
  .cm-block.white .cm-table td.diag { background: #d1fae5; color: #047857; }

  .dl-btn {
    display: inline-block; margin-top: 10px; padding: 6px 12px; border-radius: 8px;
    background: var(--panel-3); color: var(--text); text-decoration: none; font-size: 12px;
    border: 1px solid var(--border-2); cursor: pointer;
  }
  .dl-btn:hover { background: var(--border-2); }
  .toolbar {
    display: flex; gap: 12px; align-items: center; margin-bottom: 12px;
    font-size: 13px; color: var(--muted);
  }
  .toolbar label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }

  pre.report {
    background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; overflow-x: auto; font-size: 12px; line-height: 1.6;
    color: var(--text);
  }

  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .kpi {
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px;
  }
  .kpi .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; }
  .kpi .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .kpi.best .value { background: var(--grad-2); -webkit-background-clip: text; background-clip: text; color: transparent; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">🧫</div>
    <div>
      <h1>Cell Culture Classifier</h1>
      <p>Agentic pipeline · x5 / x20 magnification comparison</p>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <button type="button" id="themeToggle" class="theme-toggle" title="Toggle light / dark background" aria-label="Toggle light / dark background">🌙</button>
    <span class="pill" id="headerStatus"><span class="dot"></span><span id="headerStatusText">idle</span></span>
  </div>
</header>

<main>
  <aside class="sidebar">
    <section class="card">
      <h2>1. Select Data</h2>

      <div id="dropzone" class="dropzone">
        <div class="icon">📁</div>
        <div class="title">Choose a folder</div>
        <div class="hint">Click or drop your <code>data/</code> folder<br/>(must contain <code>x5/</code>, <code>x20/</code> …)</div>
        <input type="file" id="folderInput" webkitdirectory directory multiple style="display:none" />
      </div>
      <div class="folder-summary" id="folderSummary"></div>
      <div class="progress" id="uploadProgress"><div class="bar"></div></div>

      <div class="divider">or use a server path</div>

      <input type="text" id="data_dir" placeholder="/absolute/path/to/data" />
    </section>

    <section class="card">
      <h2>2. Configure</h2>

      <label for="magnifications">Magnifications (auto-detected if empty)</label>
      <input type="text" id="magnifications" placeholder="x5 x20" />

      <div class="row">
        <div>
          <label for="backbone">Backbone</label>
          <select id="backbone">
            <option value="resnet50">ResNet-50</option>
            <option value="vgg16">VGG-16</option>
            <option value="dummy">Dummy (no torch)</option>
          </select>
        </div>
        <div>
          <label for="knn">KNN k</label>
          <input type="number" id="knn" value="5" min="1" max="50" />
        </div>
      </div>

      <label for="test_size">Test size</label>
      <input type="text" id="test_size" value="0.2" />

      <label>Output folder</label>
      <input type="text" id="output_dir" placeholder="/absolute/path/to/output (leave empty for ./outputs)" />
      <div style="display:flex; gap:8px; margin-top:6px; flex-wrap:wrap;">
        <button type="button" id="pickOutBtn" class="secondary" style="flex:1; min-width:140px;">📂 Pick folder…</button>
        <button type="button" id="defaultOutBtn" class="secondary" style="flex:1; min-width:140px;">Use ./outputs</button>
      </div>
      <div class="folder-summary" id="outFolderSummary"></div>

      <label class="checkbox">
        <input type="checkbox" id="use_dummy" />
        Force dummy extractor (no torch required)
      </label>

      <button id="runBtn">▶ Run Pipeline</button>
      <button id="cancelBtn" class="secondary" disabled>■ Cancel Run</button>
      <button id="resetBtn" class="secondary">Reset</button>
    </section>
  </aside>

  <section class="content">
    <div class="card">
      <h2>Live Log Stream</h2>
      <div id="logbox"><div class="dim">Waiting for a run…</div></div>
    </div>

    <div class="card">
      <h2>Results</h2>
      <div id="resultsArea"><div class="empty">Run the pipeline to see metrics.</div></div>
    </div>

    <div class="card">
      <h2>Charts</h2>
      <div class="toolbar">
        <label><input type="checkbox" id="whiteBgToggle"> White background</label>
        <button class="dl-btn" id="dlAllChartsBtn" type="button">⬇ Download all images (ZIP)</button>
      </div>
      <div class="charts">
        <div class="chart-wrap" id="accWrap">
          <canvas id="accChart" height="200"></canvas>
          <button class="dl-btn" data-chart="acc" type="button">⬇ Download PNG</button>
        </div>
        <div class="chart-wrap" id="f1Wrap">
          <canvas id="f1Chart" height="200"></canvas>
          <button class="dl-btn" data-chart="f1" type="button">⬇ Download PNG</button>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Confusion Matrices</h2>
      <div id="cmArea"><div class="empty">No results yet.</div></div>
    </div>

    <div class="card">
      <h2>Final Report</h2>
      <div id="reportArea"><div class="empty">Report will appear here after a successful run.</div></div>
    </div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);

// --- Light / dark theme toggle ---------------------------------------------
const THEME_KEY = "cellClassifierTheme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = $("themeToggle");
  if (btn) btn.textContent = theme === "light" ? "🌙" : "☀️";
  if (typeof lastResult !== "undefined" && lastResult) {
    drawCharts(lastResult);
  }
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (e) {}
  if (!saved) {
    saved = (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark";
  }
  applyTheme(saved);
})();
document.addEventListener("DOMContentLoaded", () => {
  const btn = $("themeToggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "light" ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
  });
});

const logbox = $("logbox");
let accChart = null, f1Chart = null;
let whiteBg = false;
let lastResult = null;
let eventSource = null;
let selectedFiles = null;
let uploadedServerPath = null;
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:5000" : "";
const apiUrl = (path) => API_BASE + path;

function appendLog(line) {
  const div = document.createElement("div");
  if (/error|failed|!!!|traceback/i.test(line)) div.className = "err";
  else if (/completed|success|===/.test(line)) div.className = "ok";
  else if (/Memory Updated|Pipeline|Agent:|tamamlandı|başlıyor/.test(line)) div.className = "info";
  div.textContent = line;
  logbox.appendChild(div);
  logbox.scrollTop = logbox.scrollHeight;
}

function setStatus(status, text) {
  const pill = $("headerStatus");
  pill.className = "pill " + status;
  $("headerStatusText").textContent = text || status;
}

function humanBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
  if (n < 1024*1024*1024) return (n/1024/1024).toFixed(1) + " MB";
  return (n/1024/1024/1024).toFixed(2) + " GB";
}

// --- Folder picker & upload ------------------------------------------------
const dropzone = $("dropzone");
const folderInput = $("folderInput");

dropzone.addEventListener("click", () => folderInput.click());
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("hover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("hover"));
dropzone.addEventListener("drop", async e => {
  e.preventDefault();
  dropzone.classList.remove("hover");
  const items = e.dataTransfer.items;
  const files = [];
  const walkers = [];
  for (const it of items) {
    const entry = it.webkitGetAsEntry && it.webkitGetAsEntry();
    if (entry) walkers.push(walkEntry(entry, "", files));
  }
  await Promise.all(walkers);
  handleFiles(files);
});
folderInput.addEventListener("change", e => {
  handleFiles(Array.from(e.target.files));
});

// --- Output folder picker -------------------------------------------------
const outSummary = $("outFolderSummary");
const outInput = $("output_dir");

$("defaultOutBtn").addEventListener("click", () => {
  outInput.value = "./outputs";
  outSummary.innerHTML = `Output → <strong>./outputs</strong> (next to app.py)`;
});

$("pickOutBtn").addEventListener("click", async () => {
  if (window.showDirectoryPicker) {
    try {
      const handle = await window.showDirectoryPicker({ mode: "read" });
      await resolveOutputFolder(handle.name);
      return;
    } catch (e) {
      if (e && e.name === "AbortError") return;
    }
  }
  const p = prompt("Your browser can't pick folders. Paste the ABSOLUTE path to the output folder:", outInput.value || "");
  if (p && p.trim()) {
    outInput.value = p.trim();
    outSummary.innerHTML = `Output → <strong>${p.trim()}</strong>`;
  }
});

outInput.addEventListener("input", () => {
  const v = outInput.value.trim();
  outSummary.innerHTML = v ? `Output → <strong>${v}</strong>` : "";
});

async function resolveOutputFolder(name) {
  try {
    const r = await fetch(apiUrl("/api/resolve-output"), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ name })
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "resolve failed");
    outInput.value = d.path;
    outSummary.innerHTML = `Output → <strong>${d.path}</strong>`;
  } catch (e) {
    outSummary.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
  }
}

function walkEntry(entry, path, out) {
  return new Promise(resolve => {
    if (entry.isFile) {
      entry.file(f => {
        f.relativePath = path + f.name;
        out.push(f);
        resolve();
      });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const all = [];
      const readBatch = () => reader.readEntries(async batch => {
        if (!batch.length) {
          await Promise.all(all);
          resolve();
        } else {
          for (const e of batch) all.push(walkEntry(e, path + entry.name + "/", out));
          readBatch();
        }
      });
      readBatch();
    } else resolve();
  });
}

function handleFiles(files) {
  const exts = /\.(png|jpe?g|tiff?|bmp)$/i;
  const imageFiles = files.filter(f => exts.test(f.name));
  if (window.location.protocol === "file:" && !API_BASE) {
    $("folderSummary").innerHTML = `<span style="color:var(--danger)">Start the app with <code>python app.py</code> and open <code>http://127.0.0.1:5000</code>.</span>`;
    return;
  }
  if (!imageFiles.length) {
    $("folderSummary").innerHTML = `<span style="color:var(--danger)">No image files found.</span>`;
    return;
  }
  // Ensure each file has a relativePath
  for (const f of imageFiles) {
    if (!f.relativePath) f.relativePath = f.webkitRelativePath || f.name;
  }
  selectedFiles = imageFiles;
  uploadedServerPath = null;
  dropzone.classList.add("filled");

  // Detect magnifications from top-level subfolders (2nd path segment)
  const mags = new Set();
  let totalSize = 0;
  for (const f of imageFiles) {
    totalSize += f.size;
    const parts = f.relativePath.split("/");
    if (parts.length >= 3) mags.add(parts[1]);
    else if (parts.length >= 2) mags.add(parts[0]);
  }
  const magList = Array.from(mags).sort();
  $("magnifications").value = magList.join(" ");
  $("folderSummary").innerHTML =
    `<strong>${imageFiles.length}</strong> images · <strong>${humanBytes(totalSize)}</strong>` +
    (magList.length ? ` · magnifications: <strong>${magList.join(", ")}</strong>` : "");
}

async function uploadSelectedFolder() {
  if (!selectedFiles || !selectedFiles.length) return null;
  const prog = $("uploadProgress");
  const bar = prog.querySelector(".bar");
  prog.classList.add("active");
  bar.style.width = "0%";

  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const f of selectedFiles) {
      form.append("files", f, f.relativePath || f.webkitRelativePath || f.name);
      form.append("paths", f.relativePath);
    }
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/api/upload"));
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) bar.style.width = (e.loaded / e.total * 100).toFixed(1) + "%";
    };
    xhr.onload = () => {
      prog.classList.remove("active");
      if (xhr.status < 200 || xhr.status >= 300) {
        let msg = `upload failed (HTTP ${xhr.status})`;
        try {
          const d = JSON.parse(xhr.responseText || "{}");
          if (d.error) msg = d.error;
        } catch (_) {
          if (xhr.responseText) msg += `: ${xhr.responseText.slice(0, 300)}`;
        }
        reject(new Error(msg));
        return;
      }
      try {
        const d = JSON.parse(xhr.responseText);
        if (!d.ok) reject(new Error(d.error || "upload failed"));
        else resolve(d.path);
      } catch (e) { reject(new Error(`upload failed: ${e.message}`)); }
    };
    xhr.onabort = () => { prog.classList.remove("active"); reject(new Error("upload cancelled")); };
    xhr.onerror = () => { prog.classList.remove("active"); reject(new Error("upload failed — make sure the Flask app is running at http://127.0.0.1:5000")); };
    xhr.send(form);
  });
}

// --- SSE log stream --------------------------------------------------------
function startStream() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(apiUrl("/api/logs"));
  eventSource.onmessage = (e) => {
    if (e.data === "__END__") { closeStream(); fetchResults(); return; }
    appendLog(e.data);
  };
  eventSource.onerror = () => {};
}

function closeStream() {
  if (eventSource) { eventSource.close(); eventSource = null; }
}

async function fetchResults() {
  const r = await fetch(apiUrl("/api/status"));
  const d = await r.json();
  setStatus(d.status, d.status);
  if (d.status === "completed" && d.result) renderResults(d.result, d.report_md);
  else if (d.status === "failed") {
    $("resultsArea").innerHTML = `<div class="empty" style="color:var(--danger)">Failed: ${d.error || "unknown error"}</div>`;
  } else if (d.status === "cancelled") {
    $("resultsArea").innerHTML = `<div class="empty" style="color:var(--warn)">Run cancelled by user.</div>`;
    $("cmArea").innerHTML = `<div class="empty">No results (cancelled).</div>`;
    $("reportArea").innerHTML = `<div class="empty">No report (cancelled).</div>`;
  }
  $("runBtn").disabled = false;
  $("cancelBtn").disabled = true;
}

function renderResults(result, reportMd) {
  const metrics = result.metrics || {};
  const mags = Object.keys(metrics);
  if (!mags.length) return;
  lastResult = result;

  // KPI cards
  let kpi = `<div class="kpi-grid">`;
  const best = result.best;
  const bestM = metrics[best] || metrics[mags[0]];
  kpi += `<div class="kpi best"><div class="label">Best</div><div class="value">${best || "—"}</div></div>`;
  kpi += `<div class="kpi"><div class="label">Accuracy</div><div class="value">${(bestM.accuracy*100).toFixed(1)}%</div></div>`;
  kpi += `<div class="kpi"><div class="label">F1 (weighted)</div><div class="value">${(bestM.f1_score*100).toFixed(1)}%</div></div>`;
  kpi += `<div class="kpi"><div class="label">Test samples</div><div class="value">${bestM.n_test_samples}</div></div>`;
  kpi += `</div>`;

  let html = kpi + `<table><thead><tr><th>Magnification</th><th>Accuracy</th><th>F1</th><th>Train</th><th>Test</th><th>Backbone</th></tr></thead><tbody>`;
  mags.forEach(m => {
    const x = metrics[m];
    const isBest = m === result.best;
    html += `<tr class="${isBest?'best':''}"><td>${m}${isBest?' ★':''}</td>`
         +  `<td class="num">${(x.accuracy*100).toFixed(2)}%</td>`
         +  `<td class="num">${(x.f1_score*100).toFixed(2)}%</td>`
         +  `<td class="num">${x.n_train_samples}</td>`
         +  `<td class="num">${x.n_test_samples}</td>`
         +  `<td>${x.backbone}</td></tr>`;
  });
  html += `</tbody></table>`;
  const outEnc = encodeURIComponent(result.output_dir);
  const btnCss = "padding:8px 14px; border-radius:8px; background:var(--accent); color:#ffffff; font-weight:600; text-decoration:none; font-size:13px;";
  const btnCss2 = "padding:8px 14px; border-radius:8px; background:var(--panel-3); color:var(--text); text-decoration:none; font-size:13px; border:1px solid var(--border-2);";
  html += `<div style="margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
    <a href="${apiUrl("/api/export/zip")}?path=${outEnc}" download style="${btnCss}">⬇ Download all results (ZIP)</a>
    <a href="${apiUrl("/api/export/metrics.csv")}?path=${outEnc}" download style="${btnCss2}">⬇ Metrics CSV</a>
    <a href="${apiUrl("/api/export/results.json")}?path=${outEnc}" download style="${btnCss2}">⬇ Results JSON</a>
    <a href="${apiUrl("/api/export/report.md")}?path=${outEnc}" download style="${btnCss2}">⬇ Report (MD)</a>
  </div>`;
  html += `<div style="margin-top:10px; font-size:12px; color:var(--muted);">Output: <code>${result.output_dir}</code></div>`;
  $("resultsArea").innerHTML = html;

  drawCharts(result);
  drawConfusionMatrices(result);

  if (reportMd) {
    $("reportArea").innerHTML = `<pre class="report">${reportMd.replace(/[<>&]/g, s=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[s]))}</pre>`;
  }
}

function chartTheme() {
  if (whiteBg) return { text: "#111", tick: "#444", grid: "#e5e7eb" };
  if (document.documentElement.getAttribute("data-theme") === "light") {
    return { text: "#131b2c", tick: "#5b6580", grid: "#dde3ee" };
  }
  return { text: "#eaf0f8", tick: "#8896ab", grid: "#26324a" };
}

function drawCharts(result) {
  const metrics = result.metrics || {};
  const mags = Object.keys(metrics);
  const labels = mags;
  const accData = mags.map(m => +(metrics[m].accuracy*100).toFixed(2));
  const f1Data  = mags.map(m => +(metrics[m].f1_score*100).toFixed(2));
  if (accChart) accChart.destroy();
  if (f1Chart) f1Chart.destroy();
  const th = chartTheme();
  const chartOpts = (title) => ({
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: title, color: th.text, font: { size: 13, weight: '600' } }
    },
    scales: {
      y: { beginAtZero: true, max: 100, ticks: { color: th.tick }, grid: { color: th.grid } },
      x: { ticks: { color: th.tick }, grid: { display: false } }
    }
  });
  accChart = new Chart($("accChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "Accuracy (%)", data: accData, backgroundColor: "#6ea8ff", borderRadius: 6 }]},
    options: chartOpts("Accuracy (%)")
  });
  f1Chart = new Chart($("f1Chart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "F1 (%)", data: f1Data, backgroundColor: "#22d3a5", borderRadius: 6 }]},
    options: chartOpts("F1 Score (%)")
  });
}

function drawConfusionMatrices(result) {
  const metrics = result.metrics || {};
  const mags = Object.keys(metrics);
  const bg = whiteBg ? "white" : "";
  let cm = "";
  mags.forEach(m => {
    const x = metrics[m];
    cm += `<div class="cm-block ${bg}"><h3>${m}</h3><div class="sub">classes: ${x.class_names.join(", ")}</div>`;
    cm += `<table class="cm-table"><thead><tr><th></th>`;
    x.class_names.forEach(c => cm += `<th>Pred: ${c}</th>`);
    cm += `</tr></thead><tbody>`;
    x.confusion_matrix.forEach((row, i) => {
      cm += `<tr><th>True: ${x.class_names[i]}</th>`;
      row.forEach((v, j) => cm += `<td class="${i===j?'diag':''}">${v}</td>`);
      cm += `</tr>`;
    });
    cm += `</tbody></table>`;
    cm += `<button class="dl-btn" data-cm="${m}" type="button">⬇ Download PNG</button>`;
    cm += `</div>`;
  });
  $("cmArea").innerHTML = `<div class="cm-grid">${cm}</div>`;
  // update chart-wrap background class
  $("accWrap").classList.toggle("white", whiteBg);
  $("f1Wrap").classList.toggle("white", whiteBg);
}

// --- Image downloads -------------------------------------------------------
function canvasWithBg(srcCanvas) {
  const c = document.createElement("canvas");
  c.width = srcCanvas.width; c.height = srcCanvas.height;
  const ctx = c.getContext("2d");
  ctx.fillStyle = whiteBg ? "#ffffff" : "#0f1521";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(srcCanvas, 0, 0);
  return c;
}
function triggerDownload(dataUrl, name) {
  const a = document.createElement("a");
  a.href = dataUrl; a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
function downloadChart(which) {
  const chart = which === "acc" ? accChart : f1Chart;
  if (!chart) return;
  const c = canvasWithBg(chart.canvas);
  triggerDownload(c.toDataURL("image/png"), `${which}_chart.png`);
}
function renderCMCanvas(mag, x) {
  const cell = 70, pad = 90, top = 60;
  const n = x.class_names.length;
  const w = pad + cell * n + 20;
  const h = top + cell * n + 40;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  const fg = whiteBg ? "#111" : "#eaf0f8";
  const muted = whiteBg ? "#555" : "#8896ab";
  ctx.fillStyle = whiteBg ? "#ffffff" : "#0f1521";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = fg;
  ctx.font = "bold 16px system-ui, sans-serif";
  ctx.fillText(`Confusion Matrix — ${mag}`, 12, 26);
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = muted;
  ctx.fillText(`classes: ${x.class_names.join(", ")}`, 12, 46);
  // find max for color scaling
  let max = 1;
  x.confusion_matrix.forEach(r => r.forEach(v => { if (v > max) max = v; }));
  // headers
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = fg;
  ctx.textAlign = "center";
  for (let j = 0; j < n; j++) {
    ctx.fillText(x.class_names[j], pad + cell*j + cell/2, top - 8);
  }
  ctx.textAlign = "right";
  for (let i = 0; i < n; i++) {
    ctx.fillText(x.class_names[i], pad - 8, top + cell*i + cell/2 + 4);
  }
  // cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = x.confusion_matrix[i][j];
      const t = v / max;
      // teal for diag, blue for off-diag
      if (i === j) {
        ctx.fillStyle = whiteBg
          ? `rgba(4, 120, 87, ${0.15 + 0.75*t})`
          : `rgba(34, 211, 165, ${0.15 + 0.75*t})`;
      } else {
        ctx.fillStyle = whiteBg
          ? `rgba(59, 130, 246, ${0.10 + 0.55*t})`
          : `rgba(110, 168, 255, ${0.10 + 0.55*t})`;
      }
      ctx.fillRect(pad + cell*j, top + cell*i, cell - 2, cell - 2);
      ctx.fillStyle = fg;
      ctx.textAlign = "center";
      ctx.font = "bold 14px system-ui, sans-serif";
      ctx.fillText(String(v), pad + cell*j + cell/2, top + cell*i + cell/2 + 5);
    }
  }
  return c;
}
function downloadCM(mag) {
  if (!lastResult) return;
  const x = lastResult.metrics[mag];
  if (!x) return;
  const c = renderCMCanvas(mag, x);
  triggerDownload(c.toDataURL("image/png"), `confusion_matrix_${mag}.png`);
}
async function downloadAllImagesZip() {
  if (!lastResult) { alert("Run the pipeline first."); return; }
  // Load JSZip on demand
  if (!window.JSZip) {
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
      s.onload = res; s.onerror = () => rej(new Error("Failed to load JSZip"));
      document.head.appendChild(s);
    });
  }
  const zip = new JSZip();
  const toBlob = (canvas) => new Promise(r => canvas.toBlob(r, "image/png"));
  if (accChart) zip.file("accuracy_chart.png", await toBlob(canvasWithBg(accChart.canvas)));
  if (f1Chart)  zip.file("f1_chart.png",       await toBlob(canvasWithBg(f1Chart.canvas)));
  const metrics = lastResult.metrics || {};
  for (const mag of Object.keys(metrics)) {
    zip.file(`confusion_matrix_${mag}.png`, await toBlob(renderCMCanvas(mag, metrics[mag])));
  }
  const blob = await zip.generateAsync({ type: "blob" });
  triggerDownload(URL.createObjectURL(blob), "cell_classifier_images.zip");
}

// Delegated clicks for per-image download buttons
document.addEventListener("click", (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  if (t.dataset && t.dataset.chart) downloadChart(t.dataset.chart);
  else if (t.dataset && t.dataset.cm) downloadCM(t.dataset.cm);
});
$("dlAllChartsBtn").addEventListener("click", () => downloadAllImagesZip());
$("whiteBgToggle").addEventListener("change", (e) => {
  whiteBg = e.target.checked;
  if (lastResult) { drawCharts(lastResult); drawConfusionMatrices(lastResult); }
  else {
    $("accWrap").classList.toggle("white", whiteBg);
    $("f1Wrap").classList.toggle("white", whiteBg);
  }
});

// --- Run -------------------------------------------------------------------
$("runBtn").onclick = async () => {
  const runBtn = $("runBtn");
  const cancelBtn = $("cancelBtn");
  runBtn.disabled = true;
  cancelBtn.disabled = false;

  let dataDir = $("data_dir").value.trim();
  logbox.innerHTML = "";
  $("resultsArea").innerHTML = `<div class="empty">Running…</div>`;
  $("cmArea").innerHTML = `<div class="empty">Running…</div>`;
  $("reportArea").innerHTML = `<div class="empty">Running…</div>`;
  setStatus("running", "running");

  // Upload folder if needed
  if (!dataDir && selectedFiles && !uploadedServerPath) {
    appendLog(`[web] Uploading ${selectedFiles.length} files…`);
    try {
      uploadedServerPath = await uploadSelectedFolder();
      appendLog(`[web] Upload complete → ${uploadedServerPath}`);
    } catch (e) {
      appendLog("Upload failed: " + e.message);
      setStatus("failed", "upload failed");
      runBtn.disabled = false; cancelBtn.disabled = true; return;
    }
  }
  if (!dataDir && uploadedServerPath) dataDir = uploadedServerPath;
  if (!dataDir) {
    appendLog("Please select a folder or enter a server path.");
    setStatus("failed", "no data");
    runBtn.disabled = false; cancelBtn.disabled = true; return;
  }

  const payload = {
    data_dir: dataDir,
    magnifications: $("magnifications").value.trim().split(/\s+/).filter(Boolean),
    backbone: $("backbone").value,
    knn_neighbors: parseInt($("knn").value, 10) || 5,
    test_size: parseFloat($("test_size").value) || 0.2,
    output_dir: $("output_dir").value.trim() || "./outputs",
    use_dummy: $("use_dummy").checked,
  };
  try {
    const r = await fetch(apiUrl("/api/run"), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (!d.ok) {
      appendLog("Error: " + d.error);
      setStatus("failed", d.error);
      runBtn.disabled = false; cancelBtn.disabled = true; return;
    }
    startStream();
  } catch (e) {
    appendLog("Request failed: " + e);
    setStatus("failed", "request failed");
    runBtn.disabled = false; cancelBtn.disabled = true;
  }
};

$("cancelBtn").onclick = async () => {
  const cancelBtn = $("cancelBtn");
  cancelBtn.disabled = true;
  appendLog("[web] Cancelling run…");
  try {
    const r = await fetch(apiUrl("/api/cancel"), { method: "POST" });
    const d = await r.json();
    if (!d.ok) appendLog("[web] Cancel error: " + d.error);
  } catch (e) {
    appendLog("[web] Cancel request failed: " + e);
  }
  // Stream will close itself when the pipeline thread emits __END__.
};

$("resetBtn").onclick = async () => {
  await fetch(apiUrl("/api/reset"), { method: "POST" });
  logbox.innerHTML = `<div class="dim">Waiting for a run…</div>`;
  $("resultsArea").innerHTML = `<div class="empty">Run the pipeline to see metrics.</div>`;
  $("cmArea").innerHTML = `<div class="empty">No results yet.</div>`;
  $("reportArea").innerHTML = `<div class="empty">Report will appear here after a successful run.</div>`;
  if (accChart) { accChart.destroy(); accChart = null; }
  if (f1Chart) { f1Chart.destroy(); f1Chart = null; }
  selectedFiles = null; uploadedServerPath = null;
  dropzone.classList.remove("filled");
  $("folderSummary").innerHTML = "";
  $("outFolderSummary").innerHTML = "";
  $("output_dir").value = "";
  setStatus("idle", "idle");
};

// Resume on page load
(async () => {
  const r = await fetch(apiUrl("/api/status"));
  const d = await r.json();
  if ((d.log_history || []).length) {
    logbox.innerHTML = "";
    d.log_history.forEach(appendLog);
  }
  setStatus(d.status, d.status);
  if (d.status === "running") {
    $("runBtn").disabled = true;
    $("cancelBtn").disabled = false;
    startStream();
  } else if (d.status === "completed" && d.result) {
    renderResults(d.result, d.report_md);
  }
})();
</script>
</body>
</html>

"""


@app.route("/")
def index():
    # Prefer the on-disk index.html so edits show up without touching app.py.
    ext = Path(__file__).parent / "index.html"
    if ext.exists():
        return ext.read_text(encoding="utf-8")
    return render_template_string(INDEX_HTML)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        files = request.files.getlist("files")
        paths = request.form.getlist("paths")
        if not files:
            return jsonify(ok=False, error="No files uploaded. Select the data folder again and try Run Pipeline."), 400
        if len(paths) != len(files):
            paths = [f.filename for f in files]

        run_dir = tempfile.mkdtemp(prefix=f"upload_{int(time.time())}_", dir=UPLOAD_ROOT)
        saved = 0
        for f, rel in zip(files, paths):
            parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".", "..")]
            if not parts:
                continue
            safe_parts = [secure_filename(p) or "x" for p in parts]
            dest = os.path.join(run_dir, *safe_parts)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            f.save(dest)
            saved += 1

        if not saved:
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify(ok=False, error="Upload contained no usable files."), 400

        entries = os.listdir(run_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(run_dir, entries[0])):
            run_dir = os.path.join(run_dir, entries[0])

        STATE.tmp_upload_dir = run_dir
        return jsonify(ok=True, path=run_dir, files=saved)
    except Exception as exc:
        return jsonify(ok=False, error=f"Upload failed on the server: {type(exc).__name__}: {exc}"), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    if STATE.status == "running":
        return jsonify(ok=False, error="A run is already in progress.")
    data = request.get_json(force=True) or {}
    try:
        output_dir = os.path.abspath(data.get("output_dir") or "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        cache_dir = os.path.join(output_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        config = PipelineConfig(
            data_dir=os.path.abspath(data.get("data_dir") or "./data"),
            magnifications=data.get("magnifications") or ["x5", "x20"],
            backbone=data.get("backbone", "resnet50"),
            knn_neighbors=int(data.get("knn_neighbors", 5)),
            test_size=float(data.get("test_size", 0.2)),
            cache_dir=cache_dir,
            output_dir=output_dir,
            use_dummy_extractor=bool(data.get("use_dummy", False)),
            cv_folds=int(data.get("cv_folds", 0) or 0),
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e))

    if not os.path.isdir(config.data_dir):
        return jsonify(ok=False, error=f"Data folder does not exist: {config.data_dir}")

    # If no magnifications given or invalid, auto-detect from subfolders
    subs = [d for d in os.listdir(config.data_dir)
            if os.path.isdir(os.path.join(config.data_dir, d))]
    if not config.magnifications or not any(m in subs for m in config.magnifications):
        if subs:
            config.magnifications = sorted(subs)

    STATE.reset()
    t = threading.Thread(target=_run_pipeline, args=(config,), daemon=True)
    STATE.thread = t
    t.start()
    return jsonify(ok=True)


@app.route("/api/logs")
def api_logs():
    def stream():
        for line in list(STATE.log_history):
            yield f"data: {line}\n\n"
        while True:
            try:
                line = STATE.log_queue.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"
                if STATE.status != "running":
                    yield "data: __END__\n\n"
                    return
                continue
            yield f"data: {line}\n\n"
            if line == "__END__":
                return
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/status")
def api_status():
    return jsonify(
        status=STATE.status,
        error=STATE.error,
        result=STATE.result,
        report_md=STATE.report_md,
        log_history=STATE.log_history[-800:],
    )


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    if STATE.status != "running":
        return jsonify(ok=False, error=f"No run to cancel (status: {STATE.status}).")
    STATE.cancel_flag.set()
    STATE.push_log("[web] Cancel requested — waiting for pipeline to stop…")
    return jsonify(ok=True)



@app.route("/api/resolve-output", methods=["POST"])
def api_resolve_output():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify(ok=False, error="No folder name provided.")
    safe = secure_filename(name) or "outputs"
    # Look for an existing folder with that name near the app; otherwise create it under HERE.
    candidates = [
        os.path.join(os.getcwd(), name),
        os.path.join(HERE, name),
        os.path.join(os.path.expanduser("~"), name),
        os.path.join(os.path.expanduser("~/Desktop"), name),
        os.path.join(os.path.expanduser("~/Downloads"), name),
        os.path.join(os.path.expanduser("~/Documents"), name),
    ]
    chosen = next((p for p in candidates if os.path.isdir(p)), None)
    if not chosen:
        chosen = os.path.join(HERE, safe)
        os.makedirs(chosen, exist_ok=True)
    return jsonify(ok=True, path=os.path.abspath(chosen))


def _safe_output_dir(path_arg: str) -> str:
    """Resolve and validate an output-dir path passed via querystring."""
    if not path_arg:
        abort(400, "missing path")
    p = os.path.abspath(path_arg)
    if not os.path.isdir(p):
        abort(404, f"not a directory: {p}")
    return p


@app.route("/api/export/zip")
def api_export_zip():
    out = _safe_output_dir(request.args.get("path", ""))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out):
            # skip cache directory
            dirs[:] = [d for d in dirs if d != ".cache"]
            for fn in files:
                fp = os.path.join(root, fn)
                arc = os.path.relpath(fp, out)
                try:
                    zf.write(fp, arc)
                except Exception:
                    pass
    buf.seek(0)
    name = f"{os.path.basename(out) or 'results'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=name)


@app.route("/api/export/metrics.csv")
def api_export_metrics_csv():
    out = _safe_output_dir(request.args.get("path", ""))
    result = STATE.result or {}
    metrics = (result.get("metrics") or {}) if result else {}
    if not metrics:
        # try to load from disk
        jp = os.path.join(out, "results.json")
        if os.path.isfile(jp):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    metrics = (json.load(f) or {}).get("metrics", {})
            except Exception:
                metrics = {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["magnification", "accuracy", "f1_score", "n_train_samples", "n_test_samples", "backbone"])
    for mag, m in metrics.items():
        w.writerow([
            mag,
            m.get("accuracy", ""),
            m.get("f1_score", ""),
            m.get("n_train_samples", ""),
            m.get("n_test_samples", ""),
            m.get("backbone", ""),
        ])
    data = buf.getvalue().encode("utf-8")
    return send_file(io.BytesIO(data), mimetype="text/csv",
                     as_attachment=True, download_name="metrics.csv")


@app.route("/api/export/results.json")
def api_export_results_json():
    out = _safe_output_dir(request.args.get("path", ""))
    fp = os.path.join(out, "results.json")
    if os.path.isfile(fp):
        return send_file(fp, mimetype="application/json",
                         as_attachment=True, download_name="results.json")
    data = json.dumps(STATE.result or {}, indent=2).encode("utf-8")
    return send_file(io.BytesIO(data), mimetype="application/json",
                     as_attachment=True, download_name="results.json")


@app.route("/api/export/report.md")
def api_export_report_md():
    out = _safe_output_dir(request.args.get("path", ""))
    fp = os.path.join(out, "final_report.md")
    if not os.path.isfile(fp):
        abort(404, "report not found")
    return send_file(fp, mimetype="text/markdown",
                     as_attachment=True, download_name="final_report.md")




@app.route("/api/reset", methods=["POST"])
def api_reset():
    if STATE.status == "running":
        return jsonify(ok=False, error="Cannot reset while running.")
    if STATE.tmp_upload_dir and os.path.isdir(STATE.tmp_upload_dir):
        try:
            shutil.rmtree(STATE.tmp_upload_dir, ignore_errors=True)
        except Exception:
            pass
        STATE.tmp_upload_dir = None
    STATE.reset()
    return jsonify(ok=True)


def main():
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    print(f"\n>> Cell Culture Classifier — Web UI at {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
