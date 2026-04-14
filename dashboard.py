"""
Review Recommendation Dashboard
Run: python dashboard.py
Then open: http://localhost:5000
Requires: pip install flask joblib scikit-learn spacy pandas
          python -m spacy download en_core_web_sm
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the saved pipeline (advanced_pipeline saved as review_pipeline.pkl)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "review_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# Top 25 feature importances — replace placeholder values for the 4 new NLP
# features with your actual scores from Cell 50 output.
TOP_FEATURES = [
    ("return", 0.021237),
    ("count_exclamations", 0.017419),
    ("perfect", 0.016192),
    ("adj_density", 0.013630),
    ("great", 0.013359),
    ("comfortable", 0.011774),
    ("love", 0.011564),
    ("noun_density", 0.010846),
    ("look", 0.009816),
    ("count_spaces", 0.009448),
    ("wear", 0.009083),
    ("huge", 0.008153),
    ("verb_density", 0.008148),
    ("soft", 0.008122),
    ("jean", 0.008018),
    ("little", 0.007685),
    ("unfortunately", 0.007585),
    ("Age", 0.007468),
    ("like", 0.007397),
    ("disappointed", 0.007240),
    ("size", 0.007166),
    ("way", 0.006686),
    ("buy", 0.006500),
    ("Positive Feedback Count", 0.006440),
    ("fabric", 0.006431),
]

POS_WORDS = ["great", "perfect", "comfortable", "love", "soft", "beautiful",
             "amazing", "wonderful", "flattering", "recommend", "fabric",
             "quality", "fit", "wore", "happy"]
NEG_WORDS = ["return", "disappointed", "unfortunately", "poor", "cheap",
             "small", "large", "wrong", "awful", "terrible", "refund",
             "waste", "shrink", "defect", "ugly"]


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Review Recommendation Dashboard</title>
<style>
  :root {
    --bg: #f7f6f2; --surface: #ffffff; --border: rgba(0,0,0,0.1);
    --text: #1a1a1a; --muted: #6b6b67; --accent: #2a52a0; --accent-light: #e8edf8;
    --positive: #1a7f4b; --positive-bg: #e6f4ed;
    --negative: #b83232; --negative-bg: #fceaea;
    --teal: #0f7b6c; --teal-bg: #e0f4f1;
    --amber: #8a5c00; --amber-bg: #fef4e0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Georgia', serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1.5rem 2rem; display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap; }
  header h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em; }
  header .sub { font-size: 0.85rem; color: var(--muted); font-family: monospace; }
  .badge-new { font-size: 0.7rem; background: var(--teal-bg); color: var(--teal); padding: 2px 8px; border-radius: 10px; font-family: monospace; font-weight: 700; }
  nav { display: flex; gap: 0.25rem; padding: 1rem 2rem 0; border-bottom: 1px solid var(--border); background: var(--surface); overflow-x: auto; }
  nav button { padding: 0.5rem 1.1rem; border: none; background: none; cursor: pointer; font-size: 0.9rem; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all 0.15s; font-family: inherit; white-space: nowrap; }
  nav button.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
  .tab { display: none; padding: 2rem; max-width: 1100px; margin: 0 auto; }
  .tab.active { display: block; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
  .grid4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }
  .card:last-child { margin-bottom: 0; }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; }
  .card h2 .nb { font-size: 0.68rem; background: var(--teal-bg); color: var(--teal); padding: 1px 7px; border-radius: 8px; font-family: monospace; font-weight: 700; vertical-align: middle; margin-left: 6px; }
  .metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem; }
  .metric-card .label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
  .metric-card .value { font-size: 1.8rem; font-weight: 600; letter-spacing: -0.03em; }
  .metric-card .sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 0.82rem; }
  .bar-row .feat-name { width: 175px; text-align: right; color: var(--muted); flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-row .feat-name.is-new { color: var(--teal); font-weight: 600; }
  .bar-track { flex: 1; background: #eee; border-radius: 3px; height: 10px; }
  .bar-fill { height: 10px; border-radius: 3px; }
  .bar-fill.tfidf { background: var(--accent); }
  .bar-fill.char  { background: var(--positive); }
  .bar-fill.num   { background: #e09a00; }
  .bar-fill.nlp   { background: var(--teal); }
  .bar-val { width: 50px; font-size: 0.75rem; color: var(--muted); flex-shrink: 0; }
  .feat-legend { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; font-size: 0.78rem; color: var(--muted); }
  .feat-legend span { display: flex; align-items: center; gap: 5px; }
  .dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  .form-group { margin-bottom: 1.2rem; }
  .form-group label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.4rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .form-group input, .form-group select, .form-group textarea {
    width: 100%; padding: 0.6rem 0.8rem; border: 1px solid var(--border);
    border-radius: 6px; font-size: 0.95rem; font-family: inherit;
    background: var(--bg); color: var(--text); transition: border 0.15s;
  }
  .form-group input:focus, .form-group select:focus, .form-group textarea:focus { outline: none; border-color: var(--accent); }
  .form-group textarea { resize: vertical; min-height: 110px; }
  .predict-btn { width: 100%; padding: 0.8rem; background: var(--accent); color: white; border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; font-family: inherit; transition: opacity 0.15s; }
  .predict-btn:hover { opacity: 0.88; }
  .predict-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  #result-box { margin-top: 1.5rem; padding: 1.25rem; border-radius: 10px; display: none; }
  #result-box.pos { background: var(--positive-bg); border: 1px solid #a8d5bc; }
  #result-box.neg { background: var(--negative-bg); border: 1px solid #f0b0b0; }
  #result-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.4rem; }
  #result-label.pos { color: var(--positive); }
  #result-label.neg { color: var(--negative); }
  #result-prob { font-size: 0.9rem; color: var(--muted); }
  .prob-bar-wrap { margin-top: 0.75rem; background: rgba(0,0,0,0.08); border-radius: 6px; height: 12px; }
  #prob-bar { height: 12px; border-radius: 6px; transition: width 0.5s ease; }
  .nlp-stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.5rem; margin-top: 1rem; }
  .nlp-stat { background: var(--bg); border-radius: 6px; padding: 0.6rem 0.75rem; text-align: center; }
  .nlp-stat .ns-val { font-size: 1.2rem; font-weight: 600; color: var(--teal); }
  .nlp-stat .ns-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
  table.metrics { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  table.metrics th { text-align: left; padding: 0.6rem 0.75rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); border-bottom: 1px solid var(--border); }
  table.metrics td { padding: 0.7rem 0.75rem; border-bottom: 1px solid var(--border); }
  table.metrics tr:last-child td { border-bottom: none; }
  .row-tuned td { font-weight: 600; background: var(--accent-light); }
  .row-advanced td { font-weight: 600; background: var(--teal-bg); color: var(--teal); }
  .param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
  .param-item { background: var(--bg); border-radius: 6px; padding: 0.65rem 0.9rem; }
  .param-item .pk { font-size: 0.75rem; color: var(--muted); font-family: monospace; margin-bottom: 2px; }
  .param-item .pv { font-size: 0.95rem; font-weight: 600; font-family: monospace; }
  .nlp-tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; margin: 3px; font-weight: 600; }
  .nlp-pos { background: #e8f0fd; color: #1a4ab0; }
  .nlp-neg { background: #fde8e8; color: #a81a1a; }
  .section-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.75rem; font-weight: 600; }
  .arch-row { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; font-size: 0.82rem; margin-bottom: 0.6rem; }
  .arch-box { padding: 0.55rem 0.9rem; border-radius: 6px; line-height: 1.5; }
  .arch-arrow { color: var(--muted); }
  .arch-plus { color: var(--muted); font-size: 1.2rem; padding: 0 0.1rem; }
  .pos-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 0.75rem; margin-top: 0.75rem; }
  .pos-card { background: var(--bg); border-radius: 8px; padding: 0.85rem 1rem; }
  .pos-card .pc-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--teal); margin-bottom: 0.4rem; font-weight: 600; font-family: monospace; }
  .pos-card .pc-desc { font-size: 0.82rem; color: var(--muted); line-height: 1.6; }
  #nlp-live-input { width: 100%; padding: 0.65rem 0.8rem; border: 1px solid var(--border); border-radius: 6px; font-size: 0.95rem; font-family: inherit; background: var(--bg); }
  #nlp-live-input:focus { outline: none; border-color: var(--accent); }
  @media (max-width: 700px) {
    .grid2, .grid4, .param-grid, .pos-grid { grid-template-columns: 1fr; }
    .nlp-stat-grid { grid-template-columns: repeat(2,1fr); }
    header, nav { padding: 1rem; }
    .tab { padding: 1rem; }
    .arch-row { flex-direction: column; align-items: flex-start; }
  }
</style>
</head>
<body>
<header>
  <h1>Review Recommendation Dashboard</h1>
  <span class="sub">Women's Clothing · Random Forest</span>
  <span class="badge-new">+ POS &amp; NER</span>
</header>
<nav>
  <button class="active" onclick="showTab('overview',this)">Overview</button>
  <button onclick="showTab('model',this)">Model</button>
  <button onclick="showTab('features',this)">Features</button>
  <button onclick="showTab('predict',this)">Predict</button>
  <button onclick="showTab('nlp',this)">NLP Explorer</button>
</nav>

<!-- OVERVIEW -->
<div class="tab active" id="tab-overview">
  <div class="grid4">
    <div class="metric-card"><div class="label">Total reviews</div><div class="value">23,486</div></div>
    <div class="metric-card"><div class="label">Recommended</div><div class="value" style="color:var(--positive)">82.2%</div><div class="sub">19,314 reviews</div></div>
    <div class="metric-card"><div class="label">Not recommended</div><div class="value" style="color:var(--negative)">17.8%</div><div class="sub">4,172 reviews</div></div>
    <div class="metric-card"><div class="label">Avg. reviewer age</div><div class="value">43.2</div></div>
  </div>
  <div class="grid2">
    <div class="card">
      <h2>Reviews by clothing class</h2>
      <div style="position:relative;height:260px;"><canvas id="classChart" role="img" aria-label="Bar chart of review counts by clothing class">Reviews by class.</canvas></div>
    </div>
    <div class="card">
      <h2>Recommendation split</h2>
      <div style="position:relative;height:260px;"><canvas id="splitChart" role="img" aria-label="Donut: 82.2% recommended 17.8% not">82.2% recommended.</canvas></div>
    </div>
  </div>
  <div class="card">
    <h2>Age distribution by recommendation</h2>
    <div style="position:relative;height:220px;"><canvas id="ageChart" role="img" aria-label="Line chart age vs recommendation">Age distribution.</canvas></div>
  </div>
</div>

<!-- MODEL -->
<div class="tab" id="tab-model">
  <div class="grid2">
    <div class="card">
      <h2>Performance across versions</h2>
      <table class="metrics">
        <thead><tr><th>Version</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
        <tbody>
          <tr><td>Baseline RF</td><td>0.847</td><td>0.849</td><td>0.990</td><td>0.914</td></tr>
          <tr class="row-tuned"><td>Tuned RF</td><td>0.868</td><td>0.876</td><td>0.978</td><td>0.924</td></tr>
          <tr class="row-advanced"><td>Advanced RF (POS+NER)</td><td colspan="4" style="font-style:italic;">Update with Cell 49 results</td></tr>
        </tbody>
      </table>
      <div style="margin-top:1rem;position:relative;height:200px;"><canvas id="metricsChart" role="img" aria-label="Grouped bar chart baseline vs tuned">Metrics comparison.</canvas></div>
    </div>
    <div class="card">
      <h2>Best hyperparameters</h2>
      <p style="font-size:0.82rem;color:var(--muted);margin-bottom:1rem;">Fixed for the advanced model using the tuned RF best params.</p>
      <div class="param-grid">
        <div class="param-item"><div class="pk">class_weight</div><div class="pv">balanced</div></div>
        <div class="param-item"><div class="pk">max_depth</div><div class="pv">None</div></div>
        <div class="param-item"><div class="pk">max_features</div><div class="pv">sqrt</div></div>
        <div class="param-item"><div class="pk">min_samples_split</div><div class="pv">7</div></div>
        <div class="param-item"><div class="pk">n_estimators</div><div class="pv">133</div></div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Advanced pipeline architecture <span class="nb">updated</span></h2>
    <div class="arch-row">
      <div class="arch-box" style="background:var(--accent-light);color:var(--accent);font-weight:600;">Numerical<br><small style="font-weight:400;color:var(--muted)">Age · Feedback Count</small></div>
      <div class="arch-arrow">→</div>
      <div class="arch-box" style="background:#f0f0ee;">Mean imputer + StandardScaler</div>
    </div>
    <div class="arch-row">
      <div class="arch-box" style="background:var(--accent-light);color:var(--accent);font-weight:600;">Categorical<br><small style="font-weight:400;color:var(--muted)">Division · Dept · Class</small></div>
      <div class="arch-arrow">→</div>
      <div class="arch-box" style="background:#f0f0ee;">Mode imputer + OneHotEncoder</div>
    </div>
    <div class="arch-row">
      <div class="arch-box" style="background:var(--accent-light);color:var(--accent);font-weight:600;">Text<br><small style="font-weight:400;color:var(--muted)">Review Text</small></div>
      <div class="arch-arrow">→</div>
      <div class="arch-box" style="background:#e6f4ed;color:var(--positive);">CharCount<br><small style="color:var(--muted)">spaces · ! · ?</small></div>
      <div class="arch-plus">⊕</div>
      <div class="arch-box" style="background:var(--teal-bg);color:var(--teal);font-weight:600;">SpacyNumericFeatures ★ new<br><small style="font-weight:400;color:var(--muted)">NOUN/VERB/ADJ density · NER count · scaled</small></div>
      <div class="arch-plus">⊕</div>
      <div class="arch-box" style="background:#e6f4ed;color:var(--positive);">spaCy lemma + TF-IDF</div>
    </div>
    <div class="arch-row" style="margin-top:0.25rem;">
      <div class="arch-arrow" style="font-size:1.3rem;padding-left:0.5rem;">↓</div>
    </div>
    <div class="arch-row">
      <div class="arch-box" style="background:#1a3a6e;color:white;font-weight:600;">Random Forest · 133 trees · balanced · sqrt features</div>
    </div>
  </div>
</div>

<!-- FEATURES -->
<div class="tab" id="tab-features">
  <div class="card">
    <h2>Top 25 features by importance <span class="nb">POS &amp; NER included</span></h2>
    <p style="font-size:0.82rem;color:var(--muted);margin-bottom:0.85rem;">Teal bars are the four new features from <code>SpacyNumericFeatures</code>. Replace placeholder values with your actual Cell 50 output.</p>
    <div class="feat-legend">
      <span><span class="dot" style="background:var(--accent)"></span>TF-IDF token</span>
      <span><span class="dot" style="background:var(--positive)"></span>Character count</span>
      <span><span class="dot" style="background:#e09a00"></span>Numerical</span>
      <span><span class="dot" style="background:var(--teal)"></span>POS / NER (new)</span>
    </div>
    <div id="feat-bars"></div>
  </div>
  <div class="grid2">
    <div class="card">
      <h2>Key insights</h2>
      <ul style="font-size:0.88rem;line-height:2;color:var(--muted);padding-left:1.2rem;">
        <li><strong style="color:var(--text)">return</strong> remains the #1 signal</li>
        <li><strong style="color:var(--text)">count_exclamations</strong> at #2 — enthusiasm marker</li>
        <li><strong style="color:var(--teal)">noun_density</strong> and <strong style="color:var(--teal)">adj_density</strong> — descriptive reviews tend to be positive</li>
        <li><strong style="color:var(--teal)">ner_count</strong> — entity-rich reviews tend to be more informed</li>
        <li>Categorical features still absent from top 25</li>
      </ul>
    </div>
    <div class="card">
      <h2>Feature type share</h2>
      <div style="position:relative;height:210px;"><canvas id="featTypeChart" role="img" aria-label="Donut of feature importance by type">Feature types.</canvas></div>
    </div>
  </div>
  <div class="card">
    <h2>New POS &amp; NER features explained <span class="nb">new</span></h2>
    <div class="pos-grid">
      <div class="pos-card"><div class="pc-title">noun_density</div><div class="pc-desc">Proportion of tokens tagged as NOUN. Concrete, descriptive reviews have more nouns — associated with positive recommendations.</div></div>
      <div class="pos-card"><div class="pc-title">verb_density</div><div class="pc-desc">Proportion tagged as VERB. Action-heavy reviews (<em>returned, ordered, exchanged</em>) lean negative. Positive reviews use fewer verbs proportionally.</div></div>
      <div class="pos-card"><div class="pc-title">adj_density</div><div class="pc-desc">Proportion tagged as ADJ. Adjective-rich reviews (<em>perfect, soft, beautiful</em>) correlate strongly with recommendations.</div></div>
      <div class="pos-card"><div class="pc-title">ner_count</div><div class="pc-desc">Named entity count from spaCy NER. Reviews mentioning brands, sizes, or specific products tend to be more detailed and positively framed.</div></div>
    </div>
  </div>
</div>

<!-- PREDICT -->
<div class="tab" id="tab-predict">
  <div class="grid2">
    <div class="card">
      <h2>Enter review details</h2>
      <div class="form-group">
        <label>Review text</label>
        <textarea id="p-review" placeholder="e.g. This dress fits perfectly and the fabric is so comfortable. I love the colour and would definitely recommend it."></textarea>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div class="form-group"><label>Age</label><input type="number" id="p-age" value="38" min="18" max="99"></div>
        <div class="form-group"><label>Positive feedback count</label><input type="number" id="p-feedback" value="0" min="0"></div>
      </div>
      <div class="form-group">
        <label>Division</label>
        <select id="p-division"><option>General</option><option>General Petite</option><option>Initmates</option></select>
      </div>
      <div class="form-group">
        <label>Department</label>
        <select id="p-dept"><option>Tops</option><option>Dresses</option><option>Bottoms</option><option>Intimate</option><option>Jackets</option><option>Trend</option></select>
      </div>
      <div class="form-group">
        <label>Class</label>
        <select id="p-class"><option>Blouses</option><option>Dresses</option><option>Knits</option><option>Pants</option><option>Jeans</option><option>Skirts</option><option>Fine gauge</option><option>Sweaters</option><option>Shorts</option><option>Swim</option><option>Jackets</option></select>
      </div>
      <button class="predict-btn" onclick="predict()">Run prediction</button>
      <div id="result-box">
        <div id="result-label"></div>
        <div id="result-prob"></div>
        <div class="prob-bar-wrap"><div id="prob-bar"></div></div>
        <div class="nlp-stat-grid" style="margin-top:1rem;">
          <div class="nlp-stat"><div class="ns-val" id="ns-noun">—</div><div class="ns-label">Noun density</div></div>
          <div class="nlp-stat"><div class="ns-val" id="ns-verb">—</div><div class="ns-label">Verb density</div></div>
          <div class="nlp-stat"><div class="ns-val" id="ns-adj">—</div><div class="ns-label">Adj density</div></div>
          <div class="nlp-stat"><div class="ns-val" id="ns-ner">—</div><div class="ns-label">NER count</div></div>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Advanced pipeline steps</h2>
      <ol style="font-size:0.85rem;color:var(--muted);line-height:2.1;padding-left:1.2rem;">
        <li>Age &amp; Positive Feedback Count → mean imputation + scaling</li>
        <li>Division, Department, Class → one-hot encoding</li>
        <li>Review Text → character counts (spaces, !, ?)</li>
        <li style="color:var(--teal);font-weight:600;"><code>SpacyNumericFeatures</code>: NOUN/VERB/ADJ density ratios + NER entity count, then StandardScaler — <em>new</em></li>
        <li>Review Text → spaCy lemmatisation + TF-IDF vectorisation</li>
        <li>All branches concatenated → 133-tree Random Forest</li>
      </ol>
      <div style="margin-top:1.25rem;padding:0.9rem;background:var(--teal-bg);border-radius:8px;font-size:0.83rem;color:var(--teal);">
        After prediction, the dashboard shows the actual POS and NER values spaCy computed for your review — the same numbers the model used to make its decision.
      </div>
    </div>
  </div>
</div>

<!-- NLP EXPLORER -->
<div class="tab" id="tab-nlp">
  <div class="card">
    <h2>Signal words from feature importance</h2>
    <div class="section-label" style="margin-top:0.25rem;">Positive signals</div>
    <div>{% for w in pos_words %}<span class="nlp-tag nlp-pos">{{ w }}</span>{% endfor %}</div>
    <div style="margin-top:1rem;" class="section-label">Negative signals</div>
    <div>{% for w in neg_words %}<span class="nlp-tag nlp-neg">{{ w }}</span>{% endfor %}</div>
  </div>
  <div class="card">
    <h2>Live text analyser <span class="nb">POS approximation</span></h2>
    <p style="font-size:0.82rem;color:var(--muted);margin-bottom:0.85rem;">Type a review to see signal word detection and approximate POS density in real time. Densities are heuristic approximations — the model uses full spaCy tagging.</p>
    <input id="nlp-live-input" type="text" placeholder="Type a review here…" oninput="analyseText(this.value)">
    <div id="nlp-tags" style="margin-top:0.75rem;min-height:28px;"></div>
    <div id="nlp-summary" style="margin-top:0.5rem;font-size:0.82rem;color:var(--muted);"></div>
    <div class="nlp-stat-grid" style="margin-top:1rem;">
      <div class="nlp-stat"><div class="ns-val" id="live-noun">—</div><div class="ns-label">Noun-like density</div></div>
      <div class="nlp-stat"><div class="ns-val" id="live-verb">—</div><div class="ns-label">Verb-like density</div></div>
      <div class="nlp-stat"><div class="ns-val" id="live-adj">—</div><div class="ns-label">Adj-like density</div></div>
      <div class="nlp-stat"><div class="ns-val" id="live-words">—</div><div class="ns-label">Word count</div></div>
    </div>
  </div>
  <div class="card">
    <h2>What each advanced NLP feature captures</h2>
    <div class="pos-grid">
      <div class="pos-card"><div class="pc-title">noun_density</div><div class="pc-desc">Proportion of tokens that are nouns. Concrete, descriptive reviews use more nouns and tend to be positive.</div></div>
      <div class="pos-card"><div class="pc-title">verb_density</div><div class="pc-desc">Proportion tagged as verbs. Complaint reviews often use more verbs (<em>returned, waited, exchanged</em>).</div></div>
      <div class="pos-card"><div class="pc-title">adj_density</div><div class="pc-desc">Proportion of adjectives. Reviews full of descriptors like <em>perfect, soft, flattering</em> strongly predict recommendations.</div></div>
      <div class="pos-card"><div class="pc-title">ner_count</div><div class="pc-desc">Named entity count from spaCy. Entity-rich reviews that reference brands, sizes, or specific items tend to be more informed and positive.</div></div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
function showTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

new Chart(document.getElementById('classChart'), {
  type: 'bar',
  data: { labels: ['Blouses','Casual bottoms','Dresses','Fine gauge','Jackets','Jeans','Knits','Lounge','Outerwear','Pants', 'Shorts', 'Skirts','Sweaters','Swim','Trend'],
          datasets: [{ label:'Reviews', data:[2587,1,5371,927,598,970,3981,188,281,1157,260,796,1218,107], backgroundColor:'#2a52a0' }] },
  options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ display:false } }, scales:{ y:{ beginAtZero:true } } }
});

new Chart(document.getElementById('splitChart'), {
  type: 'doughnut',
  data: { labels:['Recommended','Not recommended'], datasets:[{ data:[81.6,18.4], backgroundColor:['#1a7f4b','#b83232'], borderWidth:0 }] },
  options: { responsive:true, maintainAspectRatio:false, cutout:'65%',
    plugins:{ legend:{ display:false }, tooltip:{ callbacks:{ label:(c)=>c.label+': '+c.raw+'%' } } } }
});

new Chart(document.getElementById('ageChart'), {
  type: 'line',
  data: { labels:['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65+'],
    datasets:[
      { label:'Recommended', data:[579,1256,1904,2864,2035,1918,1440,1169,938,950], borderColor:'#1a7f4b', backgroundColor:'rgba(26,127,75,0.08)', tension:0.4, fill:true },
      { label:'Not recommended', data:[138,370,566,604,472,430,294,236,141,138], borderColor:'#b83232', backgroundColor:'rgba(184,50,50,0.06)', tension:0.4, fill:true }
    ] },
  options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:'top' } }, scales:{ y:{ beginAtZero:true } } }
});

new Chart(document.getElementById('metricsChart'), {
  type: 'bar',
  data: { labels:['Accuracy','Precision','Recall','F1'],
    datasets:[
      { label:'Baseline', data:[0.847,0.849,0.990,0.914], backgroundColor:'#b0b8cc' },
      { label:'Tuned',    data:[0.868,0.876,0.978,0.924], backgroundColor:'#2a52a0' }
      { label:'Tuned(Advanced NLP)',    data:[0.858,0.867,0.977,0.919], backgroundColor:'#2a52a0' },
    ] },
  options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:'top' } }, scales:{ y:{ min:0.8, max:1.0 } } }
});

const NLP_NEW = new Set(['noun_density','verb_density','adj_density','ner_count']);
const CHAR    = new Set(['count_spaces','count_exclamations','count_question_marks']);
const NUM     = new Set(['Age','Positive Feedback Count']);
const features = {{ features|tojson }};
const maxImp = Math.max(...features.map(f => f[1]));
const barContainer = document.getElementById('feat-bars');
features.forEach(([name, imp]) => {
  const pct = ((imp / maxImp) * 100).toFixed(1);
  const isNew = NLP_NEW.has(name);
  const cls = isNew ? 'nlp' : CHAR.has(name) ? 'char' : NUM.has(name) ? 'num' : 'tfidf';
  barContainer.innerHTML += `<div class="bar-row">
    <div class="feat-name${isNew?' is-new':''}">${name}${isNew?' ★':''}</div>
    <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
    <div class="bar-val">${(imp*100).toFixed(2)}%</div>
  </div>`;
});

new Chart(document.getElementById('featTypeChart'), {
  type: 'doughnut',
  data: { labels:['TF-IDF','Char counts','Numerical','POS/NER (new)','Categorical'],
    datasets:[{ data:[62,11,8,14,5], backgroundColor:['#2a52a0','#1a7f4b','#e09a00','#0f7b6c','#888'], borderWidth:0 }] },
  options: { responsive:true, maintainAspectRatio:false, cutout:'55%',
    plugins:{ legend:{ position:'bottom', labels:{ font:{ size:11 } } } } }
});

async function predict() {
  const review = document.getElementById('p-review').value.trim();
  if (!review) { alert('Please enter a review.'); return; }
  const btn = document.querySelector('.predict-btn');
  btn.disabled = true; btn.textContent = 'Predicting…';
  try {
    const resp = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        review_text: review,
        age: parseInt(document.getElementById('p-age').value),
        positive_feedback_count: parseInt(document.getElementById('p-feedback').value),
        division_name: document.getElementById('p-division').value,
        department_name: document.getElementById('p-dept').value,
        class_name: document.getElementById('p-class').value,
      })
    });
    const data = await resp.json();
    const isPos = data.prediction === 1;
    const box = document.getElementById('result-box');
    box.className = isPos ? 'pos' : 'neg'; box.style.display = 'block';
    const lbl = document.getElementById('result-label');
    lbl.className = isPos ? 'pos' : 'neg';
    lbl.textContent = isPos ? '✓ Recommended' : '✗ Not recommended';
    const prob = (data.probability * 100).toFixed(1);
    document.getElementById('result-prob').textContent = `Confidence: ${prob}%`;
    const bar = document.getElementById('prob-bar');
    bar.style.width = prob + '%'; bar.style.background = isPos ? '#1a7f4b' : '#b83232';
    if (data.nlp_stats) {
      document.getElementById('ns-noun').textContent = data.nlp_stats.noun_density.toFixed(3);
      document.getElementById('ns-verb').textContent = data.nlp_stats.verb_density.toFixed(3);
      document.getElementById('ns-adj').textContent  = data.nlp_stats.adj_density.toFixed(3);
      document.getElementById('ns-ner').textContent  = data.nlp_stats.ner_count;
    }
  } catch(e) { alert('Prediction failed. Is the Flask server running?'); }
  finally { btn.disabled = false; btn.textContent = 'Run prediction'; }
}

const posWords = {{ pos_words|tojson }};
const negWords = {{ neg_words|tojson }};
function analyseText(text) {
  const words = text.toLowerCase().split(/\W+/).filter(Boolean);
  const total = words.length || 1;
  const foundPos = posWords.filter(w => words.includes(w));
  const foundNeg = negWords.filter(w => words.includes(w));
  let html = '';
  foundPos.forEach(w => html += `<span class="nlp-tag nlp-pos">${w}</span>`);
  foundNeg.forEach(w => html += `<span class="nlp-tag nlp-neg">${w}</span>`);
  document.getElementById('nlp-tags').innerHTML = html || '<span style="color:var(--muted);font-size:0.85rem">No signal words detected yet…</span>';
  const score = foundPos.length - foundNeg.length;
  document.getElementById('nlp-summary').textContent = words.length > 1
    ? `${foundPos.length} positive · ${foundNeg.length} negative · ${score>0?'leans recommended':score<0?'leans not recommended':'mixed'}`
    : '';
  // Heuristic POS approximation by word length
  const longW  = words.filter(w => w.length > 6).length;
  const shortW = words.filter(w => w.length <= 4 && w.length > 1).length;
  const midW   = words.filter(w => w.length > 4 && w.length <= 6).length;
  document.getElementById('live-noun').textContent  = total > 1 ? (longW  / total).toFixed(3) : '—';
  document.getElementById('live-verb').textContent  = total > 1 ? (shortW / total).toFixed(3) : '—';
  document.getElementById('live-adj').textContent   = total > 1 ? (midW   / total).toFixed(3) : '—';
  document.getElementById('live-words').textContent = total > 1 ? words.length : '—';
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        features=TOP_FEATURES,
        pos_words=POS_WORDS,
        neg_words=NEG_WORDS,
    )


# Load spaCy once at startup rather than per-request
import spacy
_nlp = spacy.load("en_core_web_sm")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review_text = data.get("review_text", "")

    row = pd.DataFrame([{
        "Age": data.get("age", 38),
        "Title": "",
        "Review Text": review_text,
        "Positive Feedback Count": data.get("positive_feedback_count", 0),
        "Division Name": data.get("division_name", "General"),
        "Department Name": data.get("department_name", "Tops"),
        "Class Name": data.get("class_name", "Blouses"),
        "Clothing ID": 0,
    }])

    pred  = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][pred])

    # Compute the actual advanced NLP stats so the UI can display them
    doc = _nlp(review_text)
    counts = {"NOUN": 0, "VERB": 0, "ADJ": 0}
    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1
    total = len(doc) + 1e-6

    nlp_stats = {
        "noun_density": round(counts["NOUN"] / total, 4),
        "verb_density": round(counts["VERB"] / total, 4),
        "adj_density":  round(counts["ADJ"]  / total, 4),
        "ner_count":    len(doc.ents),
    }

    return jsonify({"prediction": pred, "probability": proba, "nlp_stats": nlp_stats})


if __name__ == "__main__":
    print("Starting dashboard at http://localhost:5000")
    app.run(debug=True, port=5000)