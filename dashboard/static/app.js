/* Live GPU workload classifier — frontend.
 *
 * Reads one state frame per second from /api/stream and draws four things:
 * the current call, how long detection took, how the classifier handled a
 * workload change, and how many false alarms have piled up.
 *
 * Everything is inline SVG built from the state frame; there is no chart
 * library and no build step, so this file is the whole of the rendering.
 */
"use strict";

const CLASS_COLOR = {
  ml_training:  "#f0883e",
  ml_inference: "#58a6ff",
  other:        "#6e7a86",
  rest:         "#6e7a86",
};
const CLASS_LABEL = {
  ml_training:  "ML TRAINING",
  ml_inference: "ML INFERENCE",
  other:        "OTHER",
  rest:         "NOT TRAINING",
};
const colorFor = (c) => CLASS_COLOR[c] || "#8b96a5";

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

let MODEL = null;         // model card, sent once when the stream opens
let LAST_STATE = null;
let baseRate = 0.05;
let lastTruthSource = null;

/* ── tiny SVG helpers ─────────────────────────────────────────────────── */

function el(name, attrs, text) {
  const n = document.createElementNS(SVG_NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : v.toFixed(d);
const pct = (v, d = 1) => (v === null || v === undefined) ? "—" : (100 * v).toFixed(d) + "%";

/* ── provenance banner ────────────────────────────────────────────────── */

function renderBanner(model) {
  const warns = model.provenance_warnings || [];
  const box = $("banner");
  const text = $("banner-text");
  const p = model.provenance || {};

  const worst = warns.some((w) => w.level === "error") ? "error"
              : warns.length ? "warn" : "ok";
  box.className = "banner banner-" + worst;

  clear(text);
  const trained = [
    p.gpu_name || "GPU not recorded",
    model.trained_power_w ? `~${model.trained_power_w.toFixed(0)} W power regime` : "power regime not recorded",
  ].join(" · ");

  const head = document.createElement("div");
  head.style.flex = "1";
  head.innerHTML = `<strong>Model trained on:</strong> ${trained}. ` +
    `<strong>Features:</strong> ${model.n_features} window-local.`;

  if (warns.length) {
    const ul = document.createElement("ul");
    for (const w of warns) {
      const li = document.createElement("li");
      li.textContent = w.message;
      ul.appendChild(li);
    }
    head.appendChild(ul);
  } else {
    head.innerHTML += " <em>Model matches this machine.</em>";
  }
  text.appendChild(head);
}

/* ── current verdict ──────────────────────────────────────────────────── */

function renderVerdict(s) {
  const latest = s.latest;
  const cls = $("verdict-class");
  const conf = $("verdict-conf");

  if (!latest) {
    cls.textContent = "—";
    cls.className = "verdict-class";
    conf.textContent = s.ready ? "waiting for the first classification"
                               : `filling the window — ${s.buffered_samples} / ${s.window_sec} samples`;
  } else {
    cls.textContent = CLASS_LABEL[latest.predicted] || latest.predicted;
    cls.className = "verdict-class " + latest.predicted;
    const full = latest.window_full;
    conf.textContent = `${pct(latest.confidence)} confidence` +
      (full ? "" : `  ·  window only ${latest.window_samples}/${s.window_sec} s full — partial evidence`);
  }

  const bars = $("prob-bars");
  clear(bars);
  const probs = latest ? latest.probs : Object.fromEntries(s.classes.map((c) => [c, 0]));
  for (const c of s.classes) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML =
      `<span class="prob-name">${c}</span>` +
      `<span class="prob-track"><span class="prob-fill" style="width:${(100 * (probs[c] || 0)).toFixed(1)}%;background:${colorFor(c)}"></span></span>` +
      `<span class="prob-val">${pct(probs[c] || 0, 1)}</span>`;
    bars.appendChild(row);
  }

  $("fillbar").style.width = (100 * s.window_fill_frac).toFixed(1) + "%";
  $("fill-label").textContent =
    `window ${Math.min(s.buffered_samples, s.window_sec)} / ${s.window_sec} s` +
    `  ·  ${s.buffered_samples} samples buffered` +
    `  ·  ${fmt(s.timing_ms.extract, 1)} ms features + ${fmt(s.timing_ms.predict, 1)} ms predict`;
}

/* ── panel 1: detection latency ───────────────────────────────────────── */

function renderLatency(s) {
  const ep = s.events.current;
  const svg = $("latency-svg");
  clear(svg);

  const val = $("ttc-value");
  const det = $("ttc-detail");

  if (!ep) {
    val.textContent = "—";
    val.className = "stat-value";
    det.textContent = s.events.gpu_active
      ? "workload running, but no start was observed — reset the books to time the next one"
      : "no workload running";
    svg.appendChild(el("text", { x: 320, y: 110, "text-anchor": "middle", class: "axis-text" },
      "waiting for a workload to start"));
    return;
  }

  // time_to_correct is the honest number when truth is known: confident AND
  // right. time_to_confident alone can be a confident wrong answer.
  const known = ep.true_class !== null && ep.true_class !== undefined;
  const t = known ? ep.time_to_correct : ep.time_to_confident;

  if (ep.at_monitor_start) {
    // Measuring from here would report the buffer filling as if it were the
    // model deciding, and it would report a flatteringly small number.
    val.textContent = "—";
    val.className = "stat-value";
    det.textContent =
      "This workload was already running when monitoring started, so there is " +
      "no before-and-after to measure. The next transition will be timed " +
      "properly.";
  } else if (t !== null && t !== undefined) {
    val.textContent = t.toFixed(1) + " s";
    val.className = "stat-value";
    const frac = Math.min(1, t / s.window_sec);
    det.textContent =
      `from ${ep.kind === "transition" ? "the declared switch" : "workload start"}` +
      ` to a confident${known ? " and correct" : ""} call, held ${3} s.` +
      (frac >= 0.9
        ? "  Most of that is the 30 s window flushing out the previous state, not the model hesitating."
        : "");
  } else {
    val.textContent = fmt(ep.age_s, 0) + " s";
    val.className = "stat-value pending";
    det.textContent = `counting — no confident${known ? " and correct" : ""} call yet` +
      (ep.window_fill < 1 ? `. Window is ${pct(ep.window_fill, 0)} filled with the new workload.` : ".");
  }

  /* chart */
  const W = 640, H = 220, L = 46, R = 626, T = 14, B = 178;
  const traj = ep.trajectory || [];
  // Show the convergence, not the plateau after it. Once a call has settled
  // there is nothing to learn from another five minutes of a flat line, and
  // stretching the axis to cover it squashes the part that matters.
  const settleT = (t !== null && t !== undefined) ? t : null;
  const maxT = settleT !== null
    ? Math.max(60, settleT * 2.5)
    : Math.max(60, Math.min(300, ep.age_s || 0));
  const shown = traj.filter((p) => p[0] <= maxT);
  const x = (sec) => L + (R - L) * (sec / maxT);
  const y = (p) => B - (B - T) * p;

  // The window-contamination band: until t = window_sec the classifier is
  // still reading samples from before this workload started.
  const bandEnd = x(Math.min(s.window_sec, maxT));
  svg.appendChild(el("rect", { x: L, y: T, width: bandEnd - L, height: B - T,
                               fill: "#58a6ff", "fill-opacity": ".055" }));
  svg.appendChild(el("line", { x1: bandEnd, y1: T, x2: bandEnd, y2: B,
                               stroke: "#58a6ff", "stroke-opacity": ".35",
                               "stroke-dasharray": "3 3" }));
  svg.appendChild(el("text", { x: L + 6, y: T + 12, class: "chart-label",
                               fill: "#58a6ff", "fill-opacity": ".75" },
                     `window still filling (0–${s.window_sec} s)`));

  for (let p = 0; p <= 1.0001; p += 0.25) {
    svg.appendChild(el("line", { x1: L, y1: y(p), x2: R, y2: y(p), class: "gridline" }));
    svg.appendChild(el("text", { x: L - 7, y: y(p) + 3.5, "text-anchor": "end", class: "axis-text" },
                       p.toFixed(2)));
  }
  const thr = s.events.confidence_threshold;
  svg.appendChild(el("line", { x1: L, y1: y(thr), x2: R, y2: y(thr),
                               stroke: "#d29922", "stroke-dasharray": "5 4", "stroke-width": 1.2 }));
  svg.appendChild(el("text", { x: R, y: y(thr) - 5, "text-anchor": "end", class: "chart-label",
                               fill: "#d29922" }, `confidence threshold ${thr}`));

  for (let sec = 0; sec <= maxT; sec += Math.max(10, Math.round(maxT / 6 / 10) * 10)) {
    svg.appendChild(el("text", { x: x(sec), y: B + 15, "text-anchor": "middle", class: "axis-text" },
                       sec + "s"));
  }
  svg.appendChild(el("line", { x1: L, y1: B, x2: R, y2: B, class: "axis" }));

  const classes = s.classes;
  for (const c of classes) {
    const pts = shown.map((p) => `${x(p[0]).toFixed(1)},${y(p[1][c] || 0).toFixed(1)}`).join(" ");
    if (!pts) continue;
    svg.appendChild(el("polyline", {
      points: pts, fill: "none", stroke: colorFor(c),
      "stroke-width": c === ep.true_class ? 2.2 : 1.4,
      "stroke-opacity": (!ep.true_class || c === ep.true_class) ? 1 : .55,
    }));
  }

  if (t !== null && t !== undefined && t <= maxT) {
    svg.appendChild(el("line", { x1: x(t), y1: T, x2: x(t), y2: B,
                                 stroke: "#3fb950", "stroke-width": 1.4 }));
    svg.appendChild(el("text", { x: x(t) + 5, y: T + 26, class: "chart-label", fill: "#3fb950" },
                       `confident at ${t.toFixed(1)}s`));
  }

  const sub = ep.kind === "transition"
    ? `${ep.from_class || "?"} → ${ep.true_class || "?"}`
    : (ep.true_label ? `start of ${ep.true_label}` : "workload start (undeclared)");
  svg.appendChild(el("text", { x: L, y: B + 32, class: "chart-label" },
                     `seconds since ${ep.kind === "transition" ? "declared switch" : "onset"} — ${sub}`));
}

/* ── panel 2: transitions over time ───────────────────────────────────── */

function renderTimeline(s) {
  const svg = $("timeline-svg");
  clear(svg);
  const hist = s.history || [];
  if (!hist.length) {
    svg.appendChild(el("text", { x: 500, y: 130, "text-anchor": "middle", class: "axis-text" },
      "no classifications yet"));
    return;
  }

  const W = 1000, L = 52, R = 990, T = 12, B = 186;
  const TRUTH_Y = 198, PRED_Y = 224, BAND_H = 18;

  const n = hist.length;
  const t0 = hist[0].t, t1 = hist[n - 1].t;
  const span = Math.max(1, t1 - t0);
  const x = (t) => L + (R - L) * ((t - t0) / span);
  const y = (p) => B - (B - T) * p;

  // Probabilities sum to 1, so a stacked area fills the plot and the boundary
  // between colours IS the decision — no separate "predicted" line needed.
  const classes = s.classes;
  const cum = hist.map(() => 0);
  for (const c of classes) {
    const top = [], bottom = [];
    hist.forEach((h, i) => {
      const lo = cum[i], hi = lo + (h.probs[c] || 0);
      cum[i] = hi;
      top.push(`${x(h.t).toFixed(1)},${y(hi).toFixed(1)}`);
      bottom.push(`${x(h.t).toFixed(1)},${y(lo).toFixed(1)}`);
    });
    svg.appendChild(el("polygon", {
      points: top.concat(bottom.reverse()).join(" "),
      fill: colorFor(c), "fill-opacity": ".82",
    }));
  }

  for (let p = 0.25; p <= 0.76; p += 0.25) {
    svg.appendChild(el("line", { x1: L, y1: y(p), x2: R, y2: y(p),
                                 stroke: "#0e1116", "stroke-opacity": ".35" }));
  }
  svg.appendChild(el("text", { x: L - 7, y: y(1) + 4, "text-anchor": "end", class: "axis-text" }, "1.0"));
  svg.appendChild(el("text", { x: L - 7, y: y(0) + 4, "text-anchor": "end", class: "axis-text" }, "0.0"));

  // False alarms as ticks along the top.
  for (const h of hist) {
    if (!h.false_alarm) continue;
    svg.appendChild(el("line", { x1: x(h.t), y1: T, x2: x(h.t), y2: T + 9,
                                 stroke: "#f85149", "stroke-width": 1.5 }));
  }

  // Truth and prediction bands: the horizontal offset between a colour change
  // in one and the other is the detection lag, readable by eye.
  const band = (yTop, key, label) => {
    let runStart = 0;
    for (let i = 1; i <= hist.length; i++) {
      const prev = hist[i - 1][key];
      if (i === hist.length || hist[i][key] !== prev) {
        const x0 = x(hist[runStart].t), x1v = x(hist[i - 1].t);
        svg.appendChild(el("rect", {
          x: x0, y: yTop, width: Math.max(1, x1v - x0), height: BAND_H,
          fill: prev ? colorFor(prev) : "#262d38",
          "fill-opacity": prev ? ".9" : ".5",
        }));
        runStart = i;
      }
    }
    svg.appendChild(el("text", { x: L - 7, y: yTop + 13, "text-anchor": "end", class: "axis-text" }, label));
  };
  band(TRUTH_Y, "declared", "declared");
  band(PRED_Y, "predicted", "predicted");

  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const t = t0 + (span * i) / ticks;
    const rel = Math.round(t - t1);
    svg.appendChild(el("text", { x: x(t), y: PRED_Y + BAND_H + 15, "text-anchor": "middle", class: "axis-text" },
                       rel === 0 ? "now" : rel + "s"));
  }

  const legend = $("timeline-legend");
  clear(legend);
  for (const c of classes) {
    const d = document.createElement("div");
    d.className = "legend-item";
    d.innerHTML = `<span class="swatch" style="background:${colorFor(c)}"></span>${c}`;
    legend.appendChild(d);
  }
  const fa = document.createElement("div");
  fa.className = "legend-item";
  fa.innerHTML = `<span class="swatch" style="background:#f85149"></span>false alarm`;
  legend.appendChild(fa);

  /* episode summaries */
  const list = $("episode-list");
  clear(list);
  const eps = s.events.recent_episodes || [];
  if (!eps.length) {
    const d = document.createElement("div");
    d.className = "fp-empty";
    d.textContent = "No workload starts or transitions observed yet.";
    list.appendChild(d);
  }
  for (const e of eps.slice(0, 5)) {
    const known = e.true_class !== null && e.true_class !== undefined;
    const t = e.at_monitor_start ? null : (known ? e.time_to_correct : e.time_to_confident);
    const d = document.createElement("div");
    d.className = "episode " + (e.at_monitor_start ? ""
      : t !== null && t !== undefined ? "settled" : "pending");
    const what = e.kind === "transition"
      ? `${e.from_class || "?"} → ${e.true_class || "?"}`
      : `start · ${e.true_class || "undeclared"}`;
    d.innerHTML =
      `<span class="k">${e.kind}</span>` +
      `<span>${what}</span>` +
      `<span class="k">detected in</span>` +
      `<span>${e.at_monitor_start ? "n/a — already running at startup"
        : t !== null && t !== undefined ? t.toFixed(1) + " s"
        : "not yet (" + fmt(e.age_s, 0) + " s)"}</span>` +
      `<span class="k">age ${fmt(e.age_s, 0)} s</span>`;
    list.appendChild(d);
  }
}

/* ── panel 3: false positives ─────────────────────────────────────────── */

function precisionAt(op, b) {
  const tp = b * op.recall, fp = (1 - b) * op.fpr;
  return tp + fp > 0 ? tp / (tp + fp) : 0;
}

function renderLedger(s) {
  const L = s.events.ledger;
  $("fp-count").textContent = L.false_alarms;
  $("fp-obs").textContent = L.windows_known_not_training;
  $("fp-miss").textContent = L.missed_detections;

  $("fp-rate").textContent = L.observed_fp_rate === null
    ? "declare the GPU's state to start counting"
    : `${pct(L.observed_fp_rate, 1)} of windows · ${L.false_alarms_per_min_5m}/min over the last 5 min`;
  $("fp-miss-rate").textContent = L.observed_miss_rate === null
    ? "while declared training"
    : `${pct(L.observed_miss_rate, 1)} of declared-training windows`;

  const log = $("fp-log");
  clear(log);
  if (!L.recent || !L.recent.length) {
    const d = document.createElement("div");
    d.className = "fp-empty";
    const how = (LAST_STATE && LAST_STATE.truth_source === "operator")
      ? "Declare what the GPU is really doing (top right)"
      : "Once ground truth is known";
    d.textContent = L.windows_known_not_training
      ? `No false alarms yet across ${L.windows_known_not_training} windows of declared non-training work.`
      : `${how} and every spurious ml_training window will be logged here as it arrives.`;
    log.appendChild(d);
  }
  for (const a of (L.recent || [])) {
    const d = document.createElement("div");
    d.className = "fp-line";
    d.innerHTML = `<span class="t">-${fmt(a.age_s, 0)}s</span>` +
      `<span>ml_training @ ${pct(a.confidence, 1)}</span>` +
      `<span class="t">actually ${a.true_label || a.true_class}</span>`;
    log.appendChild(d);
  }

  renderPrecision();
}

function renderPrecision() {
  const svg = $("precision-svg");
  clear(svg);
  const op = MODEL && MODEL.operating_point;
  const note = $("precision-note");

  if (!op) {
    $("precision-value").textContent = "—";
    $("precision-detail").textContent = "";
    note.textContent = "This bundle records no held-out confusion matrix, so precision cannot be projected.";
    return;
  }

  const p = precisionAt(op, baseRate);
  $("precision-value").textContent = pct(p, 0);
  $("precision-detail").textContent =
    `Held-out recall ${pct(op.recall, 1)}, false-positive rate ${pct(op.fpr, 1)} ` +
    `(${op.folds}-fold, grouped by workload). At a ${pct(baseRate, 0)} base rate that means ` +
    `roughly ${(1 / Math.max(p, 1e-9)).toFixed(1)} flagged windows for every one that is really training.`;

  const W = 640, H = 200, L = 46, R = 620, T = 14, B = 164;
  const maxB = 0.8;
  const x = (b) => L + (R - L) * (b / maxB);
  const y = (v) => B - (B - T) * v;

  for (let v = 0; v <= 1.0001; v += 0.25) {
    svg.appendChild(el("line", { x1: L, y1: y(v), x2: R, y2: y(v), class: "gridline" }));
    svg.appendChild(el("text", { x: L - 7, y: y(v) + 3.5, "text-anchor": "end", class: "axis-text" },
                       (100 * v).toFixed(0) + "%"));
  }
  svg.appendChild(el("line", { x1: L, y1: B, x2: R, y2: B, class: "axis" }));
  for (const b of [0.05, 0.2, 0.4, 0.6, 0.8]) {
    svg.appendChild(el("text", { x: x(b), y: B + 15, "text-anchor": "middle", class: "axis-text" },
                       (100 * b).toFixed(0) + "%"));
  }
  svg.appendChild(el("text", { x: (L + R) / 2, y: B + 32, "text-anchor": "middle", class: "chart-label" },
                     "fraction of GPU-time that is really training  →"));

  const pts = [];
  for (let b = 0.005; b <= maxB; b += 0.005) pts.push(`${x(b).toFixed(1)},${y(precisionAt(op, b)).toFixed(1)}`);
  svg.appendChild(el("polyline", { points: pts.join(" "), fill: "none", stroke: "#f0883e", "stroke-width": 2 }));

  // The paper's corpus was 62% training. That single fact, not the model,
  // explains most of the gap between a headline accuracy and field precision.
  const anno = (b, label, color) => {
    svg.appendChild(el("line", { x1: x(b), y1: T, x2: x(b), y2: B,
                                 stroke: color, "stroke-dasharray": "3 3", "stroke-opacity": ".7" }));
    svg.appendChild(el("circle", { cx: x(b), cy: y(precisionAt(op, b)), r: 3.5, fill: color }));
    svg.appendChild(el("text", { x: x(b) + 6, y: y(precisionAt(op, b)) - 8, class: "chart-label", fill: color },
                       `${label}: ${pct(precisionAt(op, b), 0)}`));
  };
  anno(0.62, "paper's corpus (62%)", "#8b96a5");
  anno(baseRate, "you are here", "#f85149");

  note.textContent =
    "Same model, same recall, same false-positive rate — only the prior changes. " +
    "A classifier measured on a corpus that is 62% training looks very different " +
    "pointed at a fleet where training is a few percent of GPU-time.";
}

/* ── same window, two models ──────────────────────────────────────────── */

const SEEN = {
  run:     ["in-sample", "TRAINED ON THIS RUN"],
  config:  ["saw-config", "TRAINED ON THIS WORKLOAD"],
  none:    ["held-out", "NEVER SAW THIS WORKLOAD"],
  unknown: ["unknown-seen", "TRAINING DATA NOT RECORDED"],
};

function compareSide(node, title, meta, probs, predicted, seenLevel, scores, warns) {
  const [cls, badge] = SEEN[seenLevel] || SEEN.unknown;
  node.className = "compare-side " + cls;
  const bars = Object.keys(probs || {}).map((c) =>
    `<div class="cs-bar"><span>${c}</span>` +
    `<span class="track"><span class="fill" style="width:${(100 * probs[c]).toFixed(1)}%;background:${colorFor(c)}"></span></span>` +
    `<span>${pct(probs[c], 0)}</span></div>`).join("");
  const sw = scores && scores.grouped_by_workload;
  node.innerHTML =
    `<div class="cs-name">${title}` +
    `<span class="cs-badge ${cls}">${badge}</span></div>` +
    `<div class="cs-verdict ${predicted || ""}">${predicted ? (CLASS_LABEL[predicted] || predicted) : "—"}</div>` +
    `<div class="cs-meta">${meta}${sw ? ` · held-out accuracy ${pct(sw.accuracy, 1)}` : ""}</div>` +
    `<div class="cs-bars">${bars}</div>` +
    ((warns && warns.length)
      ? `<div class="cs-warn">${warns.map((w) => `<div>${w.message}</div>`).join("")}</div>`
      : "");
}

function renderCompare(s) {
  const card = $("compare-card");
  const cmp = MODEL && MODEL.compare;
  const latest = s.latest;
  if (!cmp || !latest || !latest.compare) { card.hidden = true; return; }
  card.hidden = false;

  const regime = (m) => m && m.trained_power_w
    ? ` · trained at ${m.trained_power_w.toFixed(0)} W` : "";
  compareSide($("compare-primary"),
    MODEL.corpus.holdout_configs && MODEL.corpus.holdout_configs.length
      ? "primary · held-out model" : "primary model",
    `${MODEL.corpus.n_runs} runs, ${MODEL.corpus.n_configs} configs${regime(MODEL)}`,
    latest.probs, latest.predicted, MODEL.seen_level, MODEL.scores,
    MODEL.provenance_warnings);

  compareSide($("compare-secondary"),
    cmp.holdout_configs && cmp.holdout_configs.length
      ? "comparison · held-out model" : "comparison model",
    `${cmp.corpus.n_runs} runs, ${cmp.corpus.n_configs} configs${regime(cmp)}`,
    latest.compare.probs, latest.compare.predicted, cmp.seen_level, cmp.scores,
    cmp.provenance_warnings);

  const ag = s.compare_agreement;
  const box = $("compare-agree");
  if (ag) {
    box.innerHTML =
      `The two models agree on <strong>${pct(ag.rate, 1)}</strong> of ${ag.n} windows so far. ` +
      `Mean confidence — primary <strong>${ag.mean_conf_primary}</strong>, ` +
      `comparison <strong>${ag.mean_conf_compare}</strong>.`;
  } else {
    box.textContent = "";
  }

  const rank = { run: 3, config: 2, none: 1, unknown: 0 };
  const gap = (rank[MODEL.seen_level] || 0) - (rank[cmp.seen_level] || 0);
  $("compare-note").textContent = gap !== 0
    ? "These two models stand in different relations to the workload on screen: one has trained on it, " +
      "the other has not. Where they diverge, the confident one is not the better one — it is the one " +
      "closer to its own training data. This is what the gap between grouped-by-run and " +
      "grouped-by-workload accuracy looks like second by second."
    : "Both models stand in the same relation to this workload, so a disagreement here reflects their " +
      "training corpora, not memorisation.";
}

/* ── which model is running ───────────────────────────────────────────── */

function renderModelId(s) {
  const card = $("modelid-card");
  const card_meta = MODEL && MODEL.model_id;
  const latest = s.latest;
  const mid = latest && latest.model_id;
  if (!card_meta || !mid) { card.hidden = true; return; }
  card.hidden = false;

  const acc = s.model_id || {};
  const gate = $("modelid-gate");

  // The classifier always names one of its ten models. Whether that answer is
  // worth anything is a separate question, and this is where it gets answered.
  if (mid.applicable) {
    gate.className = "modelid-gate applies";
    gate.textContent = mid.basis === "declared"
      ? "The declared workload is LLM inference, so this classifier is being asked a question it was built for."
      : "The workload classifier currently reads ml_inference, so this answer is plausibly meaningful — but nothing has declared what is actually running.";
  } else {
    gate.className = "modelid-gate not-applies";
    gate.textContent = mid.basis === "declared"
      ? `The GPU is running ${s.events.declared_label}, which is not LLM inference. This classifier is closed-set over ${card_meta.classes.length} LLMs and has no way to answer "none of these" — so the name below is meaningless right now, not a detection.`
      : `The workload does not currently read as LLM inference. This classifier can only answer with one of its ${card_meta.classes.length} models, so treat the name below as an artefact.`;
  }

  const v = $("modelid-verdict");
  v.textContent = mid.predicted;
  v.className = "modelid-verdict" + (!mid.applicable ? " muted"
    : (mid.truth && !mid.correct) ? " wrong" : "");

  const det = $("modelid-truth");
  if (mid.truth) {
    det.textContent = mid.correct
      ? `${pct(mid.confidence, 1)} confidence · actually ${mid.truth} · correct`
      : `${pct(mid.confidence, 1)} confidence · actually ${mid.truth} · WRONG`;
  } else {
    det.textContent = `${pct(mid.confidence, 1)} confidence · ground truth unknown`;
  }

  const bars = $("modelid-bars");
  clear(bars);
  const ranked = Object.entries(mid.probs).sort((a, b) => b[1] - a[1]).slice(0, 6);
  for (const [name, prob] of ranked) {
    const row = document.createElement("div");
    row.className = "mid-row" + (name === mid.truth ? " truth" : "");
    row.innerHTML = `<span class="n">${name}</span>` +
      `<span class="track"><span class="fill" style="width:${(100 * prob).toFixed(1)}%"></span></span>` +
      `<span class="v">${pct(prob, 0)}</span>`;
    bars.appendChild(row);
  }

  // Accuracy split by serving batch size. The corpus is a single
  // configuration, so a pooled number would hide the one effect worth seeing.
  const chips = $("modelid-batch");
  clear(chips);
  const trained = acc.trained_batch_size;
  if (acc.scored) {
    const overall = document.createElement("div");
    overall.className = "batch-chip";
    overall.innerHTML = `<b>${pct(acc.accuracy, 1)}</b> overall` +
      `<div class="sub">${acc.correct}/${acc.scored} windows with known truth</div>`;
    chips.appendChild(overall);
  }
  for (const [bs, r] of Object.entries(acc.by_batch || {})) {
    const isTrained = String(trained) === bs;
    const d = document.createElement("div");
    d.className = "batch-chip " + (isTrained ? "trained" : "off");
    d.innerHTML = `batch ${bs} — <b>${pct(r.accuracy, 1)}</b>` +
      `<div class="sub">${r.correct}/${r.scored} windows` +
      `${isTrained ? " · the trained configuration" : " · off-distribution"}</div>`;
    chips.appendChild(d);
  }

  const sc = card_meta.serving_config || "a single serving configuration";
  $("modelid-note").textContent =
    `Trained on ${card_meta.n_runs} runs of ${card_meta.n_models} LLMs at ${sc}. ` +
    (card_meta.memory_features_removed
      ? "All memory-footprint features were removed from this variant, so it cannot simply read model size off the memory gauge — at a fixed batch size that one signal alone reaches 99.4%. "
      : "") +
    "Held-out accuracy on its own corpus is near-perfect, which reflects three runs per model in one configuration rather than a measured error rate. " +
    (trained ? `Windows at any batch size other than ${trained} are outside the distribution it was fitted on.` : "");
}

/* ── raw signals ──────────────────────────────────────────────────────── */

const SIGNALS = [
  ["gpu_utilization_pct", "GPU util", "%"],
  ["mem_utilization_pct", "Mem util", "%"],
  ["mem_used_mb", "Mem used", "MB"],
  ["power_draw_w", "Power", "W"],
  ["temperature_c", "Temp", "°C"],
  ["sm_clock_mhz", "SM clock", "MHz"],
  ["mem_clock_mhz", "Mem clock", "MHz"],
  ["pcie_tx_mbps", "PCIe TX", "MB/s"],
  ["pcie_rx_mbps", "PCIe RX", "MB/s"],
];

function renderSparks(s) {
  const wrap = $("sparks");
  const sp = s.sparklines;
  if (!sp || !sp.n) {
    if (!wrap.dataset.empty) { wrap.innerHTML = '<div class="fp-empty">waiting for telemetry…</div>'; wrap.dataset.empty = "1"; }
    return;
  }
  delete wrap.dataset.empty;

  if (wrap.childElementCount !== SIGNALS.length) {
    clear(wrap);
    for (const [key, label, unit] of SIGNALS) {
      const d = document.createElement("div");
      d.className = "spark";
      d.innerHTML = `<div class="spark-head"><span>${label}</span><span class="spark-val" id="sv-${key}">—</span></div>`;
      const svg = el("svg", { viewBox: "0 0 200 46", id: "sk-" + key });
      d.appendChild(svg);
      wrap.appendChild(d);
    }
  }

  const n = sp.n;
  for (const [key, label, unit] of SIGNALS) {
    const vals = sp.signals[key] || [];
    const svg = $("sk-" + key);
    clear(svg);
    if (!vals.length) continue;

    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 1e-9) { hi = lo + 1; lo -= 0; }
    const pad = (hi - lo) * 0.12;
    lo -= pad; hi += pad;
    const x = (i) => (200 * i) / Math.max(1, n - 1);
    const y = (v) => 42 - 38 * ((v - lo) / (hi - lo));

    // The trailing region the classifier is reading right now.
    const wStart = Math.max(0, n - sp.window_samples);
    svg.appendChild(el("rect", { x: x(wStart), y: 2, width: 200 - x(wStart), height: 42,
                                 fill: "#58a6ff", "fill-opacity": ".08" }));

    svg.appendChild(el("polyline", {
      points: vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "),
      fill: "none", stroke: "#8b96a5", "stroke-width": 1.3,
    }));
    const last = vals[vals.length - 1];
    svg.appendChild(el("circle", { cx: x(n - 1), cy: y(last), r: 2, fill: "#d8dee9" }));
    $("sv-" + key).textContent = `${last.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit}`;
  }
}

/* ── model card ───────────────────────────────────────────────────────── */

function renderModelCard(m) {
  const wrap = $("modelcard");
  clear(wrap);
  const block = (title, rows) => {
    const d = document.createElement("div");
    d.className = "mc-block";
    d.innerHTML = `<h3>${title}</h3><dl>` +
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") + `</dl>`;
    wrap.appendChild(d);
  };

  const p = m.provenance || {};
  block("Provenance", [
    ["GPU", p.gpu_name || "not recorded"],
    ["power regime", m.trained_power_w ? "~" + m.trained_power_w.toFixed(0) + " W" : "not recorded"],
    ["trained", (m.corpus.created_utc || "—").replace("T", " ").replace("+00:00", "Z")],
  ]);

  const c = m.corpus || {};
  block("Training corpus", [
    ["windows", c.n_windows ?? "—"],
    ["runs", c.n_runs ?? "—"],
    ["configs", c.n_configs ?? "—"],
    ["per class", c.runs_per_class
      ? Object.entries(c.runs_per_class).map(([k, v]) => `${k.replace("ml_", "")} ${v}`).join(", ")
      : "—"],
  ]);

  const sw = m.scores.grouped_by_workload, sr = m.scores.grouped_by_run;
  block("Held-out accuracy", [
    ["by workload", sw ? `${pct(sw.accuracy, 1)} (F1 ${sw.macro_f1.toFixed(3)})` : "—"],
    ["by run", sr ? `${pct(sr.accuracy, 1)} (F1 ${sr.macro_f1.toFixed(3)})` : "—"],
    ["use", "the by-workload row"],
  ]);

  block("Feature set", [
    ["variant", m.feature_variant],
    ["count", m.n_features],
    ["window", `${m.window_sec} s, restride 1 s`],
  ]);
}

/* ── stream ───────────────────────────────────────────────────────────── */

function render(s) {
  LAST_STATE = s;
  renderVerdict(s);
  renderLatency(s);
  renderTimeline(s);
  renderLedger(s);
  renderCompare(s);
  renderModelId(s);
  renderSparks(s);

  // Where ground truth comes from decides whether the operator's dropdown is
  // meaningful. With a collection running, the job's process table is
  // authoritative and a manual mark would just be overwritten each second.
  const sel = $("mark-select");
  const ts = s.truth_source;
  if (ts !== lastTruthSource) {
    lastTruthSource = ts;
    const label = sel.previousElementSibling || sel.parentElement.querySelector("span");
    sel.disabled = ts !== "operator";
    if (ts === "collection") {
      if (label) label.textContent = "Ground truth: tracking the collection job";
    } else if (ts === "trace") {
      if (label) label.textContent = "Ground truth: from the replayed trace";
    } else if (label) {
      label.textContent = "Declare what this GPU is really doing";
    }
  }
  if (ts !== "operator") {
    const declared = s.events.declared_label;
    if (declared && sel.value !== declared) {
      if (![...sel.options].some((o) => o.value === declared)) {
        sel.appendChild(new Option(declared, declared));
      }
      sel.value = declared;
    }
  }

  const src = s.source;
  $("source-line").textContent =
    `${src.detail}${src.gpu_name ? " · " + src.gpu_name : ""}` +
    (src.trace_names && src.trace_names.length ? "  ·  " + src.trace_names.join(" → ") : "") +
    `  ·  ${fmt(s.uptime_s, 0)} s of telemetry` +
    (s.exhausted ? "  ·  SOURCE EXHAUSTED" : "");
  $("live-dot").className = "live-dot " + (s.exhausted ? "stale" : "on");
}

function connect() {
  const es = new EventSource("/api/stream");
  es.addEventListener("model", (e) => {
    MODEL = JSON.parse(e.data);
    renderBanner(MODEL);
    renderModelCard(MODEL);
    renderPrecision();
  });
  es.addEventListener("state", (e) => render(JSON.parse(e.data)));
  es.onerror = () => { $("live-dot").className = "live-dot stale"; };
}

/* ── controls ─────────────────────────────────────────────────────────── */

$("mark-select").addEventListener("change", (e) => {
  fetch("/api/mark", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: e.target.value || null }),
  });
});
$("reset-btn").addEventListener("click", () => fetch("/api/reset", { method: "POST" }));
$("base-slider").addEventListener("input", (e) => {
  baseRate = Number(e.target.value) / 100;
  $("base-value").textContent = e.target.value + "%";
  renderPrecision();
});

connect();
