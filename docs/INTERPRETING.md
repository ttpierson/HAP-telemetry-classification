# Reading the dashboard

What each panel means, what a good reading looks like, and how each can
mislead you.

In short, **this dashboard is for questioning a classifier. It is not a
detector to trust.** Each panel puts the ways it could flatter the model on
screen next to the result.

---

## The banner (top of page)

Read the banner first. It is the only part of the page that can tell you the
rest of the page is meaningless.

| Colour | Meaning |
|---|---|
| **Green** | The loaded model matches this GPU model and power regime. |
| **Amber** | Something could not be checked, usually because the bundle records no power regime. A claim that can't be checked is not a passing one. |
| **Red** | A definite mismatch. Read the message: the numbers below are miscalibrated. |

The most important red case is **power regime**. A GPU capped at 200 W and one
running at 350 W look like different machines. A model trained on one treats
every workload on the other as unlike anything it has seen: a capped GPU pins
near 200 W under any real load, which never happens in a 350 W corpus. Live,
the banner compares the GPU's enforced power limit with the training corpus's
regime (its recorded limit, or else its peak draw). In replay, where no limit
is reported, it compares the trace's peak draw instead.

The other red case is **in-sample replay**: replaying a run the model was
trained on. See "The honesty problem" below.

---

## Current call

The three class probabilities, updated every second from the last 30 s.

Read the fill note under the bar. Until the window is full, the classifier is
working from partial evidence, and the panel says how much. A confident call
on an 8-second window means less than one on a full window.

---

## Same window, two models

Look at this panel hardest. It is what makes the rest of the dashboard worth
trusting.

Two bundles score the **identical** window: same telemetry, same features. The
only difference is what each was trained on, and the badge shows which:

| Badge | Meaning |
|---|---|
| **TRAINED ON THIS RUN** (red) | Scoring its own training data. Its output here is not evidence of anything. |
| **TRAINED ON THIS WORKLOAD** (amber) | Never saw this run, but did see this configuration. It is recognising a workload it has seen, as in grouped-by-run scoring. |
| **NEVER SAW THIS WORKLOAD** (green) | The meaningful case: the configuration was held out entirely, as in grouped-by-workload scoring. |

**When the two disagree, the more confident model is not the better one. It is
the one closer to its own training data.**

In the original study the gap was large. On one real `train_resnet18_bs64`
trace, the model that had seen that config called 601 of 604 windows training
(mean confidence 0.98). The model that had never seen it called 431 of 604
windows *inference* (mean confidence 0.55). A single-model dashboard would
only have shown the first result.

To reproduce this, train one bundle normally and one with
`--holdout-config train_resnet18_bs64`, then load them as `--model` and
`--compare-model`.

---

## Panel 1 · Detection latency

**The question:** once a workload starts, how many seconds until the
classifier is confident?

**How to read it.** The big number is the time from workload start to a
confident *and correct* call that holds for 3 consecutive seconds. One lucky
second is not a detection. The chart plots each class probability from t₀,
with the confidence threshold as a dashed line.

**The shaded band is the point of this panel.** For the first 30 seconds, the
window still includes samples from *before* the workload started. Confidence
rising inside that band is mostly old samples leaving the window, not the model
deciding. The original study measured about 31 s from an idle→training switch
to a confident correct call, with a 30 s window. Detection latency is set by
window length, not by the model. A shorter window would cut it, at some cost
in accuracy (`train_classifier.py --window`).

**Onset comes from utilisation crossing a threshold, not from the classifier's
output.** If it came from the model's own predictions, "time from onset to
confident" would be measuring the model against itself.

**When it shows "—".** If the workload was already running when monitoring
started, there is no before and after: the window held that workload from the
first sample. A latency here would just measure the window filling up, and it
would be misleadingly small. The panel says so and waits for the next real
transition.

---

## Panel 2 · Workload transitions

**The question:** when inference switches to training mid-stream, how long
does the classifier take to follow, and what does it do in between?

Class probabilities are drawn as a stacked area, so the boundary between
colours *is* the decision. Below it are two bands: **declared** (what the GPU
was really doing) and **predicted**.

**Read the horizontal offset between the two bands.** That gap is the
detection lag. Confusion during the switch shows up as the predicted band
changing back and forth while the declared band is already steady. Lag shows
up as the predicted band keeping the old class well after the switch.

To produce a transition offline, give `--trace` twice (e.g. an `infer_*`
trace, then a `train_*` trace).

Red ticks along the top are false alarms. The episode list below gives each
transition's measured lag.

---

## Panel 3 · False positives, accumulating

**The question:** with the GPU idle or running non-ML work, how often does the
classifier wrongly report `ml_training`? Each false alarm shows up as it
happens.

The counters only move when the true workload is known: from a replayed
trace's own labels, from `--track-collection` (which reads the running
`collect_telemetry.py` command), or from the "declare" dropdown. Without it,
the panel stays blank rather than guessing.

**The base-rate slider matters most.** Drag it and watch precision change
while the model stays the same. Recall and false-positive rate come from the
loaded bundle's own held-out (grouped-by-workload) confusion matrix, not from
any published headline figure.

In the original study, a held-out model with 69% recall and 14% false-positive
rate reached 89% precision when training made up 62% of GPU time (as in the
research corpus). At 5%, a plausible share for a real fleet, precision fell to
21%, meaning **about four in five training alarms were false**. More data
raised recall but not precision, because the false-positive rate rose along
with it.

**Missed detections are the other half, and on this corpus the bigger
problem.** For detecting *undeclared* training, a miss is the failure that
matters, and it is the one headline accuracy hides best.

---

## Which model is running?

A second classifier (`--model-id-model`) answering a different question: not
*what kind of work* this is, but *which LLM*. It scores the same 30 s window.

**Read the coloured gate above the verdict first.** This classifier is
**closed-set**: its probabilities always sum to one across the models it was
trained on. It has no "none of these" option. Point it at an idle GPU, a
ResNet, or an LLM outside its set, and it will still name one of its models,
sometimes confidently. The gate is green when the workload really is LLM
inference and amber when the name below is an artefact, not a detection.

**The batch-size chips are the key finding.** The model-ID corpus uses one
serving configuration (batch 4, 256-token prompts, 64 new tokens), and
accuracy depends heavily on it. In the original study, accuracy was perfect at
batch 4 and at or below chance at other batch sizes. At batch 16 and 32, every
model was called `mistral_7b`, the largest in the set. A larger batch does
more work per step, and at batch 4 that was the sign of a bigger model. The
classifier learned "how hard is the GPU working" as a stand-in for "how big is
the model."

`mem_used_mb_mean` alone nearly identifies the model, because fp16 weights take
~2 bytes per parameter. The no-memory variant avoids that shortcut but falls
into a similar one: load. Both shortcuts shift with batch size.
`train_classifier.py model_id` on a batch-sweep directory measures
this directly by holding out one batch size at a time.

**So the honest claim is narrow:** at one fixed serving configuration,
telemetry tells these LLMs apart almost perfectly. Whether that survives a
changing configuration is not shown, and the sweep suggests it does not.

---

## The nine signals

`gpu_utilization_pct`, `mem_utilization_pct`, `mem_used_mb`, `power_draw_w`,
`temperature_c`, `sm_clock_mhz`, `mem_clock_mhz`, `pcie_tx_mbps` and
`pcie_rx_mbps` are the classifier's entire input. It doesn't instrument the
workload or see model weights or data.

The shaded box on each sparkline marks the exact 30 s the classifier is
reading right now. On a power-capped GPU, `power_draw_w` stays flat at the cap
under any real load. This is why a model trained without a cap can't read a
capped GPU.

---

## The honesty problem, stated plainly

A model scoring data it was trained on looks flawless: near-zero false alarms,
instant detection, 98% confidence. None of that generalises. Three defences
are built in:

1. Bundles record `train_run_ids`. Replaying one of those runs turns the
   banner red.
2. `train_classifier.py --holdout-config` builds a model with named configs
   left out, so replaying those configs is truly out-of-sample.
3. `--compare-model` shows both models on screen at once.

**If you show this to anyone, show the held-out model.** The in-sample one is
there to show what you would have believed without it.

---

## What the numbers are not

The model card shows the loaded bundle's grouped-by-workload accuracy. That's
the row to read. Grouped-by-run keeps other runs of the same configuration in
training, so it measures recognising a known workload, not generalising.

The corpus covers one GPU model (RTX 3090), 3 repetitions per configuration,
and **no adversarial workloads**. Nothing here has been tested against anyone
trying to hide training. This is a research prototype. Its output is what one
random forest infers from nine management signals, not the truth about what a
GPU is doing.
