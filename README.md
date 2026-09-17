# AI Support Agent for @AmericanAir

A take-home project: an AI customer-support agent for AmericanAir, built from
the Kaggle "Customer Support on Twitter" dataset. Classifies customer intent,
retrieves and drafts a grounded reply from AmericanAir's own historical
resolutions, and decides auto-handle vs escalate-to-human with a stated reason.

**Read `report/REPORT.md` first** -- it has the problem framing, results
against baselines, failure analysis, and (important) a section on what's
misleading about the headline number. `report/DECISION_LOG.md` has 15
non-obvious decisions and why we made them.

## Quickstart (reproduces headline results in ~20 seconds, no API key needed)

```bash
pip install -r requirements.txt
bash run_all.sh
```

This trains the intent classifier, runs the evaluation harness against the
195-example human golden set, and sanity-checks the end-to-end agent -- using
the already-committed processed data (`data/processed/pairs_clean.csv`,
`data/eval/golden_set.csv`). **No Kaggle download and no API key required**
for this path.

## What's committed vs. regenerated

| File | Committed? | Why |
|---|---|---|
| `data/raw/twcs.csv` (full ~500MB Kaggle dump) | No | Too large for git; download it yourself if you want to rebuild from scratch (see below) |
| `data/processed/pairs_clean.csv` (36,065 cleaned AmericanAir pairs, ~18MB) | Yes | This is what training/eval actually runs against |
| `data/processed/pairs_sample.csv` (6k time-stratified subsample) | Yes | For quick experimentation without touching the full set |
| `data/eval/golden_set.csv` (195 hand-labeled examples) | Yes | The evaluation ground truth |
| `models/intent_clf.joblib` | No | Regenerates in ~2 seconds via `src/train_classifier.py` |

## Rebuilding from scratch (optional, needs the raw Kaggle file)

1. Download `thoughtvector/customer-support-on-twitter` from Kaggle, unzip,
   place `twcs.csv` at `data/raw/twcs.csv`.
2. `PYTHONPATH=src python3 src/clean_data.py` -- reconstructs and cleans all
   AmericanAir pairs.
3. `PYTHONPATH=src python3 eval/build_golden_set.py` -- resamples the golden-
   set template (only needed if you want to change the sampling; the labeled
   set is already committed).
4. `bash run_all.sh` as above.

## Pipeline / architecture

```
data/raw/twcs.csv
      |  src/clean_data.py  (thread reconstruction + text cleaning)
      v
data/processed/pairs_clean.csv  (36,065 AmericanAir customer<->agent pairs)
      |
      +-- src/weak_label.py --------> rule-based intent labels (simple baseline
      |                               AND weak-supervision training signal)
      |
      +-- src/train_classifier.py --> models/intent_clf.joblib
      |                               (TF-IDF + LogisticRegression, excludes
      |                                golden-set pair_ids from training)
      |
      +-- src/retrieval.py ---------> TF-IDF retrieval over historical
                                       resolved replies, scoped to predicted intent
      v
src/agent.py  (SupportAgent: classify -> retrieve -> src/escalation.py -> draft reply)
      |
      +-- template mode (default, offline) or LLM mode (if ANTHROPIC_API_KEY set)
```

Evaluation: `eval/build_golden_set.py` (sampling) -> `eval/gold_labels.py`
(hand labels, read the docstring for methodology and an important caveat
about label provenance) -> `eval/merge_gold_labels.py` -> `eval/run_eval.py`
(automated metrics) + `eval/llm_judge.py` (reply-quality LLM judge, requires
an API key) + `eval/label_cli.py` (tool to review/finish labels by hand).

## Repo layout

```
src/               core pipeline (data cleaning, weak labeling, classifier,
                   retrieval, escalation, the agent itself)
eval/              golden-set construction/labels, evaluation harness,
                   LLM judge, human-labeling CLI
data/raw/          raw Kaggle CSV (not committed -- see above)
data/processed/    cleaned pairs + subsample (committed)
data/eval/         golden set + evaluation outputs (committed)
models/            trained classifier (not committed, regenerates in seconds)
report/            REPORT.md (deliverable #4) and DECISION_LOG.md (deliverable #5)
run_all.sh         one-command reproduction
```

## Using an LLM for real grounded replies (optional)

Set `ANTHROPIC_API_KEY` and `SupportAgent` automatically switches from
template-filled replies to Claude-drafted replies grounded in the retrieved
historical examples:

```bash
export ANTHROPIC_API_KEY=sk-...
PYTHONPATH=src python3 src/agent.py
python3 eval/llm_judge.py --n 30   # also needs the key, scores reply quality
```

## Known limitations (see report/REPORT.md Section C for the full failure
analysis with real examples)

- Single-turn scoping only (first customer message -> first agent reply);
  multi-turn thread context isn't used yet.
- Weak-supervision circularity: the trained classifier is trained on labels
  from the same rules used as the "simple baseline" it's compared against.
- The golden set is stratified by the weak label, which structurally favors
  the rule baseline in the current headline numbers -- see Section D of the
  report before trusting the "rules beat the model" result at face value.
- Golden labels are a single LLM pass (documented in `eval/gold_labels.py`),
  not yet a human-reviewed ground truth -- `eval/label_cli.py` exists to fix
  this and should be run before treating these numbers as final.
