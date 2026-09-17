"""
Step 7: Evaluation harness -- automated metrics against the human golden set.

Reports THREE systems for intent classification, per the "at least two
baselines" requirement:
  - trivial baseline: always predict the majority class
  - simple baseline:  the rule-based weak labeler (src/weak_label.py)
  - our system:       the trained TF-IDF + LogisticRegression classifier

And escalation decision quality (precision/recall/F1 on escalate=positive)
for the deterministic rule engine (src/escalation.py) against gold_escalate.

Run: PYTHONPATH=src python3 eval/run_eval.py
"""
import sys

sys.path.insert(0, "src")

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from weak_label import weak_label
from escalation import decide_escalation


def eval_intent(golden: pd.DataFrame, clf) -> dict:
    y_true = golden["gold_intent"].tolist()

    majority_class = golden["gold_intent"].value_counts().idxmax()
    y_trivial = [majority_class] * len(golden)

    y_simple = golden["customer_clean"].apply(weak_label).tolist()

    proba = clf.predict_proba(golden["customer_clean"])
    classes = clf.classes_
    y_model = [classes[p.argmax()] for p in proba]
    conf_model = [float(p.max()) for p in proba]

    results = {}
    for name, y_pred in [("trivial_majority_class", y_trivial),
                          ("simple_rule_baseline", y_simple),
                          ("trained_classifier", y_model)]:
        results[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    return results, y_model, conf_model


def eval_escalation(golden: pd.DataFrame, y_model_intent, conf_model) -> dict:
    y_true = (golden["gold_escalate"] == "escalate").astype(int).tolist()

    y_pred = []
    for text, intent, conf in zip(golden["customer_clean"], y_model_intent, conf_model):
        decision = decide_escalation(text, intent, conf)
        y_pred.append(int(decision["escalate"]))

    trivial_always_escalate = [1] * len(golden)
    trivial_never_escalate = [0] * len(golden)

    out = {}
    for name, pred in [("trivial_always_escalate", trivial_always_escalate),
                        ("trivial_never_escalate", trivial_never_escalate),
                        ("rule_engine", y_pred)]:
        out[name] = {
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        }
    return out, y_pred, y_true


def main():
    golden = pd.read_csv("data/eval/golden_set.csv")
    clf = joblib.load("models/intent_clf.joblib")

    print("=" * 70)
    print("INTENT CLASSIFICATION vs golden set (n=%d)" % len(golden))
    print("=" * 70)
    intent_results, y_model, conf_model = eval_intent(golden, clf)
    for name, m in intent_results.items():
        print(f"  {name:28s} accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    print("\nPer-class report for TRAINED CLASSIFIER:")
    print(classification_report(golden["gold_intent"], y_model, zero_division=0))

    print("=" * 70)
    print("ESCALATION DECISION vs golden set (positive class = 'escalate')")
    print("=" * 70)
    esc_results, y_pred, y_true = eval_escalation(golden, y_model, conf_model)
    for name, m in esc_results.items():
        print(f"  {name:28s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nRule-engine confusion matrix [rows=true(auto,escalate), cols=pred(auto,escalate)]:")
    print(cm)

    golden_out = golden.copy()
    golden_out["pred_intent"] = y_model
    golden_out["pred_confidence"] = conf_model
    golden_out["pred_escalate"] = ["escalate" if p else "auto" for p in y_pred]
    golden_out["intent_correct"] = golden_out["gold_intent"] == golden_out["pred_intent"]
    golden_out["escalate_correct"] = (golden_out["gold_escalate"] == golden_out["pred_escalate"])
    golden_out.to_csv("data/eval/eval_predictions.csv", index=False)
    print("\nWrote row-level predictions to data/eval/eval_predictions.csv (used for failure analysis)")


if __name__ == "__main__":
    main()
