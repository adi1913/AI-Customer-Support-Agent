import pandas as pd

from gold_labels import LABELS

df = pd.read_csv("data/eval/golden_set_template.csv")

missing = [pid for pid in df["pair_id"] if pid not in LABELS]
if missing:
    raise SystemExit(f"{len(missing)} pair_ids missing gold labels: {missing[:10]}...")

extra = [pid for pid in LABELS if pid not in set(df["pair_id"])]
if extra:
    print(f"WARNING: {len(extra)} labels in gold_labels.py don't match any row (stale?): {extra[:5]}")

df["gold_intent"] = df["pair_id"].map(lambda p: LABELS[p][0])
df["gold_escalate"] = df["pair_id"].map(lambda p: LABELS[p][1])
df["gold_reason"] = df["pair_id"].map(lambda p: LABELS[p][2])
df["labeled_by"] = "claude_v0_single_pass"

df.to_csv("data/eval/golden_set.csv", index=False)
print(f"Wrote {len(df)} labeled rows to data/eval/golden_set.csv")
print("\nIntent distribution:")
print(df["gold_intent"].value_counts())
print("\nEscalation distribution:")
print(df["gold_escalate"].value_counts())

agree = (df["gold_intent"] == df["weak_intent_reference_only"]).mean()
print(f"\nWeak-labeler vs gold agreement (simple-baseline accuracy proxy): {agree:.1%}")
