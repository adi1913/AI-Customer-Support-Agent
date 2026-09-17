"""
Build the golden evaluation set template.

Sampling method (documented for the report's methodology section):
- We stratify by the WEAK label (not the true label, which we don't have yet)
  so that rare-looking categories aren't swamped by praise/info/other, which
  dominate the raw distribution 3:1. This is a common two-stage design:
  cheap noisy labels decide *what to sample*, humans decide the *ground truth*.
- We deliberately over-sample "other_unclear" too, because that bucket is
  where the weak labeler is most likely to be systematically wrong (e.g. non-
  English tweets, complaints phrased without our keyword list) -- we want the
  golden set to be able to catch and quantify that failure mode, not hide it.
- Target: ~25 examples per intent bucket x 8 real intents + 20 "other_unclear"
  = ~220, trimmed down to the 150-250 range Hiver asked for.
- We exclude any pair_id that appears in the committed pairs_sample.csv training
  data conceptually, but since the TF-IDF classifier is trained on the weak-
  labeled FULL set (not just the sample), we instead hold out the golden set's
  pair_ids from classifier training explicitly in train_classifier.py.
"""
import pandas as pd

from weak_label import INTENTS, weak_label

PER_CLASS_TARGET = {
    "flight_disruption": 25,
    "baggage_issue": 22,
    "booking_checkin_issue": 22,
    "refund_compensation_request": 22,
    "upgrade_loyalty": 20,
    "service_complaint": 22,
    "info_question": 22,
    "praise_thanks": 20,
    "other_unclear": 20,
}


def main(seed: int = 42):
    df = pd.read_csv("data/processed/pairs_clean.csv")
    df["weak_intent"] = df["customer_clean"].apply(weak_label)

    parts = []
    for intent, k in PER_CLASS_TARGET.items():
        bucket = df[df["weak_intent"] == intent]
        k = min(k, len(bucket))
        parts.append(bucket.sample(n=k, random_state=seed))

    golden = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    golden = golden[
        ["pair_id", "customer_clean", "agent_clean", "weak_intent", "customer_time"]
    ].rename(columns={"weak_intent": "weak_intent_reference_only"})

    golden["gold_intent"] = ""
    golden["gold_escalate"] = ""  # "auto" or "escalate"
    golden["gold_reason"] = ""
    golden["labeled_by"] = ""

    golden.to_csv("data/eval/golden_set_template.csv", index=False)
    print(f"Wrote {len(golden)} rows to data/eval/golden_set_template.csv")
    print(golden["weak_intent_reference_only"].value_counts())


if __name__ == "__main__":
    main()
