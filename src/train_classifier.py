"""
Step 3: Train the intent classifier.

Trained on WEAK labels (src/weak_label.py) over the full 36k-pair set, with the
195 golden_set pair_ids explicitly excluded from training so evaluation against
gold labels is not contaminated. Saves the fitted pipeline to models/intent_clf.joblib.

Why TF-IDF + Logistic Regression instead of a transformer:
- Runs in seconds on a laptop CPU, keeping the "<15 min reproduction" promise.
- Fully interpretable (inspect top-weighted n-grams per class) -- valuable when
  the label source is noisy weak supervision and you need to debug it.
- On short, keyword-driven text like this, a linear bag-of-words model is a
  strong baseline; the interesting question (answered in the report) is
  whether it generalizes BEYOND the weak labeler's own keyword list, not
  whether it beats a transformer.
See report/DECISION_LOG.md for the full comparison against alternatives considered.
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from weak_label import weak_label


def main(golden_csv: str, out_model: str, full_csv: str = "data/processed/pairs_clean.csv"):
    df = pd.read_csv(full_csv)
    df["weak_intent"] = df["customer_clean"].apply(weak_label)

    golden_ids = set(pd.read_csv(golden_csv)["pair_id"])
    train_df = df[~df["pair_id"].isin(golden_ids)].copy()
    print(f"Training on {len(train_df)} pairs (excluded {len(golden_ids)} golden-set pairs)")

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["customer_clean"], train_df["weak_intent"],
        test_size=0.1, random_state=42, stratify=train_df["weak_intent"],
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=2.0
        )),
    ])
    pipe.fit(X_train, y_train)

    val_acc = pipe.score(X_val, y_val)
    print(f"Held-out weak-label validation accuracy: {val_acc:.3f}")
    print("(NOTE: this is accuracy against MORE weak labels, not human ground truth --")
    print(" it mainly confirms the model learned the rules' patterns. The number that")
    print(" matters is in eval/run_eval.py, scored against the human golden set.)")

    joblib.dump(pipe, out_model)
    print(f"Saved model to {out_model}")

    with open(out_model.replace(".joblib", "_meta.json"), "w") as f:
        json.dump({"held_out_weak_val_accuracy": val_acc, "n_train": len(X_train)}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden_csv", default="data/eval/golden_set.csv")
    ap.add_argument("--out_model", default="models/intent_clf.joblib")
    args = ap.parse_args()
    main(args.golden_csv, args.out_model)
