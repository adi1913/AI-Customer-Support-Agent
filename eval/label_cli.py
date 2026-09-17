"""
Simple terminal tool for a human to review or finish golden-set labels.

Usage:
  python3 eval/label_cli.py                  # review every row, Enter to keep existing label
  python3 eval/label_cli.py --unlabeled-only  # only stop on rows with no gold_intent yet

Saves incrementally to data/eval/golden_set.csv after every row (Ctrl+C safe).
"""
import argparse
import sys

import pandas as pd

INTENTS = [
    "flight_disruption", "baggage_issue", "booking_checkin_issue",
    "refund_compensation_request", "upgrade_loyalty", "service_complaint",
    "info_question", "praise_thanks", "other_unclear",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval/golden_set.csv")
    ap.add_argument("--unlabeled-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    print("Intents:", ", ".join(f"{i+1}={x}" for i, x in enumerate(INTENTS)))
    print("For each example: enter intent number (or blank to keep), then")
    print("'a' for auto-handle / 'e' for escalate (or blank to keep). Ctrl+C to stop.\n")

    for i, row in df.iterrows():
        if args.unlabeled_only and isinstance(row.get("gold_intent"), str) and row["gold_intent"]:
            continue
        print(f"\n[{row['pair_id']}] current gold_intent={row.get('gold_intent')!r} "
              f"gold_escalate={row.get('gold_escalate')!r}")
        print(f"  customer: {row['customer_clean']}")
        print(f"  agent reply on file: {str(row.get('agent_clean'))[:150]}")
        try:
            intent_in = input("  intent # (blank=keep): ").strip()
            esc_in = input("  a/e (blank=keep): ").strip().lower()
            reason_in = input("  reason (blank=keep): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if intent_in:
            df.at[i, "gold_intent"] = INTENTS[int(intent_in) - 1]
        if esc_in == "a":
            df.at[i, "gold_escalate"] = "auto"
        elif esc_in == "e":
            df.at[i, "gold_escalate"] = "escalate"
        if reason_in:
            df.at[i, "gold_reason"] = reason_in
        df.at[i, "labeled_by"] = "human_review"

        df.to_csv(args.csv, index=False)

    print("\nDone. Saved to", args.csv)


if __name__ == "__main__":
    main()
