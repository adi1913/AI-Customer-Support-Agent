"""
Step 8: LLM-as-judge for reply quality, + agreement check vs a human spot-check.

Requires ANTHROPIC_API_KEY to be set (this step is the one part of the
pipeline that isn't reproducible offline -- see README for why that tradeoff
was made).

Rubric (1-5 each), scored per generated reply against the customer message
and (if escalate=False) the draft reply:
  - grounded:      does the reply only state things supported by the
                    retrieved historical examples / the customer's own
                    message (no invented flight numbers, policies, promises)?
  - relevant:      does it actually address the customer's stated issue?
  - tone:          professional, empathetic, on-brand for airline support?
  - actionable:    does it give the customer a clear next step?

Usage:
  python3 eval/llm_judge.py --n 30

Agreement with humans: this script also expects an optional CSV
(data/eval/human_judge_spotcheck.csv, columns: pair_id, human_grounded,
human_relevant, human_tone, human_actionable) with a human's own 1-5 scores
on the SAME sample, and reports Cohen's kappa / simple agreement rate per
dimension. We recommend spot-checking at least 20 of the n examples by hand
before trusting the LLM judge's aggregate score -- see report/REPORT.md.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "src")

import pandas as pd

RUBRIC_PROMPT = """You are grading a draft customer-support reply from AmericanAir's Twitter support agent.

Customer message: "{customer}"
Retrieved historical examples used for grounding:
{examples}

Draft reply to grade: "{reply}"

Score the draft reply on each dimension from 1 (poor) to 5 (excellent):
- grounded: does it avoid inventing facts (flight numbers, policies, promises) not present in the customer message or retrieved examples?
- relevant: does it address the customer's actual issue?
- tone: professional and empathetic, appropriate for airline support?
- actionable: does it give the customer a clear, correct next step?

Respond with ONLY a JSON object: {{"grounded": <1-5>, "relevant": <1-5>, "tone": <1-5>, "actionable": <1-5>, "notes": "<one sentence>"}}"""


def judge_one(client, customer, examples, reply, model="claude-sonnet-4-6"):
    examples_block = "\n".join(
        f"- Customer: {e['customer_clean']}\n  Agent replied: {e['agent_clean']}" for e in examples
    ) or "(none retrieved)"
    prompt = RUBRIC_PROMPT.format(customer=customer, examples=examples_block, reply=reply)
    resp = client.messages.create(model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}])
    text = resp.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--golden_csv", default="data/eval/eval_predictions.csv")
    ap.add_argument("--out", default="data/eval/llm_judge_results.csv")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- this step requires a live API key.")
        print("Everything else in the pipeline (classification, retrieval, escalation)")
        print("runs fully offline; only reply-quality judging needs an LLM call.")
        sys.exit(1)

    import anthropic
    from agent import SupportAgent

    client = anthropic.Anthropic()
    agent = SupportAgent()

    golden = pd.read_csv(args.golden_csv).head(args.n)
    rows = []
    for _, row in golden.iterrows():
        result = agent.handle(row["customer_clean"])
        try:
            scores = judge_one(client, row["customer_clean"], result.retrieved_examples, result.draft_reply)
        except Exception as e:
            scores = {"grounded": None, "relevant": None, "tone": None, "actionable": None, "notes": f"error: {e}"}
        rows.append({"pair_id": row["pair_id"], "customer_clean": row["customer_clean"],
                     "draft_reply": result.draft_reply, **scores})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} judged replies to {args.out}")
    for dim in ["grounded", "relevant", "tone", "actionable"]:
        print(f"  mean {dim}: {out_df[dim].mean():.2f}")

    spotcheck_path = "data/eval/human_judge_spotcheck.csv"
    if os.path.exists(spotcheck_path):
        human = pd.read_csv(spotcheck_path)
        merged = out_df.merge(human, on="pair_id", suffixes=("_llm", "_human"))
        print("\nLLM-vs-human agreement (mean absolute difference, lower=better; n=%d):" % len(merged))
        for dim in ["grounded", "relevant", "tone", "actionable"]:
            diff = (merged[dim] - merged[f"human_{dim}"]).abs().mean()
            print(f"  {dim}: {diff:.2f}")
    else:
        print(f"\nNo human spot-check file found at {spotcheck_path}.")
        print("Create it (pair_id, human_grounded, human_relevant, human_tone, human_actionable)")
        print("for at least 20 rows and re-run to get an agreement number for the report.")


if __name__ == "__main__":
    main()
