"""
Step 1: Load the raw Kaggle 'Customer Support on Twitter' dump, filter to a
single brand (AmericanAir), reconstruct customer -> agent reply pairs using
the in_response_to_tweet_id / response_tweet_id thread fields, and clean text.

Input:  data/raw/twcs.csv          (full ~2.8M row dataset; NOT committed to git)
Output: data/processed/pairs_clean.csv    (all ~36k AmericanAir pairs, cleaned)
        data/processed/pairs_sample.csv   (stratified-by-time subsample, committed to git,
                                            used for the <15 min reproduction path)

Design notes (see report/DECISION_LOG.md for the full rationale):
- We only reconstruct the FIRST customer message -> FIRST agent reply per thread.
  Multi-turn context is real and present in the data, but scoping to the first
  exchange keeps intent labels unambiguous (a thread's topic can drift across
  turns) and keeps the eval harness tractable. This is called out explicitly
  in the report as something we chose NOT to build.
- We drop pairs where the customer's parent tweet isn't inbound (i.e. AA replying
  to itself or to another company) -- there were zero of these for AmericanAir,
  but the check stays in for safety / other-brand reuse.
- Cleaning strips @mentions and URLs (they carry no intent signal and are mostly
  anonymized numeric handles in this dataset) and HTML-unescapes entities like &amp;.
  We do NOT strip emoji, hashtags, or punctuation -- sentiment-bearing signal
  ("#worstairlineever", "😡") is useful for the escalation model.
"""
import argparse
import html
import re

import pandas as pd

BRAND = "AmericanAir"
MIN_CLEAN_LEN = 8  # drop customer messages that are pure @mention/URL noise


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = html.unescape(text)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_pairs(raw_csv: str, brand: str = BRAND) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, dtype=str)
    df["inbound"] = df["inbound"] == "True"

    brand_rows = df[df["author_id"] == brand].copy()

    idx = df.set_index("tweet_id")
    idx = idx[~idx.index.duplicated(keep="first")]

    pairs = []
    for _, row in brand_rows.iterrows():
        parent_id = row["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id not in idx.index:
            continue
        parent = idx.loc[parent_id]
        if not parent["inbound"]:
            continue
        pairs.append(
            {
                "customer_tweet_id": parent_id,
                "customer_text": parent["text"],
                "customer_time": parent["created_at"],
                "agent_tweet_id": row["tweet_id"],
                "agent_text": row["text"],
                "agent_time": row["created_at"],
            }
        )

    pdf = pd.DataFrame(pairs)
    pdf["customer_clean"] = pdf["customer_text"].apply(clean_text)
    pdf["agent_clean"] = pdf["agent_text"].apply(clean_text)
    pdf = pdf[pdf["customer_clean"].str.len() >= MIN_CLEAN_LEN].reset_index(drop=True)

    pdf["customer_time"] = pd.to_datetime(
        pdf["customer_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    pdf = pdf.sort_values("customer_time").reset_index(drop=True)
    pdf["pair_id"] = [f"AA-{i:06d}" for i in range(len(pdf))]
    return pdf


def stratified_time_sample(pdf: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample roughly evenly across the date range so the committed subsample
    isn't all from one month (the raw data spans Aug 2015 - Sep 2017)."""
    pdf = pdf.dropna(subset=["customer_time"]).copy()
    month = pdf["customer_time"].dt.tz_localize(None).dt.to_period("M")
    frac = n / len(pdf)
    parts = [g.sample(frac=frac, random_state=seed) for _, g in pdf.groupby(month) if len(g) > 0]
    sample = pd.concat(parts)
    return sample.sort_values("customer_time").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/twcs.csv")
    ap.add_argument("--out_full", default="data/processed/pairs_clean.csv")
    ap.add_argument("--out_sample", default="data/processed/pairs_sample.csv")
    ap.add_argument("--sample_size", type=int, default=6000)
    args = ap.parse_args()

    pdf = build_pairs(args.raw)
    print(f"Built {len(pdf)} cleaned pairs for {BRAND}")
    pdf.to_csv(args.out_full, index=False)

    sample = stratified_time_sample(pdf, args.sample_size)
    print(f"Sampled {len(sample)} pairs for the committed reproduction subset")
    sample.to_csv(args.out_sample, index=False)
