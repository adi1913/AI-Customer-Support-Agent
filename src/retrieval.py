"""
Step 4: Retrieval over historical resolutions.

Given a new customer message (+ predicted intent), retrieve the top-k most
similar PAST customer messages that AmericanAir already resolved, restricted
to the same intent bucket, and return their agent replies as grounding
examples for reply drafting.

Design choice: TF-IDF cosine similarity, not embeddings. At this corpus size
(36k short texts) and given the intent-bucket restriction already does most
of the topical narrowing, a fitted embedding model would add latency and a
dependency without a clear quality win for this use case -- see decision log.
Restricting retrieval to the SAME predicted intent bucket (rather than
searching the whole 36k corpus) is the retrieval-quality lever that matters
most here; it's cheap and directly uses the classifier's output.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from weak_label import weak_label


class ResolutionRetriever:
    def __init__(self, pairs_csv: str = "data/processed/pairs_clean.csv"):
        df = pd.read_csv(pairs_csv)
        df["weak_intent"] = df["customer_clean"].apply(weak_label)
        # Only index pairs where the agent reply looks like a real, substantive
        # resolution (not just "please DM us") -- crude length filter, documented
        # in decision log as a known blunt instrument.
        df = df[df["agent_clean"].str.len() >= 25].reset_index(drop=True)
        self.df = df

        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(df["customer_clean"])

    def retrieve(self, text: str, intent: str, k: int = 3) -> pd.DataFrame:
        bucket = self.df[self.df["weak_intent"] == intent]
        if len(bucket) == 0:
            bucket = self.df  # fall back to full corpus if the bucket is empty
        bucket_idx = bucket.index.to_numpy()
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.matrix[bucket_idx]).ravel()
        top_k = np.argsort(-sims)[:k]
        result = bucket.iloc[top_k].copy()
        result["similarity"] = sims[top_k]
        return result[["pair_id", "customer_clean", "agent_clean", "similarity"]]


if __name__ == "__main__":
    r = ResolutionRetriever()
    examples = [
        ("Delayed 4hrs coming back from O'hare. No real explanation.", "flight_disruption"),
        ("My bag never showed up at baggage claim, it's been an hour.", "baggage_issue"),
        ("Thanks for the great crew today!", "praise_thanks"),
    ]
    for text, intent in examples:
        print(f"\nQuery ({intent}): {text}")
        hits = r.retrieve(text, intent, k=2)
        for _, row in hits.iterrows():
            print(f"  [{row['similarity']:.2f}] cust: {row['customer_clean'][:80]}")
            print(f"         reply: {row['agent_clean'][:100]}")
