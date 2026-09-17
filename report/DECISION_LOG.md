# Decision Log

Non-obvious decisions made while building this, and why. (Deliverable #5.)

1. **Brand: AmericanAir, not a bigger brand.** AmazonHelp/AppleSupport have
   more data, but airline complaints map to a small set of crisp, publicly
   legible intents (delay, bags, refund...), which makes labeling and grading
   far less subjective than "tech support" catch-alls. Optimized for eval
   rigor over raw data volume.

2. **Single-turn pairs only, not full threads.** ~99.5% of AA replies resolve
   to an immediate parent tweet, so thread reconstruction is easy -- but we
   scope the agent to first customer message -> first agent reply. Multi-turn
   context is real in this data and would improve grounding, but a thread's
   topic can drift, which would make intent labels ambiguous and blow up the
   labeling effort. Explicitly listed as future work.

3. **9 intents, not fewer.** We resisted collapsing to ~5 because each of the
   9 buckets implies a genuinely different escalation policy (praise never
   escalates; service_complaint escalates on a discrimination flag regardless
   of severity language). Fewer buckets would hide policy-relevant distinctions.

4. **Weak supervision (regex rules) instead of hand-labeling all 36k pairs.**
   We don't have the labeling budget for that, and it isn't what a real team
   would do either. The rules are transparent and auditable (you can read
   exactly why any given label was assigned), which matters because they're
   also the "simple baseline" -- the thing the model has to be compared against
   honestly, not a hidden black box.

5. **The weak labeler is used as BOTH the training-label source and the
   simple baseline.** We flag the circularity risk explicitly rather than
   hide it: the trained classifier's apparent skill over the rule baseline on
   rule-labeled data is partly just interpolation between the rules' own
   keyword patterns. The golden set (human-labeled, held out from training)
   is the only trustworthy comparison, and even that has a sampling caveat --
   see report/REPORT.md, "what's misleading about my headline number."

6. **TF-IDF + Logistic Regression, not a transformer/embedding classifier.**
   Trains in seconds on CPU (keeps the 15-minute repro promise), and the
   per-class top-weighted n-grams are directly inspectable for debugging weak
   labels. A transformer might classify a few points higher but we don't have
   a labeled set large enough to know that for sure, and the interpretability
   loss isn't worth it for a v1.

7. **Retrieval restricted to the predicted-intent bucket, not the full
   corpus.** The single biggest lever for retrieval relevance here is topical
   narrowing, and the classifier already does that narrowing for free. Full-
   corpus embedding search was considered and rejected as added complexity
   without a clear win at this corpus size (36k short texts).

8. **Escalation is a deterministic rule engine, not an LLM call.** This is
   the one decision in the pipeline with real safety/liability weight (routing
   discrimination complaints, medical requests, etc. to a human). We wanted it
   auditable via literal string-matched reasons, not a black-box judgment call
   that could silently change between runs or model versions.

9. **Reply drafting has two backends (template vs LLM), selected automatically
   by whether ANTHROPIC_API_KEY is set.** This was the only way to keep the
   headline classification/retrieval/escalation metrics reproducible in <15
   minutes with zero external dependencies, while still supporting the "real"
   grounded-generation version of the agent for anyone who plugs in a key.

10. **The committed dataset is a 6k-pair time-stratified subsample, not the
    full 36k.** Committing the full 500MB raw Kaggle CSV to git is a non-
    starter; the processed subsample (~3MB) is what the README's <15 min path
    actually runs against. The full pipeline against all 36k pairs is one
    extra command for anyone who downloads the raw Kaggle file themselves.

11. **Golden-set sampling stratifies on the WEAK label, not a random sample.**
    A pure random sample would be >60% praise/info/other and starve the rare-
    but-important categories (refund, baggage) of enough golden examples to
    measure per-class recall meaningfully. The tradeoff (discussed in the
    report) is that this inflates the simple baseline's apparent accuracy on
    the golden set, since the golden set was partly selected BY that baseline.

12. **Golden labels are a single Claude pass, explicitly marked v0, not final.**
    We chose honesty about provenance over the appearance of "195 human labels"
    -- see eval/gold_labels.py docstring and report/REPORT.md. A team member
    doing a second human pass (eval/label_cli.py exists for exactly this) is
    treated as a required step before submission, not an optional nice-to-have.

13. **Length filter (>=25 chars) on retrieval-indexed replies.** A large chunk
    of raw agent replies are just "Please DM us your confirmation number" --
    real but low-information as a grounding example for a NEW reply. This is a
    blunt instrument (documented as such) that trades some recall for keeping
    retrieved examples substantive.

14. **Escalation confidence threshold (0.45) picked by inspection, not tuned
    on the golden set.** Tuning it against golden-set escalation labels would
    overfit a threshold to 195 examples and make the reported escalation
    metrics circular. It's a round, conservative-ish number chosen by looking
    at the classifier's confidence distribution alone; the report's failure
    analysis discusses what tuning it *would* buy, without doing so.

15. **URLs and @mentions are stripped in cleaning, but emoji/hashtags/
    profanity are kept.** They're near-content-free (mostly anonymized
    numeric handles) but sentiment- and urgency-bearing signal like emoji and
    profanity is exactly what the escalation rules key off of.
