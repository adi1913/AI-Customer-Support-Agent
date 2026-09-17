# Report: AI Support Agent for @AmericanAir

## A. Problem framing

**Brand:** AmericanAir (36,065 reconstructed customer->agent reply pairs from
the Kaggle "Customer Support on Twitter" dataset, Aug 2015 - Sep 2017).

**What "good" means here:** for a support agent handling a Twitter firehose,
"good" is not "sounds fluent." It's three separable things, each measured
separately in this project:
1. **Correct triage** -- does the intent label match what the customer
   actually needs, well enough that downstream routing/reply logic is sane?
2. **Correct escalation** -- does the system send the RIGHT messages to a
   human (safety/legal/retention-risk, chronic failures, ambiguous cases) and
   let the rest go on a fast, auto-handled path? Recall on true escalations
   matters more than precision here -- a missed escalation is a customer
   whose discrimination complaint gets a templated apology.
3. **Grounded, not hallucinated, replies** -- does the drafted reply only
   claim things supported by the customer's message or actual historical
   resolutions, rather than inventing policy or promises?

**What we chose NOT to build, and why:**
- *Multi-turn conversation modeling.* Real, resolvable in this data, but
  scopes up the labeling problem (a thread's topic drifts) for a v1. See
  decision log #2.
- *A trained escalation model.* We use a deterministic rule engine instead --
  this is the one component where auditability mattered more than possible
  accuracy gains from an ML approach. See decision log #8.
- *Full-corpus embedding retrieval.* TF-IDF within the predicted intent
  bucket instead -- see decision log #7.
- *A fine-tuned/transformer intent classifier.* TF-IDF + LogisticRegression
  instead, for speed and interpretability. See decision log #6.

## B. Results

Evaluated against the 195-example human golden set (`data/eval/golden_set.csv`;
sampling and labeling methodology below and in `eval/build_golden_set.py` /
`eval/gold_labels.py`).

### Intent classification

| System | Accuracy | Macro-F1 |
|---|---|---|
| Trivial baseline (always predict majority class `service_complaint`) | 0.154 | 0.030 |
| **Simple baseline** (rule-based weak labeler, `src/weak_label.py`) | **0.826** | **0.825** |
| Trained classifier (TF-IDF + LogisticRegression) | 0.774 | 0.773 |

Yes, the simple rule baseline beats the trained classifier on this golden
set. That is not a bug -- see Section D, it's the headline finding, and it's
real. Per-class breakdown for the trained classifier:

```
                             precision    recall  f1-score   support
              baggage_issue       0.85      0.74      0.79        23
      booking_checkin_issue       0.86      0.90      0.88        21
          flight_disruption       0.88      0.79      0.83        28
              info_question       0.65      0.50      0.56        22
              other_unclear       0.43      0.87      0.58        15
              praise_thanks       0.65      0.93      0.76        14
refund_compensation_request       1.00      0.83      0.91        24
          service_complaint       0.90      0.63      0.75        30
            upgrade_loyalty       0.85      0.94      0.89        18
```

### Escalation decision (positive class = "should escalate")

| System | Precision | Recall | F1 |
|---|---|---|---|
| Trivial baseline (always escalate) | 0.615 | 1.000 | 0.762 |
| Trivial baseline (never escalate) | 0.000 | 0.000 | 0.000 |
| **Rule engine** (`src/escalation.py`) | **0.805** | **0.550** | **0.653** |

The rule engine is precise (when it escalates, it's usually right) but misses
almost half of true escalations (54 of 120 golden "should escalate" cases).
See failure analysis.

### Reply quality

`eval/llm_judge.py` implements the LLM-as-judge rubric (grounded / relevant /
tone / actionable, 1-5 each) and an agreement check against a human
spot-check file. **This step requires a live `ANTHROPIC_API_KEY`** and was
not run end-to-end for this report (no key was available in the build
environment) -- the harness is real and tested for wiring/error-handling, but
the actual judge scores and human-agreement numbers are not yet populated.
This is flagged explicitly rather than faked; see "Next steps."

## C. Failure analysis: top 5 failure modes

**1. Escalation recall gap on "quiet" severity.** 54 golden escalate-cases
were missed. Common pattern: real grievances phrased without our red-flag
keywords -- e.g. *"horrible check in process at ORF... unorganized cluster"*
(no "refund"/"compensat"/anger-profanity match), or *"The best [sarcasm]
screen when you try to pull up your mobile boarding pass, fix your shit"*
(profanity regex missed it, since it used a euphemism-adjacent phrasing).
**Hypothesis:** the rule engine's red flags are keyword-literal; sarcasm,
mild-but-real complaints, and euphemized anger don't trip them. A learned
escalation classifier (trained on the golden set once it's larger) would
likely catch more of this, at some cost to precision/auditability.

**2. Sarcasm and rhetorical questions confuse both intent and escalation.**
*"thanks to supervisor Anne RRL QST-not. Such a helpless feeling..."* --
gold=`other_unclear` (genuinely hard to parse), predicted=`praise_thanks`
(conf 0.86) because of the literal word "thanks." *"This is not a helpful
reply. What update can you provide..."* -- gold=`flight_disruption`,
predicted=`info_question` because it ends in a literal question mark, which
is one of our info_question weak-label triggers. **Hypothesis:** bag-of-words
features have no way to represent negation or sarcasm scope; a keyword like
"thanks" or a trailing "?" is treated the same whether sincere or not.

**3. Compound/overlapping-topic messages force an arbitrary single label.**
*"how is it I made it to #Arkansas on a flight delayed 45 minutes but my bag
didn't [make it]"* is genuinely both `flight_disruption` and `baggage_issue`
-- gold picked `baggage_issue` (the actual unresolved ask), model predicted
`flight_disruption` (matched on "delayed"). **Hypothesis:** the taxonomy
assumes one dominant intent per message; real tweets often report two related
problems, and single-label classification has no way to express that. A
multi-label setup was considered and rejected as scope creep for v1 (decision
log), but this is exactly where it would help.

**4. Single-turn scoping loses continuation context.** *"(2) In premium
economy class if that makes a difference..."* and *"No I haven't. I only
spoke to them about my bags location."* are follow-up tweets in a longer
thread; read in isolation they're nearly uninterpretable, and both were
correctly bucketed `other_unclear` by a human but the classifier (also seeing
only the isolated text) sometimes disagrees. **Hypothesis:** this is a direct
cost of decision log #2 (single-turn scoping) -- multi-turn context would fix
this class of error directly, at the cost of the added complexity discussed
there.

**5. Non-English and heavily emoji/hashtag-only tweets.** e.g. a Spanish-
language complaint about discrimination was regex-invisible to English-only
rules and fell to `other_unclear`. **Hypothesis:** neither the weak labeler
nor the TF-IDF vectorizer (fit on English tokens) has any real signal for
non-English text; this is a silent, systematic blind spot rather than a
random error, and it's concerning specifically because discrimination
complaints landing here should be a high escalation-recall case, not a
low-confidence catch-all.

## D. What is misleading about my headline number

The single most important caveat in this whole report: **the simple rule
baseline "beating" the trained classifier (82.6% vs 77.4% accuracy) is
partly an artifact of how the golden set was sampled, not proof the rules
are actually better.**

The golden set is stratified by the *weak label* (decision log #11) --
i.e., for 8 of 9 buckets, an example only entered the golden set because the
rule-based labeler confidently matched a keyword pattern in it. That's
selecting-on-the-outcome-of-the-thing-you're-testing: within those 8 buckets,
the rule baseline is nearly guaranteed to often "agree with itself" in a way
the golden set is specifically shaped to capture. Restricting to the one
stratum that ISN'T selected this way (`other_unclear`, n=20, the bucket the
rules explicitly failed to classify going in), the gap nearly closes: simple
baseline 65.0% vs trained classifier 60.0% -- both worse, and much closer,
on a small sample. The honest interpretation is: **we don't yet have an
unbiased golden-set read on whether the trained classifier generalizes
better than the rules; the current golden set structurally favors the
rules**, and n=20 in the one unbiased stratum is too small to conclude much
either way.

A second, related caveat: the trained classifier is trained ON weak labels
produced by these same rules (decision log #5), so any gap it shows is
capturing "did the model learn to interpolate the rules' own keyword
patterns," not "did it learn something the rules don't know."

Third: the golden labels themselves are a single LLM pass (decision log #12),
not a reviewed human ground truth. Every number in this report inherits
whatever labeling errors or idiosyncratic judgment calls are in that pass.

## E. What we would do next with one more week

1. **Fix the golden-set sampling bias.** Draw a second, purely random
   (not weak-label-stratified) sample of ~150 examples specifically to get an
   unbiased read on simple-baseline vs trained-classifier accuracy. This is
   the single highest-priority fix -- Section D's finding is currently
   underpowered.
2. **Human review pass on all 195 (or 345, post-fix) golden labels** via
   `eval/label_cli.py`, ideally by 2 people with a documented disagreement-
   resolution process, so we can report inter-annotator agreement.
3. **Actually run `eval/llm_judge.py`** end-to-end with a real API key, plus
   a genuine ~20-example human spot-check, to get the LLM-judge agreement
   number the harness is built for but hasn't produced yet.
4. **Address failure mode #1** (escalation recall) by either loosening the
   red-flag patterns (precision/recall tradeoff, measurable now that we have
   a labeled confusion matrix) or training a small learned escalation model
   once the golden set is large enough to hold out a real validation split.
5. **Multi-label intent tagging** for compound messages (failure mode #3),
   at least as an experiment, since it directly targets a concrete, counted
   failure class rather than a hypothetical one.
6. **Extend thread reconstruction to 2-3 turns** to address failure mode #4,
   scoped carefully to avoid the label-ambiguity problem that made us punt on
   it originally (decision log #2).
