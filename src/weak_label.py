"""
Rule-based intent labeler.

This plays TWO roles in the project (both documented in the report):
  1. It is the SIMPLE BASELINE that the trained classifier must beat.
  2. It is the WEAK SUPERVISION source used to bootstrap training labels for
     the trained classifier, since we don't have human labels for all 36k pairs
     (only for the 200-example golden set, which is held out and never used
     for training -- see eval/build_golden_set.py).

Because the trained classifier is trained on labels produced by this same
function, any accuracy gain it shows over the rule baseline on OTHER
rule-labeled data is partly circular -- it mostly shows the model can
interpolate between keyword patterns, not that it's more accurate in any
deeper sense. The only trustworthy comparison is against the held-out human
golden set, which is why report/REPORT.md's "what's misleading about my
headline number" section leads with exactly this point.

Rules are ordered by priority (first match wins) based on manual reading of
~150 real AmericanAir customer messages during taxonomy design.
"""
import re

INTENTS = [
    "flight_disruption",
    "baggage_issue",
    "booking_checkin_issue",
    "refund_compensation_request",
    "upgrade_loyalty",
    "service_complaint",
    "info_question",
    "praise_thanks",
    "other_unclear",
]

# (intent, [regex patterns]) -- checked in this order, first match wins.
RULES = [
    ("refund_compensation_request", [
        r"\brefund\b", r"\breimburs", r"\bcompensat", r"\bwaive[d]?\b",
        r"\b(fee|charge)s?\s+(back|refunded|waived)\b", r"\bcredit\s+(back|voucher)\b",
        r"\bmoney\s+back\b", r"\bchange\s+fees?\b", r"\bextra\s+fee\b", r"\bwhy\s+.{0,15}charged\b",
    ]),
    ("baggage_issue", [
        r"\bbag(s|gage)?\b.{0,25}\b(lost|missing|delayed|damaged|late|no\s)",
        r"\bno\s+bags?\b", r"\bgate.?check", r"\bcarry.?on\b", r"\bsuitcase\b",
        r"\blost\s+(my\s+)?bag", r"\bbaggage\s+claim\b",
    ]),
    ("flight_disruption", [
        r"\bdelay(ed|s)?\b", r"\bcancel(led|lation)?\b", r"\bmissed?\s+(my\s+)?connection",
        r"\bdiverted?\b", r"\bstuck\s+(on\s+the\s+)?(plane|tarmac|gate)\b", r"\bno\s+(pilot|plane|crew)\b",
        r"\bgrounded\b", r"\brebook(ing)?\b", r"\banother\s+(hour|flight)\b",
        r"\b(held|waiting|sitting)\b.{0,20}\btarmac\b", r"\bgate\s+chang",
    ]),
    ("booking_checkin_issue", [
        r"\bcheck.?in\b", r"\bboarding\s+pass\b", r"\bseat\s+assignment\b",
        r"\bcan.?t\s+book\b", r"\breservation\b", r"\bbook(ing)?\s+(a\s+)?(flight|ticket)\b",
        r"\bwon.?t\s+recline\b", r"\bwrong\s+terminal\b", r"\bapp\s+(won.?t|isn.?t|not)\b",
    ]),
    ("upgrade_loyalty", [
        r"\bupgrade\b", r"\baadvantage\b", r"\b(ep|gold|platinum|executive)\s+(status|member)\b",
        r"\bmiles\b", r"\bfirst\s+class\b.{0,20}\b(upgrade|available)\b", r"\belite\s+status\b",
    ]),
    ("praise_thanks", [
        r"\bthank(s| you)\b", r"\bkudos\b", r"\bshout.?out\b", r"\bappreciate\b",
        r"\b(amazing|wonderful|great|awesome|lovely|fantastic)\s+(service|crew|flight|staff|attendant)\b",
        r"^thanks?\b", r"\bwell\s+done\b",
    ]),
    ("service_complaint", [
        r"\bworst\s+(customer\s+service|airline|experience)\b", r"\brude\b", r"\bdisgust",
        r"\bunacceptable\b", r"\bdiscriminat", r"\bno\s+remorse\b", r"\bhorrible\b",
        r"\bterrible\s+(service|experience)\b", r"f+u+c*k", r"\bnever\s+fly(ing)?\s+again\b",
        r"\blet\s+down\b", r"\bget\s+it\s+together\b", r"\bsuck(s)?\b", r"\bridiculous\b",
        r"\bpathetic\b", r"\bunethical\b", r"\bawful\b", r"\bshame\s+on\s+you\b",
        r"\bdisappoint(ed|ing)?\b", r"\bshould(n.?t)?\s+have\s+to\b",
    ]),
    ("info_question", [
        r"\?\s*$", r"\bany\s+(idea|chance|way)\b", r"\bhow\s+(do|can|long)\b",
        r"\bwhat\s+(time|gate|terminal)\b", r"\bis\s+there\b", r"\bcan\s+i\b", r"\bwill\s+(i|we)\b",
    ]),
]


def weak_label(text: str) -> str:
    t = text.lower()
    for intent, patterns in RULES:
        for pat in patterns:
            if re.search(pat, t):
                return intent
    return "other_unclear"


if __name__ == "__main__":
    import pandas as pd
    from collections import Counter

    df = pd.read_csv("data/processed/pairs_clean.csv")
    df["weak_intent"] = df["customer_clean"].apply(weak_label)
    counts = Counter(df["weak_intent"])
    total = len(df)
    for intent in INTENTS:
        c = counts.get(intent, 0)
        print(f"{intent:30s} {c:6d}  ({c/total:.1%})")
    df.to_csv("data/processed/pairs_weak_labeled.csv", index=False)
