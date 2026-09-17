"""
Step 5: Escalation decision engine.

Deterministic, rule-based, and DELIBERATELY not an LLM call: escalation is a
safety-relevant decision (routes real customer harm/legal/retention risk to a
human), so we want it auditable and debuggable via literal reasons, not a
black-box judgment. See decision log for why this is rule-based while intent
classification and reply drafting are ML/LLM-based.

Policy mirrors the one used to hand-label the golden set (eval/gold_labels.py)
so the automated escalation metric is measuring "does the system reproduce the
policy we designed", not "does it guess an unstated policy".
"""
import re

# Intents that never need a human unless a red-flag pattern also fires.
LOW_RISK_INTENTS = {"praise_thanks"}

RED_FLAGS = [
    ("safety_or_medical", r"\b(medical|emergency|security|weapon|gun|unsafe|airworth)\w*\b"),
    ("discrimination_or_legal", r"\b(discriminat\w*|lawyer|legal action|sue|lawsuit|threat(en)?)\b"),
    ("repeated_failure", r"\b(again|3rd time|third time|multiple times|every time|every single)\b"),
    ("premium_customer_risk", r"\b(platinum|executive platinum|concierge key|elite|gold status|\d+\s*(year|yr)s?\s+(loyal|flying|member))\b"),
    ("strong_anger", r"f+u+c*k|\bworst\b.{0,20}\b(ever|airline|service)\b|\bunacceptable\b|\bdisgust"),
    ("explicit_compensation_or_refund", r"\brefund\b|\bcompensat\w*|\breimburs\w*|\bvoucher\b"),
    ("account_specific_dispute", r"\bupgrade list\b|\bmiles\b.{0,15}\b(wrong|missing|not credited)\b|\bpnr\b"),
]

ESCALATE_INTENTS_DEFAULT = {
    # intents that escalate by default unless they look routine (handled in code)
    "service_complaint", "refund_compensation_request",
}

LOW_CONFIDENCE_THRESHOLD = 0.45


def decide_escalation(text: str, intent: str, classifier_confidence: float) -> dict:
    t = text.lower()
    reasons = []

    for flag_name, pattern in RED_FLAGS:
        if re.search(pattern, t):
            reasons.append(flag_name)

    if intent in ESCALATE_INTENTS_DEFAULT and not reasons:
        reasons.append(f"intent_default_escalate:{intent}")

    if classifier_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append(f"low_classifier_confidence:{classifier_confidence:.2f}")

    if intent in LOW_RISK_INTENTS and not reasons:
        return {"escalate": False, "reasons": ["low_risk_intent:praise_thanks"]}

    if reasons:
        return {"escalate": True, "reasons": reasons}
    return {"escalate": False, "reasons": ["no_red_flags_routine_intent"]}


if __name__ == "__main__":
    tests = [
        ("Thanks for the great crew today!", "praise_thanks", 0.9),
        ("Delayed 4hrs coming back from O'hare. No real explanation.", "flight_disruption", 0.8),
        ("I need a refund, this is the third time you've cancelled on me.", "refund_compensation_request", 0.7),
        ("How do I check in online?", "booking_checkin_issue", 0.9),
        ("this is discrimination and I will get a lawyer", "service_complaint", 0.6),
    ]
    for text, intent, conf in tests:
        result = decide_escalation(text, intent, conf)
        print(f"{result['escalate']!s:5s} {result['reasons']}  <- {text[:60]}")
