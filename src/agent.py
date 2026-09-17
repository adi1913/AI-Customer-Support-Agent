"""
Step 6: The AI support agent itself.

Pipeline per incoming message:
  1. classify intent (TF-IDF + LogisticRegression, src/train_classifier.py)
  2. retrieve top-k historically resolved similar cases (src/retrieval.py)
  3. decide auto-handle vs escalate, with reasons (src/escalation.py)
  4. draft a reply grounded in the retrieved examples

Reply drafting has two modes:
  - LLM mode (if ANTHROPIC_API_KEY is set): calls Claude to synthesize a reply
    in AmericanAir's voice, grounded in the retrieved historical replies, and
    told to explicitly reference the retrieved facts rather than invent new ones.
  - Template mode (no API key / offline): fills a per-intent template using
    the single closest retrieved reply, adapted lightly for the current
    message. This keeps the FULL pipeline (classification + retrieval +
    escalation) runnable and evaluable in under 15 minutes with zero external
    dependencies -- only reply *quality* judging (eval/llm_judge.py) needs a key.
"""
import os
from dataclasses import dataclass, field

import joblib

from escalation import decide_escalation
from retrieval import ResolutionRetriever

TEMPLATE_BY_INTENT = {
    "flight_disruption": "We're sorry for the disruption to your flight, {name}. {retrieved_hint} We'll keep you posted on any updates.",
    "baggage_issue": "We're sorry about your baggage, {name}. {retrieved_hint} Please share your bag tag / claim number via DM so we can trace it.",
    "booking_checkin_issue": "Sorry for the trouble checking in, {name}. {retrieved_hint} Please DM your confirmation number and we'll help sort it out.",
    "refund_compensation_request": "Thanks for reaching out about this, {name}. {retrieved_hint} Please DM your confirmation number so our team can review your refund/compensation request.",
    "upgrade_loyalty": "Thanks for the question, {name}. {retrieved_hint} Please DM your AAdvantage number so we can look into your account.",
    "service_complaint": "We're sorry to hear about this experience, {name}. {retrieved_hint} We'd like to look into this further -- please DM us the flight details.",
    "info_question": "Thanks for reaching out, {name}. {retrieved_hint}",
    "praise_thanks": "Thank you so much for the kind words, {name}! We'll be sure to pass this along to the team.",
    "other_unclear": "Thanks for reaching out, {name}. Could you share a bit more detail (flight number, dates) so we can help?",
}


@dataclass
class AgentResponse:
    intent: str
    intent_confidence: float
    escalate: bool
    escalation_reasons: list = field(default_factory=list)
    retrieved_examples: list = field(default_factory=list)
    draft_reply: str = ""
    reply_mode: str = "template"


class SupportAgent:
    def __init__(self, model_path: str = "models/intent_clf.joblib",
                 pairs_csv: str = "data/processed/pairs_clean.csv"):
        self.clf = joblib.load(model_path)
        self.retriever = ResolutionRetriever(pairs_csv)
        self.use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _classify(self, text: str):
        probs = self.clf.predict_proba([text])[0]
        classes = self.clf.classes_
        top_idx = probs.argmax()
        return classes[top_idx], float(probs[top_idx])

    def _draft_template(self, text: str, intent: str, retrieved) -> str:
        hint = ""
        if len(retrieved):
            hint = f"(Similar past case was resolved: \"{retrieved[0]['agent_clean'][:120]}\")"
        template = TEMPLATE_BY_INTENT.get(intent, TEMPLATE_BY_INTENT["other_unclear"])
        return template.format(name="there", retrieved_hint=hint)

    def _draft_llm(self, text: str, intent: str, retrieved) -> str:
        import anthropic
        client = anthropic.Anthropic()
        examples_block = "\n".join(
            f"- Customer: {r['customer_clean']}\n  Agent replied: {r['agent_clean']}"
            for r in retrieved
        )
        prompt = f"""You are drafting a tweet reply as AmericanAir's customer support account.
Customer's new message (intent: {intent}):
"{text}"

Here are similar past cases AmericanAir has already resolved, for grounding
(match this tone and level of specificity -- don't invent facts not present
in the customer's message or these examples):
{examples_block}

Write ONLY the reply tweet text (no preamble), under 280 characters, in
AmericanAir's typical brief/empathetic support voice."""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def handle(self, text: str) -> AgentResponse:
        intent, confidence = self._classify(text)
        esc = decide_escalation(text, intent, confidence)
        retrieved_df = self.retriever.retrieve(text, intent, k=3)
        retrieved = retrieved_df.to_dict(orient="records")

        if self.use_llm:
            try:
                reply = self._draft_llm(text, intent, retrieved)
                mode = "llm"
            except Exception as e:
                reply = self._draft_template(text, intent, retrieved)
                mode = f"template_fallback (llm_error: {e})"
        else:
            reply = self._draft_template(text, intent, retrieved)
            mode = "template"

        return AgentResponse(
            intent=intent,
            intent_confidence=confidence,
            escalate=esc["escalate"],
            escalation_reasons=esc["reasons"],
            retrieved_examples=retrieved,
            draft_reply=reply,
            reply_mode=mode,
        )


if __name__ == "__main__":
    agent = SupportAgent()
    tests = [
        "Delayed 4hrs coming back from O'hare. No real explanation.",
        "My bag never showed up, it's been an hour and no one will help.",
        "Thanks for the great crew today!",
        "this is discrimination and I will get a lawyer",
    ]
    for t in tests:
        r = agent.handle(t)
        print(f"\n> {t}")
        print(f"  intent={r.intent} (conf={r.intent_confidence:.2f})  escalate={r.escalate} {r.escalation_reasons}")
        print(f"  reply [{r.reply_mode}]: {r.draft_reply}")
