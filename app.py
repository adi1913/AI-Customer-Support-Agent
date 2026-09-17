import sys
import streamlit as st

# Make src/ importable
sys.path.insert(0, "src")

from agent import SupportAgent


st.set_page_config(
    page_title="AmericanAir AI Support Agent",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def load_agent():
    return SupportAgent()


agent = load_agent()


st.title("✈️ AmericanAir AI Support Agent")
st.caption(
    "AI-powered customer support: intent classification, "
    "historical resolution retrieval, escalation, and reply drafting."
)

st.divider()

# Customer input
st.subheader("Customer Message")

message = st.text_area(
    "Enter a customer support message:",
    placeholder=(
        "Example: My flight was delayed for 4 hours and nobody "
        "explained what happened."
    ),
    height=120,
)

if st.button("🔍 Analyze Message", type="primary", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Analyzing customer message..."):
        result = agent.handle(message.strip())

    st.divider()

    # -------------------------
    # Main analysis
    # -------------------------
    st.subheader("🤖 AI Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Intent",
            result.intent.replace("_", " ").title()
        )

    with col2:
        st.metric(
            "Confidence",
            f"{result.intent_confidence:.0%}"
        )

    with col3:
        if result.escalate:
            st.error("🚨 ESCALATE")
        else:
            st.success("✅ AUTO-HANDLE")

    # -------------------------
    # Escalation reason
    # -------------------------
    st.subheader("Decision")

    if result.escalate:
        st.error(
            "This message should be reviewed by a human support agent."
        )
    else:
        st.success(
            "This message can be handled automatically."
        )

    if result.escalation_reasons:
        st.write("**Reason:**")
        for reason in result.escalation_reasons:
            st.write(f"- `{reason}`")

    # -------------------------
    # Retrieved cases
    # -------------------------
    st.subheader("🔎 Similar Historical Cases")

    if result.retrieved_examples:

        for i, example in enumerate(result.retrieved_examples, 1):

            with st.expander(f"Historical Case #{i}"):

                st.markdown("**Customer:**")
                st.write(example.get("customer_clean", ""))

                st.markdown("**Previous Agent Resolution:**")
                st.write(example.get("agent_clean", ""))

    else:
        st.info("No similar historical cases were retrieved.")

    # -------------------------
    # Suggested reply
    # -------------------------
    st.subheader("💬 Suggested Support Reply")

    st.info(result.draft_reply)

    st.caption(f"Reply generation mode: `{result.reply_mode}`")

    # -------------------------
    # Technical details
    # -------------------------
    with st.expander("Technical Details"):

        st.write({
            "intent": result.intent,
            "confidence": result.intent_confidence,
            "escalate": result.escalate,
            "escalation_reasons": result.escalation_reasons,
            "reply_mode": result.reply_mode,
            "retrieved_cases": len(result.retrieved_examples),
        })


st.divider()

st.caption(
    "Prototype built using historical AmericanAir customer-support "
    "conversations. The system is designed to ground responses in "
    "previously resolved cases and escalate selected risky requests."
)