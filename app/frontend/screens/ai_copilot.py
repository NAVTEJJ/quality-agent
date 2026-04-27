"""
AI Copilot screen — conversational chat interface backed by QualityAgent.
Phase 4 Step 1: shell only (full wiring in Phase 4 Step 2).
"""
import streamlit as st

from app.frontend.theme import COLORS


_DEMO_QUESTIONS = [
    "What is the risk level of lot L-778 and what should I do?",
    "Are there any process drift issues on our production lines?",
    "Compare SUP-A and SUP-B for a safety-critical program",
    "Why did serial SR20260008 fail in the field?",
    "What are the top 3 quality risks I should know about right now?",
]


def _submit_question(question: str):
    """Push a user question through the agent and append both turns to history."""
    if not question.strip():
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    agent = st.session_state.get("agent")
    if agent is None:
        answer = (
            "Claude API key not configured. "
            "Set ANTHROPIC_API_KEY and restart the app to enable AI responses."
        )
        suggestions: list[str] = []
    else:
        with st.spinner("Thinking..."):
            try:
                result = agent.ask(
                    question,
                    session_id=st.session_state.session_id,
                )
                answer = result.response_text
                suggestions = result.follow_up_suggestions or []
                # Update entity focus for other screens.
                entities = result.intent.entities if result.intent else {}
                if entities.get("lot_no"):
                    st.session_state.current_lot = entities["lot_no"][0]
                if entities.get("supplier"):
                    st.session_state.current_supplier = entities["supplier"][0]
            except Exception as exc:
                answer = f"Agent error: {exc}"
                suggestions = []

    st.session_state.chat_history.append({
        "role":        "agent",
        "content":     answer,
        "suggestions": suggestions,
    })


def render():
    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
          <h2 style="color:{COLORS['text_primary']};margin:0 0 4px;">AI Quality Copilot</h2>
          <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0;">
            Ask anything about lots, suppliers, process drift, or warranty failures.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Auto-submit pending question from sidebar alerts ──────────────────
    pending = st.session_state.pop("pending_question", None)
    if pending:
        _submit_question(pending)

    # ── Suggested starter questions (shown only on empty history) ─────────
    if not st.session_state.chat_history:
        st.markdown(
            f'<div class="section-header">Suggested questions</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(_DEMO_QUESTIONS))
        for col, q in zip(cols, _DEMO_QUESTIONS):
            with col:
                short = q[:42] + "..." if len(q) > 45 else q
                if st.button(short, key=f"starter_{q[:20]}", use_container_width=True):
                    _submit_question(q)
                    st.rerun()

        st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{turn["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-agent">{turn["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Render follow-up suggestion chips.
            suggestions = turn.get("suggestions", [])
            if suggestions:
                chip_html = " ".join(
                    f'<span class="suggestion-chip">{s}</span>' for s in suggestions
                )
                st.markdown(chip_html, unsafe_allow_html=True)
                for s in suggestions:
                    if st.button(s, key=f"chip_{hash(s)}", use_container_width=False):
                        _submit_question(s)
                        st.rerun()

    # ── Input bar ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="question",
                label_visibility="collapsed",
                placeholder="Ask about a lot, supplier, line, or field failure...",
                key="chat_input",
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input:
        _submit_question(user_input)
        st.rerun()

    # ── Clear chat ────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.current_lot = None
            st.session_state.current_supplier = None
            st.rerun()
