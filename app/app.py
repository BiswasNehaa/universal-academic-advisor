# =============================================================================
# app.py  —  Streamlit UI for the AI Academic Advisor
#
# HOW TO RUN:
#   1. Make sure ingest.py has been run at least once to build faiss_index/
#   2. pip install streamlit
#   3. streamlit run app.py
#
# FILE STRUCTURE EXPECTED:
#   app.py              ← this file (place alongside advisor.py)
#   advisor.py
#   ingest.py
#   faiss_index/
#   ../Data/courses.json
#   ../Data/career.json
#   .env                ← must contain GROQ_API_KEY=...
# =============================================================================

import streamlit as st
import json
import re

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Academic Advisor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Lazy-load the heavy advisor module so Streamlit doesn't crash on import ───
@st.cache_resource(show_spinner="Loading AI models… (first run takes ~30 sec)")
def load_advisor_resources():
    """
    Import and return the advisor module's shared resources.
    Cached so models are only loaded once per server session.
    """
    import advisor  # noqa: F401  — triggers module-level model loading
    return advisor

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten up default Streamlit padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Card-style containers — use rgba so they work in both light & dark mode */
    .result-card {
        background: rgba(79, 142, 247, 0.12);
        border-left: 4px solid #4f8ef7;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        color: inherit;
    }
    .roadmap-card {
        background: rgba(52, 199, 112, 0.12);
        border-left: 4px solid #34c770;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        color: inherit;
    }
    .warn-card {
        background: rgba(245, 166, 35, 0.15);
        border-left: 4px solid #f5a623;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        color: inherit;
    }
    /* All text inside cards inherits the theme colour (white in dark, dark in light) */
    .result-card  * { color: inherit; }
    .roadmap-card * { color: inherit; }
    .warn-card    * { color: inherit; }

    .course-id   { font-weight: 700; font-size: 0.95rem; opacity: 0.95; }
    .course-name { font-size: 1.05rem; font-weight: 600; }
    .why-text    { font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.85; }
    .credits-badge {
        display: inline-block;
        background: rgba(79, 142, 247, 0.25);
        border-radius: 12px;
        padding: 1px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR — Student profile (persisted in session_state)
# =============================================================================
with st.sidebar:
    st.title("🎓 Academic Advisor")
    st.caption("Powered by RAG + LLaMA 3.3 70B")
    st.divider()

    st.subheader("Your Profile")

    career_goal = st.text_input(
        "Career Goal",
        placeholder="e.g. AI Engineer, Data Scientist",
        help="Type your target career. The advisor will match it to a skills map.",
    )

    completed_input = st.text_area(
        "Completed Courses",
        placeholder="Comma-separated: Introduction to Python, BCS301, Math-I",
        height=110,
        help="Course names or codes. Partial names and abbreviations are fine.",
    )

    credit_limit = st.slider(
        "Max credits this semester",
        min_value=1, max_value=30, value=16, step=1,
    )

    # ── Mid-session: add forgotten courses ───────────────────────────────────
    st.divider()
    st.subheader("Forgot a course?")
    extra_courses = st.text_input(
        "Add more completed courses",
        placeholder="e.g. BCS302, Operating Systems",
        key="extra_input",
    )
    if st.button("➕ Add courses", use_container_width=True):
        if extra_courses.strip():
            new_raw = [c.strip().upper() for c in extra_courses.split(",") if c.strip()]
            advisor = load_advisor_resources()
            with st.spinner("Enriching new entries…"):
                new_enriched = advisor.enrich_completed_list(new_raw)
            existing = st.session_state.get("enriched_completed", [])
            merged   = list(set(existing + new_enriched))
            st.session_state["enriched_completed"] = merged
            st.success(f"Added {len(new_enriched)} course(s). Total: {len(merged)}")
        else:
            st.warning("Please enter at least one course.")

    st.divider()
    if st.button("🔄 Reset session", use_container_width=True):
        for key in ["enriched_completed", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()


# =============================================================================
# HELPERS — Parse and render the LLM's JSON response
# =============================================================================

def render_response(raw: str):
    """Parse JSON roadmap from the LLM and render it with styled cards."""
    clean = re.sub(r'```json|```', '', raw).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        st.error("The model returned an unexpected format. Raw output:")
        st.code(raw)
        return

    # ── Encouraging message ───────────────────────────────────────────────────
    msg = data.get("message", "")
    if msg:
        st.info(f"💬 {msg}")

    # ── Recommended courses ───────────────────────────────────────────────────
    enroll_now = data.get("enroll_now", [])
    if enroll_now:
        st.subheader("✅ Recommended to Enroll")
        for c in enroll_now:
            cid     = c.get("course_id", "?")
            cname   = c.get("course_name", "?")
            credits = c.get("credits", "?")
            why     = c.get("why", "")
            st.markdown(
                f"""<div class="result-card">
                    <span class="course-id">{cid}</span>
                    <span class="credits-badge">{credits} cr</span><br>
                    <span class="course-name">{cname}</span>
                    <div class="why-text">📌 {why}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="warn-card">⚠️ No eligible courses yet. '
            'Complete the roadmap steps below to unlock your path.</div>',
            unsafe_allow_html=True,
        )

    # ── Roadmap chain ─────────────────────────────────────────────────────────
    unlock_next = data.get("unlock_next", [])
    if unlock_next:
        st.subheader("🗺️ Your Roadmap")
        for step in unlock_next:
            first   = step.get("complete_first", "?")
            unlocks = step.get("this_will_unlock", "?")
            further = step.get("which_then_unlocks", "")
            further_html = f"<br>➡️ <em>then unlocks:</em> <strong>{further}</strong>" if further else ""
            st.markdown(
                f"""<div class="roadmap-card">
                    <strong>Finish:</strong> {first}<br>
                    <strong>Unlocks:</strong> {unlocks}{further_html}
                </div>""",
                unsafe_allow_html=True,
            )


# =============================================================================
# MAIN AREA — Query input + chat history
# =============================================================================

st.header("Ask your Advisor")

# ── Initialise session state ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []   # list of (query, raw_response)
if "enriched_completed" not in st.session_state:
    st.session_state["enriched_completed"] = []

# ── Query form ────────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=True):
    query = st.text_input(
        "Your question",
        placeholder="What should I study next for AI? / Which courses unlock Machine Learning?",
    )
    submitted = st.form_submit_button("Ask ✦", use_container_width=True)

if submitted:
    # ── Validate profile ──────────────────────────────────────────────────────
    if not career_goal.strip():
        st.warning("Please enter a Career Goal in the sidebar first.")
        st.stop()
    if not query.strip():
        st.warning("Please type a question.")
        st.stop()

    advisor = load_advisor_resources()

    # ── Enrich completed courses (once per unique input) ─────────────────────
    raw_completed = [c.strip().upper() for c in completed_input.split(",") if c.strip()]
    if raw_completed and not st.session_state["enriched_completed"]:
        with st.spinner("Enriching course history…"):
            st.session_state["enriched_completed"] = advisor.enrich_completed_list(raw_completed)

    completed = st.session_state["enriched_completed"]

    # ── Run the advisor pipeline ──────────────────────────────────────────────
    with st.spinner("Thinking… 🤔"):
        try:
            career_keywords = advisor.get_career_keywords(career_goal)
            candidate_docs  = advisor.retrieve_candidate_docs(query, career_goal, career_keywords)
            completed_upper = [c.upper() for c in completed]
            eligible_pool, excluded = advisor.filter_candidates(
                candidate_docs, completed_upper, career_keywords, credit_limit
            )
            raw_response = advisor.build_llm_response(
                eligible_pool, excluded, career_goal, career_keywords
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # ── Store in history and display ──────────────────────────────────────────
    st.session_state["chat_history"].append((query, raw_response))


# ── Render history (most recent first) ───────────────────────────────────────
if st.session_state["chat_history"]:
    for i, (q, resp) in enumerate(reversed(st.session_state["chat_history"])):
        turn = len(st.session_state["chat_history"]) - i
        with st.expander(f"Q{turn}: {q}", expanded=(i == 0)):
            render_response(resp)
else:
    st.markdown(
        """
        **How to get started:**
        1. Fill in your **Career Goal** and **Completed Courses** in the sidebar
        2. Set your **credit limit** for the semester
        3. Type a question above — e.g. *"What should I study next?"*
        """,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Academic Advisor · RAG pipeline · LLaMA 3.3 70B via Groq · FAISS + HuggingFace embeddings")
