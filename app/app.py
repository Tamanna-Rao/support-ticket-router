"""
app.py - Streamlit app for the Support Ticket Router (v2: multi-tab UI).

Run locally from the project root:
    streamlit run app/app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                                # router.py next to app.py (e.g. on Hugging Face)
sys.path.append(os.path.join(HERE, "..", "src"))     # router.py in src/ (local project)
from router import (  # noqa: E402
    AUTO_THRESHOLD,
    REVIEW_THRESHOLD,
    TicketRouter,
    build_queue,
    routing_decision,
)

GITHUB_URL = "https://github.com/Tamanna-Rao/support-ticket-router"
MODEL_URL = "https://huggingface.co/trishurao/support-ticket-classifier"

st.set_page_config(page_title="Support Ticket Router", page_icon="🎫", layout="wide")

# ------------------------------------------------------------------ results of our experiments
# (copied from results/comparison_table.md and results/routing_analysis.md; update if you retrain)
MODEL_ROWS = [
    ("TF-IDF + LogReg (preprocessed)", 0.8727, 0.8729),
    ("TF-IDF + LogReg (raw text)", 0.8938, 0.8942),
    ("BiLSTM (random embeddings)", 0.8792, 0.8793),
    ("BiLSTM + GloVe", 0.8990, 0.8997),
    ("DistilBERT (fine-tuned)", 0.9263, 0.9262),
]
ROUTING_ROWS = [
    ("auto-route (confidence >= 0.90)", "90.3%", 0.968, 0.988),
    ("route + needs review (0.50 to 0.90)", "8.0%", 0.576, 0.865),
    ("human triage (below 0.50)", "1.8%", 0.389, 0.593),
]
EXAMPLES = [
    "I was charged twice for my order and nobody is replying. This is unacceptable!",
    "My card was stolen yesterday, please block it immediately!",
    "How do I change my PIN?",
    "Why is my transfer still pending?",
    "My top up failed but the money left my account.",
    "What exchange rate do you use for payments abroad?",
    "asdf not working",
]
MANUAL_MIN = 2.0   # assumption: minutes a human needs to triage one ticket
REVIEW_MIN = 0.5   # assumption: minutes to check a ticket flagged "needs review"
MAX_BATCH = 200

# ------------------------------------------------------------------ styling
st.markdown(
    """<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%); padding: 2.2rem 2.4rem; border-radius: 18px; margin-bottom: 1.3rem;}
.hero h1 {color: #FFFFFF; margin: 0 0 .5rem 0; font-size: 2.2rem; line-height: 1.2;}
.hero p {color: rgba(255,255,255,.93); margin: 0; font-size: 1.08rem;}
.card {background: rgba(99,102,241,.09); border: 1px solid rgba(99,102,241,.28); border-radius: 14px; padding: 1rem 1.2rem; height: 100%;}
.card .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; opacity: .7;}
.card .value {font-size: 1.65rem; font-weight: 700; line-height: 1.3; margin: .15rem 0;}
.card .sub {font-size: .85rem; opacity: .75;}
.badge {display: inline-block; padding: .25rem .85rem; border-radius: 999px; font-weight: 700; font-size: .95rem; color: #FFFFFF;}
.b-Critical {background: #DC2626;}
.b-High {background: #EA580C;}
.b-Medium {background: #D97706;}
.b-Low {background: #16A34A;}
.step-num {display: inline-block; width: 1.9rem; height: 1.9rem; line-height: 1.9rem; text-align: center; border-radius: 50%; background: #4F46E5; color: #FFFFFF; font-weight: 700; margin-bottom: .4rem;}
div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {background: linear-gradient(120deg, #4F46E5, #7C3AED); border: none; color: #FFFFFF; font-weight: 600;}
button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>""",
    unsafe_allow_html=True,
)


def card(label, value, sub=""):
    """One-line HTML (no blank lines or indentation, so Markdown does not break it)."""
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def badge(level):
    return f'<span class="badge b-{level}">{level}</span>'


def find_asset(name):
    for folder in (os.path.join(HERE, "assets"), os.path.join(HERE, "..", "results"), HERE):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def read_text(name):
    path = find_asset(name)
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def default_thresholds():
    """Use the thresholds picked by threshold_analysis.py if the file is available."""
    path = find_asset("thresholds.json")
    if path:
        try:
            with open(path) as f:
                d = json.load(f)
            return float(d["auto"]), float(d["review"])
        except Exception:
            pass
    return AUTO_THRESHOLD, REVIEW_THRESHOLD


@st.cache_resource(show_spinner="Loading the model (first time takes a minute)...")
def load_router():
    return TicketRouter()


for key, default in (("history", []), ("last", None), ("batch", None)):
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------------------ sidebar
auto_default, review_default = default_thresholds()
st.sidebar.markdown("## 🎫 Ticket Router")
st.sidebar.header("Routing settings")
auto_t = st.sidebar.slider("Auto-route at confidence >=", 0.50, 0.99, float(auto_default), 0.01)
review_t = st.sidebar.slider("Human triage below", 0.10, 0.90, float(review_default), 0.01)
if review_t > auto_t:
    st.sidebar.warning("The triage threshold should not be higher than the auto-route threshold.")
st.sidebar.caption("Tickets between the two thresholds are routed but flagged for review. "
                   "Defaults come from the validation-set analysis.")
st.sidebar.divider()
st.sidebar.markdown(f"[GitHub repository]({GITHUB_URL})  \n[Model on Hugging Face]({MODEL_URL})")
st.sidebar.caption("Model: DistilBERT fine-tuned on Banking77")


def record(ticket, raw):
    decision = routing_decision(raw["confidence"], auto_t, review_t)
    return {
        "ticket": ticket,
        "intent": raw["intent"],
        "confidence": round(raw["confidence"], 3),
        "team": raw["team"],
        "urgency": raw["urgency"],
        "sentiment": raw["sentiment"],
        "decision": decision,
        "queue": build_queue(raw["team"], raw["urgency"], decision),
    }


tab_home, tab_try, tab_batch, tab_dash, tab_perf, tab_about = st.tabs(
    ["Home", "Classify a ticket", "Batch", "Dashboard", "Model performance", "About"]
)

# ------------------------------------------------------------------ Home
with tab_home:
    st.markdown(
        '<div class="hero"><h1>Customer Support Ticket Router</h1>'
        '<p>Reads an incoming ticket, finds the intent, estimates urgency and sentiment, '
        'and sends it to the right team, or to a human when the model is not sure.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Intent accuracy", "92.6%", "DistilBERT, 3,080 test messages"), unsafe_allow_html=True)
    c2.markdown(card("Team accuracy", "97.2%", "ticket reaches the right team"), unsafe_allow_html=True)
    c3.markdown(card("Auto-routed", "90.3%", "at confidence 0.90, 96.8% correct"), unsafe_allow_html=True)
    c4.markdown(card("Intents", "77", "Banking77 dataset"), unsafe_allow_html=True)

    st.markdown("### How it works")
    steps = [
        ("1", "Read", "The ticket text goes into a DistilBERT model fine-tuned on 77 banking intents."),
        ("2", "Analyse", "VADER scores the sentiment. Simple rules estimate urgency and explain why."),
        ("3", "Decide", "High confidence: auto-route. Medium: route with a review flag. Low: human triage."),
        ("4", "Route", "The intent maps to a team such as Fraud & Security, Cards or Transfers."),
    ]
    cols = st.columns(4)
    for col, (num, title, text) in zip(cols, steps):
        col.markdown(f'<div class="card"><div class="step-num">{num}</div>'
                     f'<div class="value" style="font-size:1.2rem">{title}</div>'
                     f'<div class="sub">{text}</div></div>', unsafe_allow_html=True)

    st.markdown("### Try it")
    st.write("Open **Classify a ticket** for one message, **Batch** for many at once, or **Dashboard** to see "
             "your session. **Model performance** shows how the models compare.")

# ------------------------------------------------------------------ Classify
with tab_try:
    st.subheader("Classify a single ticket")
    choice = st.selectbox("Load an example", ["(write your own)"] + EXAMPLES)
    start_text = "" if choice.startswith("(") else choice
    ticket = st.text_area("Customer ticket", value=start_text, height=110,
                          placeholder="Type or paste a customer message...")

    if st.button("Classify ticket", type="primary"):
        if not ticket.strip():
            st.warning("Please enter a ticket first.")
        else:
            raw = load_router().predict(ticket.strip())
            st.session_state.last = {"ticket": ticket.strip(), "raw": raw}
            st.session_state.history.append(record(ticket.strip(), raw))

    last = st.session_state.last
    if last:
        raw = last["raw"]
        rec = record(last["ticket"], raw)  # thresholds applied here, so the sliders update the result live
        if rec["decision"] == "auto-route":
            st.success(f"Auto-routed to: {rec['queue']}")
        elif rec["decision"] == "route + needs review":
            st.warning(f"Routed with a review flag: {rec['queue']}")
        else:
            st.error("Low confidence: sent to the human triage queue")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(card("Predicted team", raw["team"], "from the intent"), unsafe_allow_html=True)
        c2.markdown(card("Urgency", badge(raw["urgency"]), "rule-based"), unsafe_allow_html=True)
        c3.markdown(card("Sentiment", raw["sentiment"], f"score {raw['sentiment_score']:+.2f}"),
                    unsafe_allow_html=True)
        c4.markdown(card("Confidence", f"{raw['confidence']:.1%}", "model probability"),
                    unsafe_allow_html=True)

        st.write("")
        st.write(f"**Predicted intent:** `{raw['intent']}`")
        st.progress(min(max(raw["confidence"], 0.0), 1.0), text=f"Model confidence: {raw['confidence']:.1%}")

        left, right = st.columns(2)
        with left:
            st.write("**Top 3 intents**")
            df = pd.DataFrame(raw["top3"], columns=["intent", "probability"]).set_index("intent")
            st.bar_chart(df)
        with right:
            st.write("**Why this urgency?**")
            if raw["urgency_reasons"]:
                for reason in raw["urgency_reasons"]:
                    st.write(f"- {reason}")
            else:
                st.write("- no urgency signals found")

# ------------------------------------------------------------------ Batch
with tab_batch:
    st.subheader("Route many tickets at once")
    mode = st.radio("Input", ["Paste tickets (one per line)", "Upload a CSV file"], horizontal=True)
    tickets = []
    if mode.startswith("Paste"):
        pasted = st.text_area("Tickets", height=180, placeholder="One ticket per line...")
        tickets = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
    else:
        uploaded = st.file_uploader("CSV with a 'ticket' column (or the first column is used)", type=["csv"])
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                column = "ticket" if "ticket" in df_in.columns else df_in.columns[0]
                tickets = df_in[column].dropna().astype(str).tolist()
                st.caption(f"Using column '{column}' ({len(tickets)} rows).")
            except Exception as e:
                st.error(f"Could not read the file: {e}")

    if len(tickets) > MAX_BATCH:
        st.warning(f"Only the first {MAX_BATCH} tickets will be processed in this demo.")
        tickets = tickets[:MAX_BATCH]

    if st.button("Process tickets", type="primary"):
        if not tickets:
            st.warning("Add some tickets first.")
        else:
            router = load_router()
            bar = st.progress(0.0, text="Processing...")
            rows = []
            for i, t in enumerate(tickets, 1):
                rows.append(record(t, router.predict(t)))
                bar.progress(i / len(tickets), text=f"Processing {i}/{len(tickets)}")
            bar.empty()
            st.session_state.batch = pd.DataFrame(rows)
            st.session_state.history.extend(rows)

    result = st.session_state.batch
    if result is not None and len(result):
        n = len(result)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(card("Tickets", n), unsafe_allow_html=True)
        m2.markdown(card("Auto-routed", f"{(result['decision'] == 'auto-route').mean():.0%}"),
                    unsafe_allow_html=True)
        m3.markdown(card("Human triage", f"{(result['decision'] == 'human triage').mean():.0%}"),
                    unsafe_allow_html=True)
        m4.markdown(card("High or critical", f"{result['urgency'].isin(['High', 'Critical']).sum()}"),
                    unsafe_allow_html=True)
        st.write("")
        st.dataframe(result)
        st.download_button("Download results (CSV)", result.to_csv(index=False).encode("utf-8"),
                           "routed_tickets.csv", "text/csv")

# ------------------------------------------------------------------ Dashboard
with tab_dash:
    st.subheader("Session dashboard")
    hist = st.session_state.history
    if not hist:
        st.info("Classify a few tickets (single or batch) and they will appear here.")
    else:
        df = pd.DataFrame(hist)
        n = len(df)
        n_auto = int((df["decision"] == "auto-route").sum())
        n_review = int((df["decision"] == "route + needs review").sum())
        saved = n_auto * MANUAL_MIN + n_review * (MANUAL_MIN - REVIEW_MIN)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(card("Tickets processed", n), unsafe_allow_html=True)
        m2.markdown(card("Auto-routed", f"{n_auto / n:.0%}"), unsafe_allow_html=True)
        m3.markdown(card("High or critical", f"{int(df['urgency'].isin(['High', 'Critical']).sum())}"),
                    unsafe_allow_html=True)
        m4.markdown(card("Triage time saved", f"{saved:.0f} min", "estimate"), unsafe_allow_html=True)
        st.caption(f"Time saved assumes {MANUAL_MIN:g} min of manual triage per ticket and "
                   f"{REVIEW_MIN:g} min to check a flagged one. It is an estimate, not a measurement.")

        a, b, c = st.columns(3)
        with a:
            st.write("**Tickets per team**")
            st.bar_chart(df["team"].value_counts())
        with b:
            st.write("**Urgency levels**")
            st.bar_chart(df["urgency"].value_counts())
        with c:
            st.write("**Routing decisions**")
            st.bar_chart(df["decision"].value_counts())

        st.write("**History**")
        st.dataframe(df)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last = None
            st.session_state.batch = None
            st.rerun()

# ------------------------------------------------------------------ Model performance
with tab_perf:
    st.subheader("How the models compare")
    perf = pd.DataFrame(MODEL_ROWS, columns=["Model", "Accuracy", "Macro F1"])
    left, right = st.columns([3, 2])
    with left:
        st.bar_chart(perf.set_index("Model")["Macro F1"])
    with right:
        st.table(perf.set_index("Model"))
    st.caption("Banking77 test set (3,080 messages, 77 intents). One training run per model.")

    st.subheader("Confidence-based routing (test set)")
    routing = pd.DataFrame(ROUTING_ROWS, columns=["Decision", "Share of tickets", "Intent accuracy", "Team accuracy"])
    st.table(routing.set_index("Decision"))
    st.caption("Thresholds were chosen on a validation set, then checked on the test set. "
               "Team accuracy is higher than intent accuracy because similar intents usually belong to the same team.")

    analysis = read_text("routing_analysis.md")
    if analysis:
        with st.expander("Full routing analysis and cost assumptions"):
            st.markdown(analysis)

    errors = read_text("error_analysis.md")
    if errors:
        with st.expander("Error analysis: where DistilBERT gets confused"):
            st.markdown(errors)

    images = [("model_comparison.png", "Model comparison"),
              ("threshold_tradeoff.png", "Automation vs accuracy"),
              ("distilbert_confusion_matrix.png", "Confusion matrix")]
    available = [(find_asset(f), cap) for f, cap in images if find_asset(f)]
    if available:
        st.subheader("Charts")
        for col, (path, cap) in zip(st.columns(len(available)), available):
            col.image(path, caption=cap)

# ------------------------------------------------------------------ About
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
**Goal:** automate the first step of customer support. Read a ticket, work out what it is about and how urgent it is,
and route it to the right team, while sending uncertain tickets to a person.

**Pipeline**
1. **Text preprocessing and baselines:** TF-IDF with logistic regression.
2. **Deep learning:** BiLSTM with GloVe embeddings.
3. **Transformer:** DistilBERT fine-tuned with Hugging Face Transformers (used in this app).
4. **Sentiment:** VADER lexicon. **Urgency:** transparent rules, because the dataset has no urgency labels.
5. **Routing:** confidence thresholds tuned on validation data.

**Limitations**
- Trained on English **banking** messages only. Other domains need new training data.
- A message with several issues gets one intent; the model is not multi-label.
- Confidence scores are a little overconfident, so thresholds were checked on held-out data.
- Time-saved figures are estimates based on stated assumptions, not measurements.

**Links:** [GitHub repository]({GITHUB_URL}) | [Model on Hugging Face]({MODEL_URL})
        """
    )