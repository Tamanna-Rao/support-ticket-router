"""
app.py - TicketFlow: Streamlit app for the Support Ticket Router (v3, professional UI).

Run locally from the project root:
    streamlit run app/app.py
"""
import html
import json
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                                # router.py next to app.py (e.g. on Hugging Face)
sys.path.append(os.path.join(HERE, "..", "src"))     # router.py in src/ (local project)
from router import (  # noqa: E402
    AUTO_THRESHOLD,
    DEFAULT_TEAM,
    REVIEW_THRESHOLD,
    TEAM_RULES,
    TicketRouter,
    build_queue,
    routing_decision,
)

BRAND = "TicketFlow"
GITHUB_URL = "https://github.com/Tamanna-Rao/support-ticket-router"
MODEL_URL = "https://huggingface.co/trishurao/support-ticket-classifier"

st.set_page_config(page_title=f"{BRAND} | Support Ticket Router", page_icon="🎫",
                   layout="wide", initial_sidebar_state="expanded")

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
POLICY = [
    ("Auto-route", "confidence at or above 0.90", "90.3%", "96.8% intent accuracy", "success"),
    ("Route and review", "confidence 0.50 to 0.90", "8.0%", "57.6% intent accuracy", "warn"),
    ("Human triage", "confidence below 0.50", "1.8%", "38.9% intent accuracy", "danger"),
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
DEMO_BATCH = EXAMPLES + [
    "I do not recognise a payment on my statement from yesterday.",
    "How long will my new card take to arrive?",
    "Can I get a refund for a purchase I never received?",
    "The ATM kept my card and I need cash today.",
]
MANUAL_MIN = 2.0   # assumption: minutes a human needs to triage one ticket
REVIEW_MIN = 0.5   # assumption: minutes to check a ticket flagged "needs review"
MAX_BATCH = 200

URG_TONE = {"Critical": "danger", "High": "orange", "Medium": "warn", "Low": "success"}
SENT_TONE = {"Very negative": "danger", "Negative": "orange", "Neutral": "neutral", "Positive": "success"}
DECISION_UI = {
    "auto-route": ("Auto-routed", "success"),
    "route + needs review": ("Routed, needs review", "warn"),
    "human triage": ("Human triage", "danger"),
}
URG_COLORS = {"Critical": "#DC2626", "High": "#EA580C", "Medium": "#D97706", "Low": "#16A34A"}
DEC_COLORS = {"auto-route": "#16A34A", "route + needs review": "#D97706", "human triage": "#DC2626"}
SLA_DEFAULT_MIN = {"Critical": 15, "High": 60, "Medium": 240, "Low": 1440}   # minutes
STATUSES = ["Open", "In progress", "Resolved"]
URG_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
NEXT_STEP = {
    "Fraud & Security": "Verify the customer's identity, block or freeze the card, and open a security case.",
    "Refunds & Disputes": "Check the transaction history, confirm the duplicate or wrong charge, and start a refund or dispute.",
    "Top-ups": "Check the top-up status and the payment provider response. Escalate if the money has left the account.",
    "Transfers & Payments": "Check the transfer status and timing. Contact the receiving bank if it is delayed.",
    "Cards": "Guide the customer through the card or PIN request and check delivery or activation status.",
    "Account & Verification": "Review the verification status and required documents, then send the customer the next step.",
    "Fees & Exchange": "Explain the fee or exchange rate and link the relevant policy page.",
    "General Support": "Answer from a help-centre article or ask the customer a clarifying question.",
}

# ------------------------------------------------------------------ theme and styling
# Pick a palette by changing ACTIVE_THEME: "plum", "emerald", "graphite" or "wine".
# Also set the matching primaryColor and backgroundColor in .streamlit/config.toml.
THEMES = {
    "plum": {   # deep purple: no clash with the green / amber / red status colours
        "brand": "#6B21A8", "brand_hi": "#9333EA", "hero_a": "#1E1033", "hero_b": "#3B0764", "hero_c": "#6B21A8",
        "hero_tag": "#E9D5FF", "hero_chip": "#F3E8FF", "soft": "#F3E8FF", "soft_text": "#581C87",
        "soft_line": "#E9D5FF", "chart_lo": "#D8B4FE",
    },
    "emerald": {
        "brand": "#047857", "brand_hi": "#059669", "hero_a": "#052E2B", "hero_b": "#064E3B", "hero_c": "#047857",
        "hero_tag": "#A7F3D0", "hero_chip": "#ECFDF5", "soft": "#D1FAE5", "soft_text": "#065F46",
        "soft_line": "#A7F3D0", "chart_lo": "#6EE7B7",
    },
    "graphite": {   # charcoal with a gold highlight
        "brand": "#1F2937", "brand_hi": "#4B5563", "hero_a": "#0B0F14", "hero_b": "#1F2937", "hero_c": "#374151",
        "hero_tag": "#FCD34D", "hero_chip": "#F3F4F6", "soft": "#F3F4F6", "soft_text": "#1F2937",
        "soft_line": "#E5E7EB", "chart_lo": "#9CA3AF",
    },
    "wine": {
        "brand": "#881337", "brand_hi": "#BE123C", "hero_a": "#2A0A14", "hero_b": "#4C0519", "hero_c": "#881337",
        "hero_tag": "#FECDD3", "hero_chip": "#FFF1F2", "soft": "#FFE4E6", "soft_text": "#881337",
        "soft_line": "#FECDD3", "chart_lo": "#FDA4AF",
    },
}
ACTIVE_THEME = "plum"
THEME = THEMES.get(ACTIVE_THEME, THEMES["plum"])
ROOT_VARS = ":root {" + "".join(f"--{k.replace('_', '-')}: {v};" for k, v in THEME.items()) + "}"

CSS_RULES = """
.stApp {font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1240px;}
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E7E5E4;}
.topbar {display: flex; align-items: center; justify-content: space-between; padding: .3rem 0 1rem 0;}
.brand {display: flex; align-items: center; gap: .75rem;}
.logo {width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, var(--brand), var(--brand-hi)); color: #FFFFFF; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.25rem;}
.brand-name {font-size: 1.4rem; font-weight: 800; color: #1C1917; line-height: 1.1;}
.brand-sub {font-size: .82rem; color: #78716C;}
.live {display: inline-flex; align-items: center; gap: .45rem; background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; padding: .3rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.live-dot {width: 8px; height: 8px; border-radius: 50%; background: #10B981;}
.hero {background: linear-gradient(125deg, var(--hero-a) 0%, var(--hero-b) 45%, var(--hero-c) 100%); padding: 2.4rem 2.6rem; border-radius: 20px; margin-bottom: 1.4rem;}
.hero-tag {color: var(--hero-tag); font-size: .78rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .7rem;}
.hero h1 {color: #FFFFFF; font-size: 2.4rem; line-height: 1.15; margin: 0 0 .8rem 0; padding: 0; font-weight: 800;}
.hero p {color: #E7E5E4; font-size: 1.05rem; max-width: 760px; margin: 0 0 1.1rem 0;}
.hero-chips span {display: inline-block; background: rgba(255,255,255,.12); color: var(--hero-chip); border: 1px solid rgba(255,255,255,.2); padding: .3rem .8rem; border-radius: 999px; font-size: .82rem; margin: 0 .5rem .4rem 0;}
.kpi {display: flex; gap: .9rem; align-items: center; background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 16px; padding: 1.05rem 1.15rem; box-shadow: 0 1px 2px rgba(28,25,23,.05); height: 100%;}
.kpi-icon {width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: 700; flex-shrink: 0;}
.tone-brand {background: var(--soft); color: var(--soft-text);}
.tone-slate {background: #F5F5F4; color: #44403C;}
.tone-green {background: #DCFCE7; color: #15803D;}
.tone-amber {background: #FEF3C7; color: #B45309;}
.tone-red {background: #FEE2E2; color: #B91C1C;}
.kpi-label {font-size: .76rem; color: #78716C; text-transform: uppercase; letter-spacing: .07em; font-weight: 600;}
.kpi-value {font-size: 1.7rem; font-weight: 800; color: #1C1917; line-height: 1.2;}
.kpi-sub {font-size: .8rem; color: #78716C;}
.section-title {font-size: 1.25rem; font-weight: 700; color: #1C1917; margin: 1.6rem 0 .2rem 0;}
.section-sub {font-size: .9rem; color: #78716C; margin-bottom: .8rem;}
.panel {background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 16px; padding: 1.2rem 1.3rem; box-shadow: 0 1px 2px rgba(28,25,23,.05); height: 100%;}
.step-num {display: inline-flex; width: 30px; height: 30px; align-items: center; justify-content: center; border-radius: 50%; background: var(--brand); color: #FFFFFF; font-weight: 700; margin-bottom: .6rem;}
.panel-title {font-size: 1.05rem; font-weight: 700; color: #1C1917; margin-bottom: .25rem;}
.panel-text {font-size: .88rem; color: #57534E; line-height: 1.5;}
.policy {background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 16px; padding: 1.1rem 1.25rem; border-top: 5px solid #D6D3D1; height: 100%;}
.policy-success {border-top-color: #16A34A;}
.policy-warn {border-top-color: #D97706;}
.policy-danger {border-top-color: #DC2626;}
.policy-title {font-weight: 700; color: #1C1917; font-size: 1.05rem;}
.policy-rule {color: #78716C; font-size: .85rem; margin-bottom: .5rem;}
.policy-share {font-size: 2rem; font-weight: 800; color: #1C1917; line-height: 1.1;}
.policy-sub {color: #78716C; font-size: .82rem;}
.pill {display: inline-block; padding: .22rem .75rem; border-radius: 999px; font-size: .8rem; font-weight: 600; margin: 0 .4rem .4rem 0;}
.pill-success {background: #DCFCE7; color: #166534;}
.pill-warn {background: #FEF3C7; color: #92400E;}
.pill-orange {background: #FFEDD5; color: #9A3412;}
.pill-danger {background: #FEE2E2; color: #991B1B;}
.pill-info {background: var(--soft); color: var(--soft-text);}
.pill-neutral {background: #F5F5F4; color: #44403C;}
.result {background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 18px; padding: 1.3rem 1.4rem; box-shadow: 0 4px 14px rgba(28,25,23,.07);}
.result-head {display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: .8rem;}
.eyebrow {font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: #78716C; font-weight: 700; margin: .7rem 0 .3rem 0;}
.result-queue {font-size: 1.45rem; font-weight: 800; color: #1C1917; line-height: 1.2;}
.bar {height: 10px; border-radius: 999px; background: #E7E5E4; overflow: hidden;}
.bar.small {height: 7px;}
.bar-fill {height: 100%; border-radius: 999px; background: var(--brand-hi);}
.conf-text {font-size: .85rem; color: #57534E; margin-top: .25rem;}
.two {display: flex; gap: .8rem; margin-top: .4rem;}
.mini {flex: 1; background: #FAFAF9; border: 1px solid #E7E5E4; border-radius: 12px; padding: .7rem .9rem;}
.mini-val {font-weight: 700; color: #1C1917;}
.note {background: var(--soft); border: 1px solid var(--soft-line); color: var(--soft-text); border-radius: 12px; padding: .8rem 1rem; margin-top: .9rem; font-size: .9rem; line-height: 1.5;}
.t3 {display: grid; grid-template-columns: 1.7fr 2fr .5fr; gap: .7rem; align-items: center; margin-bottom: .35rem;}
.t3-name {font-size: .85rem; color: #44403C; font-family: ui-monospace, Consolas, monospace;}
.t3-val {font-size: .85rem; color: #44403C; text-align: right; font-weight: 600;}
.empty {background: #FFFFFF; border: 2px dashed #D6D3D1; border-radius: 18px; padding: 3.2rem 1.5rem; text-align: center; color: #78716C;}
.empty b {display: block; color: #1C1917; font-size: 1.1rem; margin-bottom: .3rem;}
.team-card {background: #FFFFFF; border: 1px solid #E7E5E4; border-radius: 14px; padding: .9rem 1rem; height: 100%;}
.team-name {font-weight: 700; color: #1C1917; margin-bottom: .4rem;}
.tag {display: inline-block; background: #F5F5F4; color: #57534E; border-radius: 6px; padding: .1rem .5rem; font-size: .75rem; margin: 0 .25rem .25rem 0;}
.footer {text-align: center; color: #A8A29E; font-size: .8rem; margin-top: 2.5rem;}
div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {background: linear-gradient(120deg, var(--brand), var(--brand-hi)); border: none; color: #FFFFFF; font-weight: 600; border-radius: 10px; padding: .55rem 1.3rem;}
button[data-baseweb="tab"] {font-size: .98rem; font-weight: 600; padding: .7rem 1.1rem;}
button[data-baseweb="tab"][aria-selected="true"] {color: var(--brand);}
div[data-baseweb="tab-highlight"] {background-color: var(--brand) !important;}
div[data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid #E7E5E4;}
div[data-testid="stSlider"] div[role="slider"] {background-color: var(--brand) !important;}
"""
st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
    + ROOT_VARS + CSS_RULES + "</style>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ small HTML helpers
# Every helper returns ONE line of HTML (no blank lines or indentation) so Markdown cannot break it.
def esc(x):
    return html.escape(str(x), quote=True)


def fmt_minutes(m):
    m = int(m)
    if m % 1440 == 0:
        d = m // 1440
        return f"{d} day" + ("s" if d != 1 else "")
    if m >= 60:
        h, r = divmod(m, 60)
        return f"{h} hour" + ("s" if h != 1 else "") + (f" {r} min" if r else "")
    return f"{m} minutes"


def pill(text, tone="neutral"):
    return f'<span class="pill pill-{tone}">{esc(text)}</span>'


def kpi(label, value, sub="", icon="•", tone="brand"):
    return (f'<div class="kpi"><div class="kpi-icon tone-{tone}">{icon}</div><div>'
            f'<div class="kpi-label">{esc(label)}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{esc(sub)}</div></div></div>')


def section(title, sub=""):
    return (f'<div class="section-title">{esc(title)}</div>'
            f'<div class="section-sub">{esc(sub)}</div>')


def bar_html(frac, color="var(--brand-hi)", small=False):
    pct = max(0.0, min(1.0, float(frac))) * 100
    cls = "bar small" if small else "bar"
    return f'<div class="{cls}"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'


def conf_color(conf, auto_t, review_t):
    if conf >= auto_t:
        return "#16A34A"
    if conf >= review_t:
        return "#D97706"
    return "#DC2626"


def top3_html(top3):
    rows = []
    for name, p in top3:
        rows.append(f'<div class="t3"><div class="t3-name">{esc(name)}</div>{bar_html(p, small=True)}'
                    f'<div class="t3-val">{p:.0%}</div></div>')
    return "".join(rows)


def result_card_html(raw, rec, auto_t, review_t, sla_min=None):
    label, tone = DECISION_UI[rec["decision"]]
    human = rec["decision"] == "human triage"
    sla = "Set by the triage agent" if human else fmt_minutes((sla_min or SLA_DEFAULT_MIN)[raw["urgency"]])
    step = ("The model is not confident. Ask the customer a clarifying question or assign the ticket manually."
            if human else NEXT_STEP.get(raw["team"], NEXT_STEP["General Support"]))
    reasons = "".join(pill(r, "info") for r in raw["urgency_reasons"]) or pill("no urgency signals", "neutral")
    return "".join([
        '<div class="result">',
        '<div class="result-head"><div><div class="eyebrow" style="margin-top:0">Routing decision</div>',
        f'<div class="result-queue">{esc(rec["queue"])}</div></div>{pill(label, tone)}</div>',
        '<div>',
        pill(f"Urgency: {raw['urgency']}", URG_TONE[raw["urgency"]]),
        pill(f"Sentiment: {raw['sentiment']}", SENT_TONE.get(raw["sentiment"], "neutral")),
        pill(raw["intent"], "info"),
        '</div>',
        '<div class="eyebrow">Model confidence</div>',
        bar_html(raw["confidence"], conf_color(raw["confidence"], auto_t, review_t)),
        f'<div class="conf-text">{raw["confidence"]:.1%} (auto-route at {auto_t:.2f}, human triage below {review_t:.2f})</div>',
        '<div class="two">',
        f'<div class="mini"><div class="eyebrow" style="margin-top:0">Target response time</div><div class="mini-val">{esc(sla)}</div></div>',
        f'<div class="mini"><div class="eyebrow" style="margin-top:0">Predicted team</div><div class="mini-val">{esc(raw["team"])}</div></div>',
        '</div>',
        f'<div class="note"><b>Suggested next step</b> (template, not model-generated)<br>{esc(step)}</div>',
        '<div class="eyebrow">Top 3 intents</div>',
        top3_html(raw["top3"]),
        '<div class="eyebrow">Why this urgency</div>',
        f'<div>{reasons}</div>',
        '</div>',
    ])


# ------------------------------------------------------------------ data helpers
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


def count_chart(df, column, color_map=None, height=None):
    counts = df[column].value_counts().rename_axis(column).reset_index(name="tickets")
    chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=5, size=24).encode(
        y=alt.Y(f"{column}:N", sort="-x", title=None),
        x=alt.X("tickets:Q", title=None, axis=alt.Axis(tickMinStep=1)),
        tooltip=[column, "tickets"],
    )
    if color_map:
        chart = chart.encode(color=alt.Color(
            f"{column}:N", legend=None,
            scale=alt.Scale(domain=list(color_map), range=list(color_map.values()))))
    else:
        chart = chart.encode(color=alt.value(THEME["brand_hi"]))
    return chart.properties(height=height or max(110, 38 * len(counts)))


def donut_chart(df, column, color_map):
    counts = df[column].value_counts().rename_axis(column).reset_index(name="tickets")
    return alt.Chart(counts).mark_arc(innerRadius=55).encode(
        theta="tickets:Q",
        color=alt.Color(f"{column}:N", legend=alt.Legend(title=None, orient="bottom"),
                        scale=alt.Scale(domain=list(color_map), range=list(color_map.values()))),
        tooltip=[column, "tickets"],
    ).properties(height=220)


for key, default in (("history", []), ("last", None), ("batch", None), ("next_id", 1)):
    if key not in st.session_state:
        st.session_state[key] = default
if "sla_min" not in st.session_state:
    st.session_state.sla_min = dict(SLA_DEFAULT_MIN)

# ------------------------------------------------------------------ sidebar
auto_default, review_default = default_thresholds()
st.sidebar.markdown(f"### {BRAND}")
st.sidebar.caption("Routing policy")
auto_t = st.sidebar.slider("Auto-route at confidence >=", 0.50, 0.99, float(auto_default), 0.01)
review_t = st.sidebar.slider("Human triage below", 0.10, 0.90, float(review_default), 0.01)
if review_t > auto_t:
    st.sidebar.warning("The triage threshold should not be higher than the auto-route threshold.")
st.sidebar.caption("Tickets between the two thresholds are routed but flagged for review. "
                   "Defaults come from the validation-set analysis.")
st.sidebar.caption("DistilBERT fine-tuned on Banking77")


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


def add_to_history(rows):
    """Store routed tickets for the Queue, Analytics and Monitoring tabs."""
    for r in rows:
        r["id"] = st.session_state.next_id
        st.session_state.next_id += 1
        r.setdefault("status", "Open")
        r.setdefault("assignee", "")
        st.session_state.history.append(r)


# ------------------------------------------------------------------ top bar and tabs
st.markdown(
    f'<div class="topbar"><div class="brand"><div class="logo">T</div><div>'
    f'<div class="brand-name">{BRAND}</div><div class="brand-sub">AI support ticket triage and routing</div>'
    f'</div></div><div class="live"><span class="live-dot"></span>Model ready: DistilBERT</div></div>',
    unsafe_allow_html=True,
)

(tab_over, tab_try, tab_batch, tab_queue, tab_dash, tab_lab, tab_set, tab_docs) = st.tabs(
    ["Overview", "Triage console", "Bulk triage", "Ticket queue", "Analytics", "Model lab", "Settings",
     "Documentation"]
)

# ------------------------------------------------------------------ Settings
# This block runs before the other tabs, so a changed response-time target shows up immediately.
with tab_set:
    st.markdown(section("Settings", "Response-time targets and the policy this workspace uses"),
                unsafe_allow_html=True)
    st.markdown("#### Response-time targets (SLA)")
    st.caption("Example targets shown on every routed ticket. Change them to match your own support policy.")
    for col, level in zip(st.columns(4), ["Critical", "High", "Medium", "Low"]):
        st.session_state.sla_min[level] = int(col.number_input(
            f"{level} (minutes)", min_value=1, max_value=10080, step=5,
            value=SLA_DEFAULT_MIN[level], key=f"sla_{level}"))
    st.markdown("#### Routing thresholds")
    st.write(f"Auto-route at confidence **{auto_t:.2f}** or higher. Human triage below **{review_t:.2f}**. "
             "Change them with the sliders in the sidebar.")
    st.markdown("#### Urgency rules")
    rules = pd.DataFrame([
        ("Critical intent (lost or stolen, compromised, not recognised, swallowed)", "Starts at High"),
        ("Problem-type intent (failed, declined, pending, not received, refund, and similar)", "Starts at Medium"),
        ("Any other intent", "Starts at Low"),
        ("Urgent wording (urgent, immediately, fraud, stolen, and similar)", "Raises urgency by one level"),
        ("Very negative sentiment (VADER score of -0.5 or lower)", "Raises urgency by one level"),
        ("Maximum", "Critical"),
    ], columns=["Signal", "Effect"])
    st.table(rules.set_index("Signal"))
    st.markdown("#### Team mapping")
    mapping = pd.DataFrame(
        [(name, ", ".join(k.replace("_", " ") for k in keys)) for name, keys in TEAM_RULES]
        + [(DEFAULT_TEAM, "everything else")],
        columns=["Team", "Handles intents containing"])
    st.dataframe(mapping, hide_index=True)

# ------------------------------------------------------------------ Overview
with tab_over:
    st.markdown(
        '<div class="hero"><div class="hero-tag">AI-POWERED SUPPORT TRIAGE</div>'
        '<h1>Route every customer ticket to the right team, instantly.</h1>'
        '<p>Reads an incoming message, identifies the intent, scores urgency and sentiment, and sends it to the '
        'right queue. When the model is unsure, a person takes over.</p>'
        '<div class="hero-chips"><span>DistilBERT</span><span>77 intents</span><span>Confidence-based routing</span>'
        '<span>Explainable urgency</span><span>Human in the loop</span></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Intent accuracy", "92.6%", "3,080 held-out test messages", "✓", "brand"), unsafe_allow_html=True)
    c2.markdown(kpi("Team accuracy", "97.2%", "ticket reaches the right team", "⇄", "slate"), unsafe_allow_html=True)
    c3.markdown(kpi("Auto-routed", "90.3%", "at 0.90 confidence, 96.8% correct", "⚡", "green"), unsafe_allow_html=True)
    c4.markdown(kpi("Triage time saved", "~91%", "estimate from stated assumptions", "⏱", "amber"), unsafe_allow_html=True)

    st.markdown(section("How it works", "From raw message to routed ticket in four steps"), unsafe_allow_html=True)
    steps = [
        ("1", "Understand", "A DistilBERT model fine-tuned on 77 banking intents reads the ticket."),
        ("2", "Assess", "VADER scores sentiment. Transparent rules estimate urgency and explain why."),
        ("3", "Decide", "High confidence is auto-routed. Medium is flagged for review. Low goes to a human."),
        ("4", "Route", "The intent maps to a team and a priority queue with a target response time."),
    ]
    for col, (num, title, text) in zip(st.columns(4), steps):
        col.markdown(f'<div class="panel"><div class="step-num">{num}</div><div class="panel-title">{title}</div>'
                     f'<div class="panel-text">{text}</div></div>', unsafe_allow_html=True)

    st.markdown(section("Routing policy", "Share of test tickets and accuracy in each confidence band"),
                unsafe_allow_html=True)
    for col, (title, rule, share, acc, tone) in zip(st.columns(3), POLICY):
        col.markdown(f'<div class="policy policy-{tone}"><div class="policy-title">{title}</div>'
                     f'<div class="policy-rule">{rule}</div><div class="policy-share">{share}</div>'
                     f'<div class="policy-sub">of tickets, {acc}</div></div>', unsafe_allow_html=True)

    st.markdown(section("Teams", "Each predicted intent is mapped to one of these queues"), unsafe_allow_html=True)
    teams = [(name, [k.replace("_", " ") for k in keys[:4]]) for name, keys in TEAM_RULES]
    teams.append((DEFAULT_TEAM, ["everything else"]))
    for row_start in range(0, len(teams), 3):
        for col, (name, tags) in zip(st.columns(3), teams[row_start:row_start + 3]):
            chips = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags)
            col.markdown(f'<div class="team-card"><div class="team-name">{esc(name)}</div>{chips}</div>',
                         unsafe_allow_html=True)
        st.write("")

# ------------------------------------------------------------------ Triage console
with tab_try:
    st.markdown(section("Triage console", "Paste one customer message and see the routing decision"),
                unsafe_allow_html=True)
    left, right = st.columns([5, 7], gap="large")
    with left:
        choice = st.selectbox("Load an example", ["(write your own)"] + EXAMPLES)
        idx = EXAMPLES.index(choice) if choice in EXAMPLES else "custom"
        ticket = st.text_area("Customer ticket", value="" if idx == "custom" else choice, height=150,
                              key=f"ticket_{idx}", placeholder="Type or paste a customer message...")
        if st.button("Route ticket", type="primary"):
            if not ticket.strip():
                st.warning("Please enter a ticket first.")
            else:
                raw = load_router().predict(ticket.strip())
                st.session_state.last = {"ticket": ticket.strip(), "raw": raw}
                add_to_history([record(ticket.strip(), raw)])
        st.caption("Move the sliders in the sidebar to see how the thresholds change the decision.")
    with right:
        last = st.session_state.last
        if last:
            rec = record(last["ticket"], last["raw"])  # thresholds applied here, so the sliders update it live
            st.markdown(result_card_html(last["raw"], rec, auto_t, review_t, st.session_state.sla_min), unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty"><b>No ticket routed yet</b>'
                        'Choose an example or write a message, then press Route ticket.</div>',
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ Bulk triage
with tab_batch:
    st.markdown(section("Bulk triage", "Route many tickets at once from pasted text or a CSV file"),
                unsafe_allow_html=True)
    mode = st.radio("Input", ["Paste tickets (one per line)", "Upload a CSV file"], horizontal=True)
    tickets = []
    if mode.startswith("Paste"):
        if st.button("Load demo tickets"):
            st.session_state["bulk_text"] = "\n".join(DEMO_BATCH)
        pasted = st.text_area("Tickets", height=200, key="bulk_text", placeholder="One ticket per line...")
        tickets = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
    else:
        template = pd.DataFrame({"ticket": DEMO_BATCH[:3]}).to_csv(index=False).encode("utf-8")
        st.download_button("Download a CSV template", template, "tickets_template.csv", "text/csv")
        uploaded = st.file_uploader("CSV with a 'ticket' column (the first column is used if there is none)",
                                    type=["csv"])
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
            progress = st.progress(0.0, text="Processing...")
            rows = []
            for i, t in enumerate(tickets, 1):
                rows.append(record(t, router.predict(t)))
                progress.progress(i / len(tickets), text=f"Processing {i}/{len(tickets)}")
            progress.empty()
            st.session_state.batch = pd.DataFrame(rows)
            add_to_history(rows)

    result = st.session_state.batch
    if result is not None and len(result):
        n = len(result)
        urgent = int(result["urgency"].isin(["High", "Critical"]).sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Tickets", n, "processed", "☰", "brand"), unsafe_allow_html=True)
        k2.markdown(kpi("Auto-routed", f"{(result['decision'] == 'auto-route').mean():.0%}", "no human needed",
                        "⚡", "green"), unsafe_allow_html=True)
        k3.markdown(kpi("Human triage", f"{(result['decision'] == 'human triage').mean():.0%}", "low confidence",
                        "!", "red"), unsafe_allow_html=True)
        k4.markdown(kpi("High or critical", urgent, "need fast attention", "▲", "amber"), unsafe_allow_html=True)
        st.write("")
        st.dataframe(
            result,
            hide_index=True,
            column_config={
                "ticket": st.column_config.TextColumn("ticket", width="large"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0,
                                                              format="%.2f"),
            },
        )
        st.download_button("Download results (CSV)", result.to_csv(index=False).encode("utf-8"),
                           "routed_tickets.csv", "text/csv")

# ------------------------------------------------------------------ Ticket queue
with tab_queue:
    st.markdown(section("Ticket queue", "Tickets routed in this session, most urgent first. "
                                        "Edit the status and assignee in the table."), unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.markdown('<div class="empty"><b>The queue is empty</b>Route a ticket (single or bulk) and it will '
                    'appear here.</div>', unsafe_allow_html=True)
    else:
        kpi_row = st.container()  # filled after the table, so the numbers include the latest edits
        q = pd.DataFrame(hist)
        q["_rank"] = q["urgency"].map(URG_RANK)
        q = q.sort_values(["_rank", "id"]).drop(columns="_rank")
        teams_sel = st.multiselect("Filter by team", sorted(q["team"].unique()))
        view = q[q["team"].isin(teams_sel)] if teams_sel else q
        cols = ["id", "urgency", "queue", "ticket", "status", "assignee", "confidence", "decision"]
        edited = st.data_editor(
            view[cols],
            hide_index=True,
            key=f"queue_{len(hist)}_{'-'.join(teams_sel)}",
            disabled=[c for c in cols if c not in ("status", "assignee")],
            column_config={
                "ticket": st.column_config.TextColumn("ticket", width="large"),
                "status": st.column_config.SelectboxColumn("status", options=STATUSES, required=True),
                "assignee": st.column_config.TextColumn("assignee", help="Type an agent name"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0,
                                                              format="%.2f"),
            },
        )
        by_id = {r["id"]: r for r in st.session_state.history}
        for _, row in edited.iterrows():
            target = by_id.get(int(row["id"]))
            if target is not None:
                target["status"] = row["status"]
                target["assignee"] = "" if pd.isna(row["assignee"]) else str(row["assignee"])
        st.caption("Statuses are kept for this browser session only. In production they would be stored in a "
                   "database or synced with your helpdesk.")

        latest = pd.DataFrame(st.session_state.history)
        n_open = int((latest["status"] == "Open").sum())
        n_prog = int((latest["status"] == "In progress").sum())
        n_done = int((latest["status"] == "Resolved").sum())
        n_hot = int(((latest["status"] != "Resolved") & latest["urgency"].isin(["High", "Critical"])).sum())
        with kpi_row:
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(kpi("Open", n_open, "waiting for an agent", "☰", "brand"), unsafe_allow_html=True)
            k2.markdown(kpi("In progress", n_prog, "being handled", "⚙", "slate"), unsafe_allow_html=True)
            k3.markdown(kpi("Resolved", n_done, "closed", "✓", "green"), unsafe_allow_html=True)
            k4.markdown(kpi("Urgent and unresolved", n_hot, "high or critical", "▲", "red"),
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ Analytics
with tab_dash:
    st.markdown(section("Analytics", "A live summary of every ticket routed in this session"),
                unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.markdown('<div class="empty"><b>No data yet</b>Route a few tickets (single or bulk) and the '
                    'analytics will appear here.</div>', unsafe_allow_html=True)
    else:
        df = pd.DataFrame(hist)
        n = len(df)
        n_auto = int((df["decision"] == "auto-route").sum())
        n_review = int((df["decision"] == "route + needs review").sum())
        saved = n_auto * MANUAL_MIN + n_review * (MANUAL_MIN - REVIEW_MIN)
        urgent = int(df["urgency"].isin(["High", "Critical"]).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(kpi("Tickets processed", n, "this session", "☰", "brand"), unsafe_allow_html=True)
        m2.markdown(kpi("Auto-routed", f"{n_auto / n:.0%}", "automation rate", "⚡", "green"), unsafe_allow_html=True)
        m3.markdown(kpi("High or critical", urgent, "priority tickets", "▲", "amber"), unsafe_allow_html=True)
        m4.markdown(kpi("Triage time saved", f"{saved:.0f} min", "estimate", "⏱", "slate"), unsafe_allow_html=True)
        st.caption(f"Time saved assumes {MANUAL_MIN:g} min of manual triage per ticket and "
                   f"{REVIEW_MIN:g} min to check a flagged one. It is an estimate, not a measurement.")

        st.write("")
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="panel-title">Tickets per team</div>', unsafe_allow_html=True)
            st.altair_chart(count_chart(df, "team"))
        with b:
            st.markdown('<div class="panel-title">Urgency levels</div>', unsafe_allow_html=True)
            st.altair_chart(count_chart(df, "urgency", URG_COLORS))
        c, d = st.columns(2)
        with c:
            st.markdown('<div class="panel-title">Routing decisions</div>', unsafe_allow_html=True)
            st.altair_chart(donut_chart(df, "decision", DEC_COLORS))
        with d:
            st.markdown('<div class="panel-title">Predicted intents</div>', unsafe_allow_html=True)
            st.altair_chart(count_chart(df, "intent"))

        st.markdown(section("Monitoring", "Do routing decisions in this session look like what we saw in testing?"),
                    unsafe_allow_html=True)
        if n >= 20:
            expected = {"auto-route": 0.903, "route + needs review": 0.080, "human triage": 0.018}
            observed = df["decision"].value_counts(normalize=True)
            mon = pd.DataFrame({
                "decision": list(expected) * 2,
                "source": ["Test set"] * 3 + ["This session"] * 3,
                "share": list(expected.values()) + [float(observed.get(k, 0.0)) for k in expected],
            })
            mon_chart = alt.Chart(mon).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("decision:N", title=None, sort=list(expected)),
                xOffset="source:N",
                y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("source:N", legend=alt.Legend(title=None, orient="bottom"),
                                scale=alt.Scale(domain=["Test set", "This session"],
                                                range=[THEME["chart_lo"], THEME["brand"]])),
                tooltip=["decision", "source", alt.Tooltip("share:Q", format=".1%")],
            ).properties(height=260)
            st.altair_chart(mon_chart)
            gap = float(observed.get("auto-route", 0.0)) - expected["auto-route"]
            if abs(gap) > 0.15:
                st.warning(f"The auto-route share differs from the test set by {abs(gap) * 100:.0f} points. "
                           "In real use this can mean the incoming tickets differ from the training data (data drift).")
            else:
                st.success("The routing mix is close to the test-set mix.")
        else:
            st.info("Route at least 20 tickets to see monitoring (Bulk triage, then Load demo tickets, twice).")

        st.markdown('<div class="panel-title">Ticket history</div>', unsafe_allow_html=True)
        st.dataframe(df, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last = None
            st.session_state.batch = None
            st.session_state.next_id = 1
            st.rerun()

# ------------------------------------------------------------------ Model lab
with tab_lab:
    st.markdown(section("Model lab", "How the five models compare on the Banking77 test set"),
                unsafe_allow_html=True)
    perf = pd.DataFrame(MODEL_ROWS, columns=["Model", "Accuracy", "Macro F1"])
    best = perf.loc[perf["Macro F1"].idxmax(), "Model"]
    base = alt.Chart(perf).encode(
        y=alt.Y("Model:N", sort="-x", title=None),
        x=alt.X("Macro F1:Q", scale=alt.Scale(domain=[0.80, 0.95], clamp=True), title="Macro F1 (axis starts at 0.80)"),
    )
    bars = base.mark_bar(cornerRadiusEnd=5, size=26).encode(
        color=alt.condition(alt.datum.Model == best, alt.value(THEME["brand"]), alt.value(THEME["chart_lo"])),
        tooltip=["Model", "Accuracy", "Macro F1"],
    )
    labels = base.mark_text(align="left", dx=5, color="#44403C").encode(text=alt.Text("Macro F1:Q", format=".3f"))
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.altair_chart((bars + labels).properties(height=260))
    with right:
        st.table(perf.set_index("Model"))
    st.caption("Banking77 test set: 3,080 messages, 77 intents. One training run per model, so small differences "
               "between models may not be meaningful.")

    st.markdown(section("Confidence-based routing", "Thresholds were chosen on a validation set and checked on the "
                                                    "test set"), unsafe_allow_html=True)
    routing = pd.DataFrame(ROUTING_ROWS, columns=["Decision", "Share of tickets", "Intent accuracy", "Team accuracy"])
    st.table(routing.set_index("Decision"))
    st.caption("Team accuracy is higher than intent accuracy because similar intents usually belong to the same team.")

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
        st.markdown(section("Charts from the experiments"), unsafe_allow_html=True)
        for col, (path, cap) in zip(st.columns(len(available)), available):
            col.image(path, caption=cap)

# ------------------------------------------------------------------ Documentation
with tab_docs:
    st.markdown(section("Documentation", "What this system does, how it was built, and where it falls short"),
                unsafe_allow_html=True)
    st.markdown(
        f"""
**Goal.** Automate the first step of customer support: read a ticket, work out what it is about and how urgent
it is, and route it to the right team, while sending uncertain tickets to a person.

#### Pipeline
1. **Text preprocessing and baselines:** TF-IDF with logistic regression.
2. **Deep learning:** BiLSTM with GloVe embeddings.
3. **Transformer:** DistilBERT fine-tuned with Hugging Face Transformers (used in this app).
4. **Sentiment:** VADER lexicon. **Urgency:** transparent rules, because the dataset has no urgency labels.
5. **Routing:** confidence thresholds tuned on validation data, then checked on the test set.

#### Metrics
- **Intent accuracy:** share of tickets where the exact intent is right.
- **Macro F1:** average F1 over all 77 intents, treating each equally.
- **Team accuracy:** share of tickets that reach the right team, even if the exact intent is wrong.

#### Responsible use
- Low-confidence tickets always go to a person.
- Urgency is rule-based and explained, so an agent can see why a ticket was prioritised.
- Time-saved figures are estimates from stated assumptions, not measurements.

#### Limitations
- Trained on English **banking** messages only. Other domains need new training data.
- A message with several issues gets one intent; the model is not multi-label.
- Confidence scores are slightly overconfident, so thresholds were checked on held-out data.
- Benchmark results come from clean, short messages. Real tickets with typos and slang may score lower.

#### Run it yourself
```
pip install -r requirements.txt
streamlit run app/app.py
```

        """
    )

    st.markdown("#### Use the router in your own code")
    st.code(
        "from router import TicketRouter\n\n"
        "router = TicketRouter()\n"
        'result = router.predict("My card was stolen, please block it!")\n'
        'print(result["queue"], result["urgency"], result["confidence"])',
        language="python",
    )
    st.markdown("#### Example output (JSON payload)")
    if st.session_state.last:
        st.json(st.session_state.last["raw"])
    else:
        st.caption("Route a ticket in the Triage console to see its JSON payload here.")

st.markdown(f'<div class="footer">{BRAND} | Built with Streamlit, PyTorch and Hugging Face Transformers</div>',
            unsafe_allow_html=True)