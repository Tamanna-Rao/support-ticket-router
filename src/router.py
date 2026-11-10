"""
router.py
Support ticket router: intent (DistilBERT) + sentiment (VADER) + urgency (rules)
+ confidence-based routing.

Quick demo from the project root:
    python src/router.py
"""
import os
import re

MODEL_ID = os.environ.get("TICKET_MODEL_ID", "trishurao/support-ticket-classifier")

# Confidence thresholds (we will tune these from data in the next step)
AUTO_THRESHOLD = 0.80    # at or above: route automatically
REVIEW_THRESHOLD = 0.50  # between the two: route, but flag for review; below: human triage

# ------------------------------------------------------------------ team mapping
# First matching rule wins, so the order matters. Edit these freely.
TEAM_RULES = [
    ("Fraud & Security", ["lost_or_stolen", "compromised", "not_recognised", "swallowed",
                          "pin_blocked", "passcode"]),
    ("Refunds & Disputes", ["refund", "charged_twice", "extra_charge", "wrong_amount", "reverted"]),
    ("Top-ups", ["top_up", "topping_up"]),
    ("Transfers & Payments", ["transfer", "payment", "direct_debit", "beneficiary",
                              "receiving_money", "balance_not_updated"]),
    ("Cards", ["card", "pin", "contactless", "atm", "cash_withdrawal"]),
    ("Account & Verification", ["verify", "verification", "identity", "personal_details",
                                "terminate", "age_limit", "country_support"]),
    ("Fees & Exchange", ["fee", "charge", "exchange", "currency"]),
]
DEFAULT_TEAM = "General Support"

# ------------------------------------------------------------------ urgency rules
CRITICAL_KEYS = ["lost_or_stolen", "compromised", "not_recognised", "swallowed"]
PROBLEM_KEYS = ["failed", "declined", "pending", "not_received", "not_updated", "not_working",
                "refund", "charged_twice", "extra_charge", "wrong_", "blocked", "reverted",
                "not_showing"]
URGENT_RE = re.compile(
    r"\b(urgent|urgently|immediately|asap|right now|emergency|third time|still waiting|"
    r"stolen|fraud|hacked|scam|unauthori[sz]ed)\b",
    re.IGNORECASE,
)
LEVELS = ["Low", "Medium", "High", "Critical"]

_SIA = None  # sentiment analyzer, loaded on first use


def team_for(intent):
    name = intent.lower()
    for team, keys in TEAM_RULES:
        if any(k in name for k in keys):
            return team
    return DEFAULT_TEAM


def get_sentiment(text):
    """VADER sentiment. Returns (label, compound score between -1 and 1)."""
    global _SIA
    if _SIA is None:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        nltk.download("vader_lexicon", quiet=True)
        _SIA = SentimentIntensityAnalyzer()
    score = _SIA.polarity_scores(text)["compound"]
    if score <= -0.5:
        label = "Very negative"
    elif score <= -0.05:
        label = "Negative"
    elif score >= 0.05:
        label = "Positive"
    else:
        label = "Neutral"
    return label, score


def assess_urgency(text, intent, sentiment_score):
    """Rule-based urgency. Returns (level, list of reasons)."""
    name = intent.lower()
    reasons = []
    if any(k in name for k in CRITICAL_KEYS):
        base = 2
        reasons.append(f"critical intent ({intent})")
    elif any(k in name for k in PROBLEM_KEYS):
        base = 1
        reasons.append(f"problem-type intent ({intent})")
    else:
        base = 0
    bump = 0
    if URGENT_RE.search(text):
        bump += 1
        reasons.append("urgent wording")
    if sentiment_score <= -0.5:
        bump += 1
        reasons.append("very negative sentiment")
    return LEVELS[min(base + bump, 3)], reasons


def routing_decision(confidence, auto=AUTO_THRESHOLD, review=REVIEW_THRESHOLD):
    if confidence >= auto:
        return "auto-route"
    if confidence >= review:
        return "route + needs review"
    return "human triage"


def build_queue(team, urgency, decision):
    if decision == "human triage":
        return "Human triage queue"
    queue = team
    if urgency in ("High", "Critical"):
        queue += " - PRIORITY"
    if decision == "route + needs review":
        queue += " (needs review)"
    return queue


# ------------------------------------------------------------------ the model
class TicketRouter:
    def __init__(self, model_id=MODEL_ID, auto=AUTO_THRESHOLD, review=REVIEW_THRESHOLD):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.auto, self.review = auto, review
        print(f"Loading model {model_id} (first time downloads ~260 MB)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text):
        torch = self.torch
        enc = self.tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]
        top = torch.topk(probs, 3)
        top3 = [(self.id2label[int(i)], float(p)) for p, i in zip(top.values, top.indices)]
        intent, confidence = top3[0]

        sent_label, sent_score = get_sentiment(text)
        urgency, reasons = assess_urgency(text, intent, sent_score)
        decision = routing_decision(confidence, self.auto, self.review)
        team = team_for(intent)
        return {
            "intent": intent,
            "confidence": confidence,
            "top3": top3,
            "team": team,
            "sentiment": sent_label,
            "sentiment_score": sent_score,
            "urgency": urgency,
            "urgency_reasons": reasons,
            "decision": decision,
            "queue": build_queue(team, urgency, decision),
        }


if __name__ == "__main__":
    samples = [
        "I was charged twice for my order and nobody is replying. This is unacceptable!",
        "My card was stolen yesterday, please block it immediately!",
        "How do I change my PIN?",
        "Why is my transfer still pending?",
        "asdf not working",
    ]
    router = TicketRouter()
    for s in samples:
        r = router.predict(s)
        print("\n" + "=" * 70)
        print("TICKET    :", s)
        print(f"INTENT    : {r['intent']}  (confidence {r['confidence']:.1%})")
        print("TOP 3     :", ", ".join(f"{n} {p:.0%}" for n, p in r["top3"]))
        print(f"SENTIMENT : {r['sentiment']} ({r['sentiment_score']:+.2f})")
        print(f"URGENCY   : {r['urgency']}  <- {'; '.join(r['urgency_reasons']) or 'no urgency signals'}")
        print("DECISION  :", r["decision"])
        print("QUEUE     :", r["queue"])