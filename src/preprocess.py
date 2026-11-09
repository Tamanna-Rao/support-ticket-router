"""
preprocess.py
Text cleaning pipeline for customer support tickets (Syllabus: Unit I).

Steps: lowercase -> noise removal -> tokenization -> stopword removal -> lemmatization
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data once (safe to call repeatedly)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# Keep negations. Removing "not" turns "my card is not working"
# into "card working", which destroys the meaning.
KEEP_WORDS = {"not", "no", "nor", "never", "cannot", "can't", "won't", "don't",
              "didn't", "isn't", "wasn't", "against", "off", "again"}

STOP_WORDS = {w for w in stopwords.words("english") if "n't" not in w} - KEEP_WORDS
LEMMATIZER = WordNetLemmatizer()

# Pre-compiled regex patterns for noise removal
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUMBER_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase and remove noise (URLs, emails, numbers, punctuation, emojis)."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenization on already-cleaned text."""
    return text.split()


def preprocess(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Full pipeline. Returns a cleaned string (tokens joined by spaces)."""
    tokens = tokenize(clean_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "I was charged TWICE for my order #12345!! Contact me at john@mail.com",
        "My card is not working, see https://bank.com/help",
        "Why hasn't my transfer arrived yet???",
    ]
    for s in samples:
        print("RAW  :", s)
        print("CLEAN:", preprocess(s))
        print()