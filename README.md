# TicketFlow: AI Support Ticket Triage and Routing

TicketFlow reads a customer support message, works out what it is about, estimates how urgent it is and how upset the customer is, and routes it to the right team. When the model is not confident, it does not guess: the ticket goes to a human.

Built as a course project for **CSE472: Deep Learning for Natural Language Processing**.

> Live demo: coming soon (Hugging Face Spaces)

![Overview](results/app_overview.png)

## Why this project

In most support teams a person reads every ticket and decides what it is about, how urgent it is and who should handle it. That is slow, expensive and risky: a "my card was stolen" message sitting in the wrong queue can cost a customer real money. TicketFlow automates this first triage step and keeps a human in the loop for anything uncertain.

## What it does

- **Understands the ticket:** predicts one of 77 banking intents (for example `lost_or_stolen_card`, `pending_transfer`) with a fine-tuned DistilBERT model.
- **Scores sentiment and urgency:** VADER for sentiment, transparent rules for urgency, with the reasons shown next to every ticket.
- **Routes by confidence:** auto-route, route with a review flag, or send to a human triage queue.
- **Maps intents to teams:** Fraud & Security, Refunds & Disputes, Top-ups, Transfers & Payments, Cards, Account & Verification, Fees & Exchange, General Support.

![Triage console](results/app_console.png)

## Results

All numbers are on the Banking77 **test set** (3,080 messages that the models never saw during training). Each model was trained once, so small differences between models may not be meaningful.

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression (preprocessed text) | 0.8727 | 0.8729 |
| TF-IDF + Logistic Regression (raw text) | 0.8938 | 0.8942 |
| BiLSTM (random embeddings) | 0.8792 | 0.8793 |
| BiLSTM + GloVe | 0.8990 | 0.8997 |
| **DistilBERT (fine-tuned)** | **0.9263** | **0.9262** |

Observations:

- Preprocessing (stopword removal, lemmatization) **hurt** the TF-IDF model. Messages are short, so removing common words also removed useful signal.
- GloVe embeddings gave the BiLSTM about 2 points of macro F1 over random embeddings.
- DistilBERT is clearly the best, so it is the model used in the app.

### Confidence-based routing

The auto-route threshold (0.90) was chosen on a validation split so that auto-routed tickets are about 97% correct, then checked on the untouched test set. Tickets below 0.50 confidence go to a human.

| Decision | Share of tickets | Intent accuracy | Team accuracy |
|---|---|---|---|
| auto-route (confidence 0.90 or higher) | 90.3% | 0.968 | 0.988 |
| route + needs review (0.50 to 0.90) | 8.0% | 0.576 | 0.865 |
| human triage (below 0.50) | 1.8% | 0.389 | 0.593 |

- Overall **team accuracy is 97.2%**, higher than intent accuracy (92.6%), because similar intents usually belong to the same team.
- Confidence is a useful signal: accuracy falls as confidence falls. The model is slightly overconfident (mean confidence 0.961 vs accuracy 0.926), so confidence should be read as a ranking signal, not an exact probability.
- **Estimated impact (assumptions, not measurements):** with 10,000 tickets a month, 2 minutes of manual triage per ticket, 10 minutes to fix a wrong-team ticket and INR 300 per hour, roughly 91% of triage time could be saved.

### Where the model makes mistakes

Most errors are between near-identical intents, such as `card_arrival` vs `card_delivery_estimate` and `declined_transfer` vs `declined_card_payment`. The weakest intent is `balance_not_updated_after_bank_transfer` (F1 0.753). See `results/error_analysis.md` and `results/distilbert_confusion_matrix.png`.

## The app

A Streamlit web app with eight tabs:

| Tab | Purpose |
|---|---|
| Overview | Key numbers, how it works, routing policy, teams |
| Triage console | Route one ticket and see the full decision card |
| Bulk triage | Route many tickets from pasted text or a CSV file, download the results |
| Ticket queue | Helpdesk-style queue sorted by urgency, with editable status and assignee |
| Analytics | Charts for the session and a monitoring check for data drift |
| Model lab | Model comparison, routing table, error analysis |
| Settings | Response-time targets (SLA), thresholds, urgency rules, team mapping |
| Documentation | Pipeline, metrics, limitations, integration example |

![Analytics](results/app_analytics.png)

## How it works

```
Customer message
      |
      v
DistilBERT  ->  intent (1 of 77) + confidence
      |
      +-->  VADER  ->  sentiment
      +-->  rules  ->  urgency (Low / Medium / High / Critical) + reasons
      |
      v
intent -> team, then confidence decides:
   >= 0.90        auto-route
   0.50 - 0.90    route + needs review
   < 0.50         human triage queue
```

Only the intent comes from a trained model. Sentiment uses the VADER lexicon, and urgency uses transparent rules because the dataset has no urgency labels.

## Tech stack

- **Language:** Python 3.13
- **Deep learning and NLP:** PyTorch, Hugging Face Transformers and Datasets, NLTK, scikit-learn, GloVe (via gensim)
- **App:** Streamlit, Altair, pandas
- **Training:** Google Colab (T4 GPU)
- **Model hosting:** Hugging Face Hub ([trishurao/support-ticket-classifier](https://huggingface.co/trishurao/support-ticket-classifier))

## Project structure

```
src/
  preprocess.py           text cleaning pipeline
  explore.py              dataset exploration and charts
  baseline.py             TF-IDF + Logistic Regression
  bilstm.py               BiLSTM classifier (optional GloVe embeddings)
  bert_train.py           DistilBERT fine-tuning
  compare.py              model comparison table and error analysis
  router.py               sentiment, urgency, team mapping and confidence routing
  threshold_analysis.py   picks thresholds on validation data, checks them on test
app/
  app.py                  Streamlit web app
results/                  metrics, charts and analysis files
deploy/                   script that prepares the Hugging Face Space
.streamlit/config.toml    app theme and settings
requirements.txt
```

## Run it locally

```bash
git clone https://github.com/Tamanna-Rao/support-ticket-router.git
cd support-ticket-router
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app/app.py
```

The first run downloads the model (about 260 MB) from the Hugging Face Hub and the VADER lexicon from NLTK, so it needs an internet connection.

## Reproduce the experiments

```bash
python src/explore.py            # dataset charts
python src/baseline.py           # TF-IDF + Logistic Regression
# on Google Colab with a GPU:
python src/bilstm.py             # BiLSTM, random embeddings
python src/bilstm.py --glove     # BiLSTM, GloVe embeddings (pip install gensim first)
python src/bert_train.py         # DistilBERT fine-tuning
# back on your machine, with the result files in results/:
python src/compare.py            # comparison table and error analysis
python src/threshold_analysis.py # confidence thresholds and routing analysis
```

## Limitations

- Trained on English **banking** messages only. Another domain needs new training data.
- Each message gets one intent. Real tickets can contain several problems (multi-label is future work).
- Urgency is rule-based, not learned.
- Benchmark messages are short and clean. Real tickets with typos, slang or mixed languages may score lower.
- Confidence scores are slightly overconfident.
- The ticket queue lives only in the browser session. A real deployment would use a database or a helpdesk integration.
- The time-saved figure is an estimate based on stated assumptions.

## Future work

Multi-label classification, Hinglish and multilingual support (mBERT, XLM-R), calibrated confidence scores, a FastAPI backend with a database-backed queue, and learned urgency from labelled data.

## Author

Tamanna Rao