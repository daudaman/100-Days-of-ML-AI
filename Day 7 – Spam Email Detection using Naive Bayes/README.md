# Day 7 – Spam Email Detection using Naive Bayes

Predicting whether a message is **Spam** or **Not Spam** using the Naive Bayes algorithm. This is a Natural Language Processing (NLP) classification project based on real SMS text message data.

---

## Dataset

The [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) contains **5,572 labelled text messages**.

| Column  | Description               |
|---------|---------------------------|
| label   | `ham` (Not Spam) or `spam`|
| message | Raw SMS text              |

Place the downloaded CSV as `dataset/spam.csv`.

---

## Model

### Multinomial Naive Bayes + TF-IDF Vectorization

| Step | Details |
|------|---------|
| Text Cleaning | Lowercase → remove punctuation → remove stopwords → stem |
| Vectorization | TF-IDF, top 5,000 features, unigrams + bigrams |
| Algorithm | `MultinomialNB(alpha=0.1)` |
| Split | 80 % train / 20 % test, stratified |

---

## Results

| Metric   | Value       |
|----------|-------------|
| Accuracy | ~95–99%     |
| Algorithm | Multinomial Naive Bayes |
| Classes  | Spam / Not Spam |

---

## Visualisations

Training auto-generates five charts inside `screenshots/`:

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart – Spam vs Ham counts |
| `wordcloud_spam.png` | Most common words in spam messages |
| `wordcloud_ham.png` | Most common words in ham messages |
| `confusion_matrix.png` | True vs predicted labels |
| `top_words.png` | Top 20 spam words by frequency |

---

## Flask Web App

The app accepts any message and returns:

- **Prediction** – Spam / Not Spam
- **Confidence** – model certainty %
- **Spam probability**
- **Ham probability**

Example inputs and outputs:

| Message | Result |
|---------|--------|
| Congratulations! You won a free iPhone, click now! |  **Spam** |
| Let's meet at 7 PM today. | **Not Spam** |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset → place at dataset/spam.csv

# 3. Train the model
python train.py

# 4. Launch the web app
python app.py
```

Open your browser at **http://localhost:5000**

---

## Project Structure

```
DAY 7-Spam Email Detection using Naive Bayes/
│
├── train.py            ← Training pipeline + visualisations
├── app.py              ← Flask web application
├── requirements.txt
├── model.pkl           ← Saved Naive Bayes model (after training)
├── vectorizer.pkl      ← Saved TF-IDF vectorizer (after training)
│
├── dataset/
│   └── spam.csv
│
├── templates/
│   └── index.html      ← Web UI
│
└── screenshots/
    ├── class_distribution.png
    ├── wordcloud_spam.png
    ├── wordcloud_ham.png
    ├── confusion_matrix.png
    └── top_words.png
```

---

## Tech Stack

| Tool | Role |
|------|------|
| Python | Core language |
| Pandas / NumPy | Data handling |
| NLTK | Text preprocessing |
| Scikit-Learn | TF-IDF + Naive Bayes |
| Matplotlib / Seaborn | Visualisations |
| WordCloud | Word cloud charts |
| Flask | Web deployment |
| Pickle | Model serialisation |

---

## Concepts Covered

- Natural Language Processing (NLP)
- Text Classification
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Data Cleaning & Stopword Removal
- Stemming (Porter Stemmer)
- Model Evaluation (Accuracy, F1, Confusion Matrix)
- Flask Web Deployment
