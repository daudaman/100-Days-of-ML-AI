import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import os
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from wordcloud import WordCloud
from collections import Counter

#  NLTK setup 
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

#  helpers 
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def load_dataset(path: str = "dataset/spam.csv") -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # normalise column names – the Kaggle CSV ships with v1/v2 or label/message
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {}
    for col in df.columns:
        if col in ("v1", "label", "class", "category"):
            rename_map[col] = "label"
        elif col in ("v2", "message", "text", "sms"):
            rename_map[col] = "message"
    df = df.rename(columns=rename_map)[["label", "message"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    return df


#  main training pipeline 
def main():
    os.makedirs("screenshots", exist_ok=True)

    
    print("  Day 7 – Spam Detection | Naive Bayes + TF-IDF")
    

    # 1 – Load
    print("\n[1/7] Loading dataset …")
    df = load_dataset()
    print(f"      Shape : {df.shape}")
    print(f"      Labels: {df['label'].value_counts().to_dict()}")

    # 2 – Clean
    print("\n[2/7] Cleaning text …")
    df["clean"] = df["message"].apply(clean_text)
    df["label_enc"] = (df["label"] == "spam").astype(int)

    # 3 – Visualisations
    print("\n[3/7] Generating visualisations …")
    _plot_class_distribution(df)
    _plot_wordcloud(df, "spam")
    _plot_wordcloud(df, "ham")
    _plot_top_words(df)

    # 4 – TF-IDF
    print("\n[4/7] Vectorising with TF-IDF …")
    vectorizer = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean"])
    y = df["label_enc"]
    print(f"      Feature matrix: {X.shape}")

    # 5 – Split
    print("\n[5/7] Train / test split (80 / 20) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6 – Train
    print("\n[6/7] Training Multinomial Naive Bayes …")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    # 7 – Evaluate
    print("\n[7/7] Evaluating …")
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy : {acc * 100:.2f} %")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    _plot_confusion_matrix(y_test, y_pred)

    # Save artefacts
    with open("model.pkl",      "wb") as f: pickle.dump(model,      f)
    with open("vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
    print("\n  model.pkl and vectorizer.pkl saved.")
    print("\n  Training complete – run `python app.py` to launch the web app.")


#  plot helpers 
def _plot_class_distribution(df):
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(["Ham (Not Spam)", "Spam"], counts[["ham", "spam"]].values,
                  color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts[["ham", "spam"]].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title("Class Distribution – Spam vs Ham", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("screenshots/class_distribution.png", dpi=150)
    plt.close()
    print("       class_distribution.png")


def _plot_wordcloud(df, label):
    text = " ".join(df[df["label"] == label]["clean"])
    cmap = "Reds" if label == "spam" else "Greens"
    wc   = WordCloud(width=800, height=400, background_color="white",
                     colormap=cmap, max_words=150).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    title = "Spam" if label == "spam" else "Ham (Not Spam)"
    plt.title(f"Word Cloud – {title} Messages", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"screenshots/wordcloud_{label}.png", dpi=150)
    plt.close()
    print(f"       wordcloud_{label}.png")


def _plot_top_words(df, n=20):
    spam_text = " ".join(df[df["label"] == "spam"]["clean"])
    words  = [w for w in spam_text.split() if len(w) > 2]
    common = Counter(words).most_common(n)
    labels_, vals = zip(*common)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.cm.YlOrRd(np.linspace(0.4, 0.9, n))
    ax.barh(list(reversed(labels_)), list(reversed(vals)),
            color=list(reversed(palette)), edgecolor="white")
    ax.set_title(f"Top {n} Most Frequent Words in Spam", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("screenshots/top_words.png", dpi=150)
    plt.close()
    print("       top_words.png")


def _plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Ham", "Spam"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("screenshots/confusion_matrix.png", dpi=150)
    plt.close()
    print("       confusion_matrix.png")


if __name__ == "__main__":
    main()
