# Data handling, utilities and Text preprocessing
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLP
from sklearn.model_selection import train_test_split

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# Define file paths for the dataset
true_news_path = "../data/True.csv"
fake_news_path = "../data/Fake.csv"


def load_dataset(true_news_path, fake_news_path):
    """
    Load and combine true and fake news datasets.

    - Adds binary labels: 1 for true news, 0 for fake news
    - Combines and shuffles both datasets
    """
    # Load datasets from CSV files
    true_news = pd.read_csv(true_news_path)
    fake_news = pd.read_csv(fake_news_path)

    # Assign labels: 1-> true news, 0-> for fake news
    true_news["label"] = 1
    fake_news["label"] = 0

    # Combine the datasets
    full_dataset = pd.concat([true_news, fake_news], ignore_index=True)

    # Shuffle the combined dataset to randomize it
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return full_dataset


def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Rejoin tokens
        text = " ".join(tokens)

        return text
    else:
        return ""


def prepare_data(df, text_column="text"):
    """
    Preprocess the text data and split into train/validation/test sets
    """
    # Preprocess text
    print("Preprocessing text data...")
    df["processed_text"] = df[text_column].apply(preprocess_text)

    # Split data into features and target
    X = df["processed_text"]
    y = df["label"]

    # Split into train, validation, and test sets (70%, 15%, 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(
        f"Train shape: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_features(X_train, X_val, X_test, max_features=5000):
    """
    Extract TF-IDF features from the text data
    """
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features, min_df=5, max_df=0.8, ngram_range=(1, 2)
    )

    # Fit and transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transform validation and test data
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"TF-IDF feature dimensions: {X_train_tfidf.shape[1]}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer


if __name__ == "__main__":
    # Paths to dataset files - update these with your actual file paths
    true_news_path = "../data/True.csv"
    fake_news_path = "../data/Fake.csv"

    # load dataset
    df = load_dataset(true_news_path, fake_news_path)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # Extract features
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
        X_train, X_val, X_test
    )

    print("Data processing complete!")
