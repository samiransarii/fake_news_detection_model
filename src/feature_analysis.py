import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def load_vectorizer_and_model(vectorizer_path, model_path):
    """
    Load the TF-IDF vectorizer and trained model
    """
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


def analyze_feature_importance(model, vectorizer, top_n=30):
    """
    Analyze feature importance for the model
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get coefficients (feature importance)
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        coefficients = model.feature_importances_
    else:
        print("Model doesn't have accessible feature importances")
        return None

    # Create DataFrame with feature names and coefficients
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": coefficients}
    )

    # Sort by absolute importance
    feature_importance["abs_importance"] = abs(feature_importance["importance"])
    feature_importance = feature_importance.sort_values(
        "abs_importance", ascending=False
    )

    # Get top positive and negative features
    positive_features = feature_importance[feature_importance["importance"] > 0].head(
        top_n
    )
    negative_features = feature_importance[feature_importance["importance"] < 0].head(
        top_n
    )

    # Plot top features
    plt.figure(figsize=(12, 10))

    # Positive features
    plt.subplot(2, 1, 1)
    sns.barplot(x="importance", y="feature", data=positive_features.iloc[::-1])
    plt.title("Top Features Indicating True News")
    plt.tight_layout()

    # Negative features
    plt.subplot(2, 1, 2)
    sns.barplot(x="importance", y="feature", data=negative_features.iloc[::-1])
    plt.title("Top Features Indicating Fake News")
    plt.tight_layout()

    plt.savefig("../results/feature_importance.png")
    plt.close()

    return feature_importance


def create_wordclouds(feature_importance):
    """
    Create word clouds for true and fake news
    """
    # Features for true news (positive coefficients)
    true_features = feature_importance[feature_importance["importance"] > 0]
    true_dict = dict(zip(true_features["feature"], true_features["importance"]))

    # Features for fake news (negative coefficients)
    fake_features = feature_importance[feature_importance["importance"] < 0]
    fake_dict = dict(zip(fake_features["feature"], abs(fake_features["importance"])))

    # Create word cloud for true news
    plt.figure(figsize=(12, 6))
    wc_true = WordCloud(
        width=800, height=400, background_color="white", max_words=200
    ).generate_from_frequencies(true_dict)
    plt.subplot(1, 2, 1)
    plt.imshow(wc_true, interpolation="bilinear")
    plt.axis("off")
    plt.title("Words Associated with True News")

    # Create word cloud for fake news
    wc_fake = WordCloud(
        width=800, height=400, background_color="white", max_words=200
    ).generate_from_frequencies(fake_dict)
    plt.subplot(1, 2, 2)
    plt.imshow(wc_fake, interpolation="bilinear")
    plt.axis("off")
    plt.title("Words Associated with Fake News")

    plt.tight_layout()
    plt.savefig("../results/wordclouds.png")
    plt.close()


def analyze_misclassifications(model, X_test, y_test, test_texts, top_n=10):
    """
    Analyze misclassified examples
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Find misclassified examples
    misclassified = np.where(y_test != y_pred)[0]

    # Create DataFrame with misclassified examples
    misclassified_df = pd.DataFrame(
        {
            "text": test_texts.iloc[misclassified].values,
            "true_label": y_test[misclassified],
            "predicted_label": y_pred[misclassified],
        }
    )

    # Map numerical labels to text
    misclassified_df["true_label"] = misclassified_df["true_label"].map(
        {0: "Fake", 1: "True"}
    )
    misclassified_df["predicted_label"] = misclassified_df["predicted_label"].map(
        {0: "Fake", 1: "True"}
    )

    # Display some misclassified examples
    print(f"\nSample of {min(top_n, len(misclassified_df))} misclassified examples:")
    for i, row in misclassified_df.head(top_n).iterrows():
        print(f"\nExample {i+1}:")
        print(f"True label: {row['true_label']}")
        print(f"Predicted label: {row['predicted_label']}")
        print(f"Text snippet: {row['text'][:200]}...\n")

    # Save misclassified examples
    misclassified_df.to_csv("../results/misclassified_examples.csv", index=False)

    return misclassified_df


if __name__ == "__main__":
    # Load vectorizer and model
    # Update these paths to your saved model and vectorizer
    vectorizer_path = "../models/tfidf_vectorizer.pkl"
    model_path = "../models/logistic_regression_model.pkl"  # Change to your best model

    vectorizer, model = load_vectorizer_and_model(vectorizer_path, model_path)

    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, vectorizer)

    # Create word clouds
    create_wordclouds(feature_importance)

    # Load test data with text
    test_data = pd.read_csv("../data/test_data_with_text.csv")
    X_test = np.load("../data/X_test_tfidf.npy")
    y_test = np.load("../data/y_test.npy")

    # Analyze misclassifications
    misclassified_df = analyze_misclassifications(
        model, X_test, y_test, test_data["text"]
    )

    print("Feature analysis complete!")
