import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_model_and_vectorizer(model_path, vectorizer_path):
    """
    Load the saved model and vectorizer
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def preprocess_and_predict(text, model, vectorizer):
    """
    Preprocess the input text and make a prediction
    """
    # Transform the text using the vectorizer
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vectorized)[0]

    # Get prediction probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_vectorized)[0]
        confidence = proba[prediction]
    else:
        # For models without predict_proba
        decision = model.decision_function(text_vectorized)[0]
        confidence = abs(decision)

    return prediction, confidence


def display_feature_importance(text, model, vectorizer):
    """
    Display the most important features for the given text
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Transform the text
    text_vectorized = vectorizer.transform([text])

    # Get the non-zero features in the transformed text
    non_zero_features = text_vectorized.nonzero()[1]

    # Get the feature values
    feature_values = text_vectorized.data

    # Get the feature importance
    if hasattr(model, "coef_"):
        feature_importance = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
    else:
        st.warning("Feature importance visualization not available for this model.")
        return

    # Create a DataFrame with the features, their values, and importance
    features_df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in non_zero_features],
            "value": feature_values,
            "importance": [feature_importance[i] for i in non_zero_features],
            "weighted_importance": [
                feature_importance[i] * feature_values[j]
                for j, i in enumerate(non_zero_features)
            ],
        }
    )

    # Sort by absolute weighted importance
    features_df["abs_weighted_importance"] = abs(features_df["weighted_importance"])
    features_df = features_df.sort_values("abs_weighted_importance", ascending=False)

    # Display top 10 features
    st.subheader("Top Features Contributing to Classification")

    # Create two columns for positive and negative features
    col1, col2 = st.columns(2)

    with col1:
        st.write("Features suggesting True news:")
        positive_features = features_df[features_df["weighted_importance"] > 0].head(5)
        if not positive_features.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(
                x="weighted_importance",
                y="feature",
                data=positive_features.iloc[::-1],
                ax=ax,
            )
            ax.set_title("True News Indicators")
            st.pyplot(fig)
        else:
            st.write("No significant positive features found.")

    with col2:
        st.write("Features suggesting Fake news:")
        negative_features = features_df[features_df["weighted_importance"] < 0].head(5)
        if not negative_features.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(
                x="weighted_importance",
                y="feature",
                data=negative_features.iloc[::-1],
                ax=ax,
            )
            ax.set_title("Fake News Indicators")
            st.pyplot(fig)
        else:
            st.write("No significant negative features found.")


def display_model_metrics(metrics_path):
    """
    Display model performance metrics
    """
    try:
        metrics = pd.read_csv(metrics_path)

        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{metrics['accuracy'][0]:.2%}")
        col2.metric("Precision", f"{metrics['precision'][0]:.2%}")
        col3.metric("Recall", f"{metrics['recall'][0]:.2%}")
        col4.metric("F1 Score", f"{metrics['f1'][0]:.2%}")

        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm = np.array(
            [
                [metrics["true_negative"][0], metrics["false_positive"][0]],
                [metrics["false_negative"][0], metrics["true_positive"][0]],
            ]
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fake", "True"],
            yticklabels=["Fake", "True"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Model metrics file not found.")


def main():
    st.title("Fake News Detector")
    st.write(
        """
    This application uses machine learning to detect fake news articles. 
    Enter a news article text below to analyze it.
    """
    )

    # Load model and vectorizer
    try:
        model_path = "../models/best_model.pkl"
        vectorizer_path = "../models/tfidf_vectorizer.pkl"
        model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

        # Add model info
        st.sidebar.header("Model Information")
        st.sidebar.write(f"Model Type: {type(model).__name__}")

        # Add model metrics
        metrics_path = "../results/model_metrics.csv"
        display_model_metrics(metrics_path)

    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please check the file paths.")
        return

    # Input text area
    st.subheader("Input News Article")
    news_text = st.text_area("Paste the news article text here:", height=200)

    # Make prediction when submit button is clicked
    if st.button("Analyze") and news_text:
        with st.spinner("Analyzing..."):
            # Make prediction
            prediction, confidence = preprocess_and_predict(
                news_text, model, vectorizer
            )

            # Display result
            st.subheader("Analysis Result")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == 1:
                    st.success("This article appears to be TRUE news.")
                else:
                    st.error("This article appears to be FAKE news.")

            with result_col2:
                st.metric("Confidence", f"{confidence:.2%}")

            # Display feature importance
            display_feature_importance(news_text, model, vectorizer)

    # Add project information to sidebar
    st.sidebar.header("About this Project")
    st.sidebar.write(
        """ This Fake News Detection project was created for the Machine Learning course. 
        The model was trained on the "Fake and Real News Dataset" from Kaggle, which contains thousands of labeled real and fake news articles.
    
        The system analyzes text patterns and linguistic features to identify potentially misleading content.
        """
    )

    # Add disclaimer
    st.sidebar.header("Disclaimer")
    st.sidebar.write(
        """
    This tool is for educational purposes only. Always verify information from reliable sources 
    and practice critical thinking when consuming news media.
    """
    )


if __name__ == "__main__":
    main()
