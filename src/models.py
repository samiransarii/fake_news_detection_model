import numpy as np
import pandas as pd
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train and evaluate a model
    """
    print(f"\nTraining {model_name}...")
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_val_pred = model.predict(X_val)

    # Calculate training time
    training_time = time.time() - start_time

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print(f"{model_name} Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "True"],
        yticklabels=["Fake", "True"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(
        f'../results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    )
    plt.close()

    return {
        "model": model,
        "name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "training_time": training_time,
    }


def compare_models(models_results):
    """
    Compare the performance of multiple models
    """
    # Create DataFrame for comparison
    results_df = pd.DataFrame(models_results)
    results_df = results_df[
        ["name", "accuracy", "precision", "recall", "f1", "training_time"]
    ]

    # Display results
    print("\nModel Comparison:")
    print(results_df)

    # Create bar chart for metrics comparison
    metrics = ["accuracy", "precision", "recall", "f1"]
    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(x="name", y=metric, data=results_df)
        plt.title(f"{metric.capitalize()} Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()

    plt.savefig("../results/model_comparison.png")
    plt.close()

    # Create bar chart for training time
    plt.figure(figsize=(10, 6))
    sns.barplot(x="name", y="training_time", data=results_df)
    plt.title("Training Time Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("../results/training_time_comparison.png")
    plt.close()

    return results_df


def evaluate_best_model(best_model, X_test, y_test):
    """
    Evaluate the best model on the test set
    """
    print(f"\nEvaluating best model ({best_model['name']}) on test set...")

    # Make predictions on test set
    y_test_pred = best_model["model"].predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"Test Results for {best_model['name']}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "True"],
        yticklabels=["Fake", "True"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f'Test Set Confusion Matrix - {best_model["name"]}')
    plt.savefig("../results/test_confusion_matrix.png")
    plt.close()

    # Save test results
    test_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return test_results


if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("../data/X_train_tfidf.npy")
    X_val = np.load("../data/X_val_tfidf.npy")
    X_test = np.load("../data/X_test_tfidf.npy")
    y_train = np.load("../data/y_train.npy")
    y_val = np.load("../data/y_val.npy")
    y_test = np.load("../data/y_test.npy")

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Define models to train
    models = [
        (LogisticRegression(max_iter=1000, C=1.0), "Logistic Regression"),
        (MultinomialNB(), "Naive Bayes"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (LinearSVC(max_iter=10000), "Linear SVM"),
    ]

    # Train and evaluate all models
    models_results = []
    for model, model_name in models:
        result = train_model(model, X_train, y_train, X_val, y_val, model_name)
        models_results.append(result)

    # Compare models
    results_df = compare_models(models_results)

    # Find the best model based on F1 score
    best_model = max(models_results, key=lambda x: x["f1"])
    print(f"\nBest model: {best_model['name']} (F1: {best_model['f1']:.4f})")

    # Evaluate best model on test set
    test_results = evaluate_best_model(best_model, X_test, y_test)

    # Save the best model
    with open(
        f'../models/{best_model["name"].lower().replace(" ", "_")}_model.pkl', "wb"
    ) as f:
        pickle.dump(best_model["model"], f)

    print(
        f"\nBest model saved as: ../models/{best_model['name'].lower().replace(' ', '_')}_model.pkl"
    )
