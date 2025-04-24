# Fake News Detection

An end-to-end machine learning project for detecting fake news articles using natural language processing and classification techniques.

## Project Overview

This project aims to build and evaluate machine learning models for fake news detection. Using the "Fake and Real News Dataset" from Kaggle, we implement a complete pipeline from data preprocessing to model evaluation and deployment.

![Fake News Detection](results/model_comparison.png)

## Dataset

The dataset used in this project is the "Fake and Real News Dataset" from Kaggle, which contains:

- Thousands of labeled articles from reliable and unreliable sources
- Headlines, article text, and publication information
- Binary classification (real vs. fake)

## Project Structure

```
fake_news_detection/
├── data/                      # Dataset storage (dataset from kaggle, both fake and true news dataset)
├── notebooks/                 # Jupyter notebooks for exploration and end-to-end ml-models
├── src/                       # Source code (Needs some refinements, see the notebook code for fully working model)
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── feature_engineering.py # Feature extraction
│   ├── models.py              # ML models implementation
│   ├── evaluation.py          # Metrics and evaluation
│   └── utils.py               # Helper functions
├── models/                    # Saved trained models (we currently our one best performing model)
├── results/                   # Results and visualizations
├── app/                       # Demo application (Still needs some fixes, might crash down due to some minor bugs)
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Features

- **Data preprocessing**: Text cleaning, tokenization, and normalization
- **Feature extraction**: TF-IDF vectorization with n-grams
- **Multiple models**: Logistic Regression, Random Forest, SVM, and Naive Bayes
- **Comprehensive evaluation**: Accuracy, precision, recall, F1 score
- **Feature importance analysis**: Understanding the most indicative terms
- **Misclassification analysis**: Examining difficult cases
- **Interactive demo app**: Test the model on custom text

## Model Performance

Our best performing model achieves:

- Accuracy: ~99.62%
- Precision: ~99.32%
- Recall: ~99.91%
- F1 Score: ~99.61%

![Model Comparison](results/model_comparison_df.png)

## Key Findings

- Random Forest was the most accurate but also the slowest to train.
- Naive Bayes trained extremely fast but had slightly lower scores overall.
- Linear SVM was nearly as strong as Random Forest and was much faster.
- Hand-crafted features like punctuation ration and sentiment scores likely boosted performance by giving the models more context about writing style

## Usage

### Setup

1. Clone the repository:

```
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` directory.

### Running the Pipeline

1. Process the data:

```
python src/data_processing.py
```

2. Train and evaluate models:

```
python src/models.py
```

3. Analyze feature importance:

```
python src/feature_engineering.py
```

### Demo Application

Run the Streamlit app:

```
cd app
streamlit run app.py
```

## Future Improvements

- Use of Deep Learning Models
- Incorporate more advanced NLP techniques (word embeddings, transformers)
- Add more features (sentiment analysis, readability metrics)
- Multilingual News Detection
- Live news Stream Integration
- Deploy the model as a browser extension

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- streamlit
- wordcloud

## Author

Samir Ansari, Krishna Sah

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the dataset
- Dr. Ning Zhang
- Computer Science Department, Fisk University
