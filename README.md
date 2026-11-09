# Restaurant Reviews Sentiment Analysis

A Natural Language Processing (NLP) project that classifies restaurant reviews as positive or negative using the Bag of Words model and compares multiple Machine Learning classifiers.

## Overview

This project demonstrates a complete NLP pipeline, from raw text preprocessing to model training and robust evaluation. It covers text cleaning, stemming, stopword removal, feature extraction using the Bag of Words technique, and a comparative analysis of four different classification algorithms.

The goal is to build a reliable model that can automatically determine the sentiment of a customer review, which can be applied to automate customer feedback monitoring and business intelligence.

## Dataset

- **Name:** Restaurant Reviews
- **Source:** Machine Learning A-Z (Udemy)
- **File:** `data/Restaurant_Reviews.tsv`
- **Format:** Tab-Separated Values (TSV)
- **Size:** 1,000 reviews
- **Features:** `Review` (text), `Liked` (0 for negative, 1 for positive)

## Methodology

The pipeline follows these key steps:

1. **Data Loading:** Loaded the TSV file using `pandas` with `delimiter='\t'` and `quoting=3` to correctly handle text data containing commas and quotes.
2. **Text Preprocessing:** 
   - Removed non-letter characters (punctuation and numbers) using Regular Expressions.
   - Converted all text to lowercase for consistency.
   - Split reviews into individual words.
   - Applied Porter Stemming to reduce words to their root forms (e.g., "loved" to "love").
   - Removed stopwords, except for "not", which is crucial for sentiment analysis.
3. **Feature Extraction:** Used `CountVectorizer` with `max_features=1500` to keep only the 1,500 most frequent words, converting the text into a sparse matrix.
4. **Model Tuning and Evaluation:** To ensure maximum performance, all models were tuned using **Grid Search** (testing different regularization strengths and smoothing parameters). The models were then evaluated using **10-Fold Cross-Validation** to provide a statistically robust estimate of real-world performance. The final ranking is based on the **F1-Score**, which balances Precision (the quality of positive predictions) and Recall (the ability to find all positive reviews).

## Model Comparison and Results

Four classifiers were tuned and evaluated. The final results represent the average metrics across 10-Fold Cross-Validation.

![Model Comparison](outputs/model_comparison.png)

| Model | Accuracy | Precision | Recall | F1-Score | Best Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **80.90%** | **82.38%** | 79.40% | **80.59%** | `C=10` |
| Linear SVM | 79.90% | 81.21% | 78.60% | 79.67% | `C=1` |
| Multinomial Naive Bayes | 78.80% | 78.71% | 79.00% | 78.78% | `alpha=1.0` |
| Gaussian Naive Bayes | 67.70% | 63.30% | 84.60% | 72.35% | N/A |

### Key Findings:
1. **Logistic Regression** achieved the highest F1-Score (80.59%), proving it has the most reliable balance between finding positive reviews and avoiding false alarms. Grid Search found that a regularization strength of `C=10` was optimal.
2. **Gaussian Naive Bayes** achieved a high Recall (84.60%) but a very poor Precision (63.30%). This means it predicted positive for almost every review, resulting in a massive number of false positives. This demonstrates why Accuracy and Recall alone can be misleading, and why F1-Score is the superior metric for ranking.
3. **Multinomial Naive Bayes** outperformed Gaussian Naive Bayes by over 6%, confirming the mathematical theory that MultinomialNB is designed for discrete word counts, while GaussianNB assumes continuous, normally distributed data.

## Project Structure

```text
restaurant-reviews-nlp/
|
|-- data/
|   |-- Restaurant_Reviews.tsv
|
|-- outputs/
|   |-- model_comparison.png
|
|-- src/
|   |-- nlp_pipeline.py
|
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/agonselmani/restaurant-reviews-nlp.git
cd restaurant-reviews-nlp
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Python pipeline script. Note: This script will clean the data, train the models using Grid Search, print evaluation metrics to the console, and automatically save the comparison chart to the `outputs/` folder:
```bash
python src/nlp_pipeline.py
```

## Requirements

The project requires the following Python libraries (available in `requirements.txt`):
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib

## Author

Agon Selmani
- GitHub: [agonselmani](https://github.com/agonselmani)