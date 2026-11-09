# =============================================================================
# RESTAURANT REVIEWS SENTIMENT ANALYSIS - NLP PIPELINE
# =============================================================================
# 
# This script performs sentiment analysis on restaurant reviews using:
#   - Natural Language Processing (NLP) for text cleaning.
#   - Bag of Words model (CountVectorizer) for feature extraction.
#   - 80/20 Train-Test Split to prevent data leakage.
#   - Grid Search on the Training set to tune hyperparameters.
#   - Final unbiased evaluation on the unseen Test set.
# =============================================================================

import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =============================================================================
# 1. SETUP: DOWNLOAD NLTK DATA (ONLY IF NOT ALREADY DOWNLOADED)
# =============================================================================

def download_nltk_data():
    """Download the stopwords corpus if not already available."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


# =============================================================================
# 2. LOAD THE DATASET
# =============================================================================

def load_data(filepath):
    """Load the TSV dataset."""
    dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
    print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    return dataset


# =============================================================================
# 3. CLEAN THE REVIEWS (TEXT PREPROCESSING)
# =============================================================================

def clean_reviews(dataset):
    """Clean all reviews and return a corpus of cleaned text."""
    corpus = []
    ps = PorterStemmer()
    
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    stopword_set = set(all_stopwords)
    
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopword_set]
        review = ' '.join(review)
        corpus.append(review)
    
    print(f"Cleaned {len(corpus)} reviews.")
    return corpus


# =============================================================================
# 4. BUILD THE BAG OF WORDS MODEL
# =============================================================================

def build_bag_of_words(corpus, max_features=1500):
    """Convert the corpus into a Bag of Words matrix."""
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()
    print(f"Bag of Words matrix shape: {X.shape}")
    return X, cv


# =============================================================================
# 5. TUNE ON TRAIN SET & EVALUATE ON TEST SET
# =============================================================================

def tune_and_evaluate(model_name, classifier, param_grid, X_train, y_train, X_test, y_test, cv=10):
    """
    Tune a model using GridSearchCV on the Training set, 
    then evaluate it once on the unseen Test set.
    """
    print(f"\nProcessing: {model_name}")
    
    # 1. Tuning (Grid Search on the Training Data ONLY)
    if param_grid:
        grid_search = GridSearchCV(
            estimator=classifier, 
            param_grid=param_grid, 
            scoring="f1",  # We tune using F1-Score
            cv=cv, 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  -> Best Params found on Train Set: {best_params}")
    else:
        # If no grid is provided (e.g., GaussianNB)
        best_model = classifier
        best_model.fit(X_train, y_train)
        best_params = "N/A"
        print("  -> No hyperparameters to tune. Trained with default settings.")
        
    # 2. Final Evaluation on the UNSEEN Test Data
    y_pred = best_model.predict(X_test)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Best Params": str(best_params)
    }


# =============================================================================
# 6. PLOT MODEL COMPARISON
# =============================================================================

def plot_comparison(results, filename="outputs/model_comparison.png"):
    """Plot a bar chart comparing model F1-Scores on the Test set."""
    df = pd.DataFrame(results).sort_values("F1-Score", ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["Model"], df["F1-Score"], color="skyblue", edgecolor="black")
    
    for bar, f1 in zip(bars, df["F1-Score"]):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{f1:.2%}",
            va="center",
            fontsize=11
        )
    
    plt.xlabel("F1-Score")
    plt.title("Final Model Comparison: F1-Score")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nComparison chart saved as: {filename}")


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete NLP pipeline with rigorous Train/Test separation."""
    
    # Step 1: Setup and Data Loading
    download_nltk_data()
    dataset = load_data("data/Restaurant_Reviews.tsv")
    
    # Step 2: Preprocessing
    corpus = clean_reviews(dataset)
    X, cv_vectorizer = build_bag_of_words(corpus, max_features=1500)
    y = dataset.iloc[:, -1].values
    
    # Step 3: THE GOLD STANDARD SPLIT
    # We lock away 20% of the data NOW. The models will never see it during tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Step 4: Define the models AND their hyperparameter grids
    models_and_params = {
        "Gaussian Naive Bayes": {
            "model": GaussianNB(),
            "params": {}  
        },
        "Multinomial Naive Bayes": {
            "model": MultinomialNB(),
            "params": {"alpha": [0.1, 0.5, 1.0]} 
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=0),
            "params": {"C": [0.1, 1, 10]} 
        },
        "Linear SVM": {
            "model": SVC(kernel='linear', random_state=0),
            "params": {"C": [0.1, 1, 10]} 
        },
    }
    
    results = []
    
    # Step 5: Tune on Train, Evaluate on Test
    for name, mp in models_and_params.items():
        result = tune_and_evaluate(
            name, mp["model"], mp["params"], 
            X_train, y_train, X_test, y_test, cv=10
        )
        results.append(result)
    
    # Step 6: Print the Final Summary Table
    print("\n" + "=" * 85)
    print("FINAL RESULTS: EVALUATED ON UNSEEN TEST DATA (20%)")
    print("=" * 85)
    
    summary = pd.DataFrame(results)
    summary = summary.sort_values("F1-Score", ascending=False)
    
    # Format metric columns as percentages
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        summary[col] = summary[col].map("{:.2%}".format)
        
    print(summary[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Best Params"]].to_string(index=False))
    
    # Step 7: Plot the Results
    plot_comparison(results)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()