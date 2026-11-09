
# RESTAURANT REVIEWS SENTIMENT ANALYSIS - NLP PIPELINE


import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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
    
    # Get stopwords and remove 'not' (because it carries sentiment)
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    stopword_set = set(all_stopwords)  # Use a set for O(1) lookup
    
    for i in range(len(dataset)):
        # Step 1: Remove non-letter characters
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        
        # Step 2: Lowercase
        review = review.lower()
        
        # Step 3: Split into words
        review = review.split()
        
        # Step 4: Stem and remove stopwords
        review = [ps.stem(word) for word in review if word not in stopword_set]
        
        # Step 5: Join back into a string
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
# 5. TUNE AND EVALUATE MODELS
# =============================================================================

def tune_and_evaluate(model_name, classifier, param_grid, X, y, cv=10):
    """
    Tune a model using GridSearchCV, then evaluate it using 
    cross_validate to get Accuracy, Precision, Recall, and F1-Score.
    """
    print(f"\nProcessing: {model_name}")
    
    # 1. Tuning (Grid Search)
    if param_grid:
        grid_search = GridSearchCV(
            estimator=classifier, 
            param_grid=param_grid, 
            scoring="f1", 
            cv=cv, 
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        # If no grid is provided (e.g., GaussianNB)
        best_model = classifier
        best_params = "N/A"
        
    # 2. Final Evaluation (Cross-Validation for all metrics)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(
        best_model, X, y, cv=cv, scoring=scoring_metrics, n_jobs=-1
    )
    
    # Calculate means
    mean_acc = cv_results['test_accuracy'].mean()
    mean_prec = cv_results['test_precision'].mean()
    mean_rec = cv_results['test_recall'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    
    return {
        "Model": model_name,
        "Accuracy": mean_acc,
        "Precision": mean_prec,
        "Recall": mean_rec,
        "F1-Score": mean_f1,
        "Best Params": str(best_params)
    }


# =============================================================================
# 6. PLOT MODEL COMPARISON
# =============================================================================

def plot_comparison(results, filename="outputs/model_comparison.png"):
    """Plot a bar chart comparing model accuracies and save it to a file."""
    # Sort by F1-Score (since it balances precision and recall)
    df = pd.DataFrame(results).sort_values("F1-Score", ascending=True)
    
    plt.figure(figsize=(10, 6))
    
    # We will plot the F1-Score, as it's the best overall metric
    bars = plt.barh(df["Model"], df["F1-Score"], color="skyblue", edgecolor="black")
    
    # Add labels on each bar
    for bar, f1 in zip(bars, df["F1-Score"]):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{f1:.2%}",
            va="center",
            fontsize=11
        )
    
    plt.xlabel("F1-Score")
    plt.title("Final Model Comparison (Tuned via Grid Search + 10-Fold CV)")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nComparison chart saved as: {filename}")


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete NLP pipeline with Grid Search tuning."""
    
    # Step 1: Setup and Data Loading
    download_nltk_data()
    dataset = load_data("data/Restaurant_Reviews.tsv")
    
    # Step 2: Preprocessing
    corpus = clean_reviews(dataset)
    X, cv_vectorizer = build_bag_of_words(corpus, max_features=1500)
    y = dataset.iloc[:, -1].values
    
    # Step 3: Define the models AND their hyperparameter grids
    models_and_params = {
        "Gaussian Naive Bayes": {
            "model": GaussianNB(),
            "params": {}  # No hyperparameters to tune
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
    
    # Step 4: Tune and Evaluate each model
    for name, mp in models_and_params.items():
        result = tune_and_evaluate(name, mp["model"], mp["params"], X, y, cv=10)
        results.append(result)
    
    # Step 5: Print the Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL RESULTS: GRID SEARCH + 10-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    summary = pd.DataFrame(results)
    # Sort by F1-Score to find the overall best model
    summary = summary.sort_values("F1-Score", ascending=False)
    
    # Format metric columns as percentages
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        summary[col] = summary[col].map("{:.2%}".format)
        
    print(summary[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Best Params"]].to_string(index=False))
    
    # Step 6: Plot the Results
    plot_comparison(results)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()