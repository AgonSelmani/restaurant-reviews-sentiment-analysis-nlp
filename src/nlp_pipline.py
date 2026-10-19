import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


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
    """
    Load the TSV dataset.
    
    Parameters:
        filepath (str): Path to the .tsv file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
    print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    return dataset



# =============================================================================
# 3. CLEAN THE REVIEWS (TEXT PREPROCESSING)
# =============================================================================

def clean_reviews(dataset):
    """
    Clean all reviews and return a corpus of cleaned text.
    
    Parameters:
        dataset (pd.DataFrame): The raw dataset with a 'Review' column.
    
    Returns:
        list: A list of cleaned reviews (the corpus).
    """
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
    """
    Convert the corpus into a Bag of Words matrix.
    
    Parameters:
        corpus (list): List of cleaned reviews.
        max_features (int): Maximum number of unique words to keep.
    
    Returns:
        np.ndarray: The Bag of Words matrix (X).
        CountVectorizer: The fitted vectorizer.
    """
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()
    print(f"Bag of Words matrix shape: {X.shape}")
    return X, cv


# =============================================================================
# 5. TRAIN A GIVEN CLASSIFIER
# =============================================================================

def train_model(X_train, y_train, X_test, classifier):
    """
    Train a given classifier and return its predictions.
    """
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return classifier, y_pred


# =============================================================================
# 6. EVALUATE THE MODEL (SINGLE TRAIN/TEST SPLIT)
# =============================================================================

def evaluate_model(model_name, y_test, y_pred):
    """
    Print the confusion matrix and accuracy score for a single split.
    """
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 50)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {ac:.2%}")
    
    return {"Model": model_name, "Accuracy": ac, "Confusion Matrix": cm}


# =============================================================================
# 7. EVALUATE THE MODEL (K-FOLD CROSS-VALIDATION)
# =============================================================================

def evaluate_with_cv(model_name, classifier, X, y, cv=10):
    """
    Evaluate a model using K-Fold Cross-Validation for robust results.
    """
    print(f"\nRunning {cv}-Fold Cross-Validation for {model_name}...")
    
    # n_jobs=-1 uses all CPU cores to speed up the process
    accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=cv, n_jobs=-1)
    
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    
    print(f"{model_name} CV Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")
    
    return {"Model": model_name, "Mean CV Accuracy": mean_acc, "Std Dev": std_acc}


# =============================================================================
# 8. PLOT MODEL COMPARISON
# =============================================================================

def plot_comparison(results, accuracy_col="Accuracy", filename="outputs/model_comparison.png"):
    """
    Plot a bar chart comparing model accuracies and save it to a file.
    """
    df = pd.DataFrame(results).sort_values(accuracy_col, ascending=True)
    
    plt.figure(figsize=(8, 5))
    bars = plt.barh(df["Model"], df[accuracy_col], color="skyblue", edgecolor="black")
    
    # Add accuracy labels on each bar
    for bar, acc in zip(bars, df[accuracy_col]):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.2%}",
            va="center",
            fontsize=11
        )
    
    plt.xlabel("Accuracy")
    plt.title("Model Comparison: Restaurant Review Sentiment Analysis")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    # Save the figure instead of displaying it
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nComparison chart saved as: {filename}")





# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete NLP pipeline with model comparison and CV."""
    
    # Step 1: Setup and Data Loading
    download_nltk_data()
    dataset = load_data("data/Restaurant_Reviews.tsv")
    
    # Step 2: Preprocessing
    corpus = clean_reviews(dataset)
    X, cv_vectorizer = build_bag_of_words(corpus, max_features=1500)
    y = dataset.iloc[:, -1].values
    
    # Step 3: Single Train/Test Split (For Confusion Matrices)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Step 4: Define the models to compare
    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
        "Linear SVM": SVC(kernel='linear', random_state=0),
    }
    
    results = []
    cv_results = []
    
    # Step 5: Train and Evaluate each model
    for name, classifier in models.items():
        print("\n" + "#" * 60)
        print(f"PROCESSING: {name}")
        print("#" * 60)
        
        # 1. Train on 80% split and evaluate
        classifier, y_pred = train_model(X_train, y_train, X_test, classifier)
        result = evaluate_model(name, y_test, y_pred)
        results.append(result)
        
        # 2. Evaluate on the FULL dataset using 10-Fold CV
        cv_result = evaluate_with_cv(name, classifier, X, y, cv=10)
        cv_results.append(cv_result)
    
    # Step 6: Print Summary Tables
    print("\n" + "=" * 60)
    print("SUMMARY: SINGLE TRAIN/TEST SPLIT")
    print("=" * 60)
    summary = pd.DataFrame(results)[["Model", "Accuracy"]]
    summary = summary.sort_values("Accuracy", ascending=False)
    summary["Accuracy"] = summary["Accuracy"].map("{:.2%}".format)
    print(summary.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SUMMARY: 10-FOLD CROSS-VALIDATION (MORE RELIABLE)")
    print("=" * 60)
    cv_summary = pd.DataFrame(cv_results)
    cv_summary = cv_summary.sort_values("Mean CV Accuracy", ascending=False)
    cv_summary["Mean CV Accuracy"] = cv_summary["Mean CV Accuracy"].map("{:.2%}".format)
    cv_summary["Std Dev"] = cv_summary["Std Dev"].map("{:.2%}".format)
    print(cv_summary.to_string(index=False))
    
    # Step 7: Plot the Results (Using CV results as they are most reliable)
    plot_comparison(cv_results, accuracy_col="Mean CV Accuracy")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()