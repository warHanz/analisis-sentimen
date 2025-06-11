from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from functools import lru_cache
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def create_vectorizer(max_features=10000, ngram_range=(1, 3), min_df=2):
    """Create and cache a TF-IDF vectorizer with optimized parameters"""
    logger.info(f"Creating TF-IDF vectorizer with max_features={max_features}, ngram_range={ngram_range}, min_df={min_df}")
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True  # Menggunakan log untuk meningkatkan bobot frekuensi
    )

def prepare_data(texts, labels, test_size=0.2, random_state=42, max_features=10000, ngram_range=(1, 3), min_df=2):
    """
    Split data into training and testing sets and convert texts to TF-IDF features.
    """
    if texts.empty or labels.empty or len(texts) != len(labels):
        logger.error("Invalid input: texts or labels are empty or mismatched")
        raise ValueError("Texts and labels must be non-empty and have the same length")

    logger.info("Preparing data: splitting and transforming to TF-IDF")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # Menjaga distribusi kelas
    )
    
    vectorizer = create_vectorizer(max_features, ngram_range, min_df)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    logger.info(f"Data prepared: {X_train_tfidf.shape[0]} training samples, {X_test_tfidf.shape[0]} test samples")
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Train and evaluate SVM and Naive Bayes models with optimized hyperparameter tuning.
    """
    if X_train_tfidf.shape[0] == 0 or X_test_tfidf.shape[0] == 0:
        logger.error("Empty training or test data")
        raise ValueError("Training and test data must not be empty")

    results = {}

    # Optimized SVM with GridSearchCV
    logger.info("Tuning and training SVM model")
    try:
        svm_param_grid = {
            'C': [0.1, 0.5, 1, 2, 5],  # Lebih banyak opsi regularisasi
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.5],
            'class_weight': [None, 'balanced']  # Menangani ketidakseimbangan kelas
        }
        
        svm_grid = GridSearchCV(
            SVC(probability=True),
            svm_param_grid,
            cv=5,  # Meningkatkan cross-validation
            n_jobs=-1,
            scoring='accuracy',
            verbose=1
        )
        svm_grid.fit(X_train_tfidf, y_train)
        
        results['SVM'] = {
            'pred': svm_grid.predict(X_test_tfidf),
            'acc': accuracy_score(y_test, svm_grid.predict(X_test_tfidf)),
            'report': classification_report(y_test, svm_grid.predict(X_test_tfidf), output_dict=True, zero_division=0),
            'best_params': svm_grid.best_params_
        }
        logger.info(f"SVM best params: {svm_grid.best_params_} Accuracy: {results['SVM']['acc']:.2f}")
    except Exception as e:
        logger.error(f"Error training SVM model: {e}")
        results['SVM'] = {'error': str(e)}

    # Optimized Naive Bayes with GridSearchCV
    logger.info("Tuning and training Naive Bayes model")
    try:
        nb_param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0],  # Lebih banyak opsi smoothing
            'fit_prior': [True, False],  # Opsi untuk prior probability
            'class_prior': [None, [0.3, 0.3, 0.4]]  # Menangani ketidakseimbangan kelas
        }
        
        nb_grid = GridSearchCV(
            MultinomialNB(),
            nb_param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            verbose=1
        )
        nb_grid.fit(X_train_tfidf, y_train)
        
        results['Naive Bayes'] = {
            'pred': nb_grid.predict(X_test_tfidf),
            'acc': accuracy_score(y_test, nb_grid.predict(X_test_tfidf)),
            'report': classification_report(y_test, nb_grid.predict(X_test_tfidf), output_dict=True, zero_division=0),
            'best_params': nb_grid.best_params_
        }
        logger.info(f"Naive Bayes best params: {nb_grid.best_params_} Accuracy: {results['Naive Bayes']['acc']:.2f}")
    except Exception as e:
        logger.error(f"Error training Naive Bayes model: {e}")
        results['Naive Bayes'] = {'error': str(e)}

    return results