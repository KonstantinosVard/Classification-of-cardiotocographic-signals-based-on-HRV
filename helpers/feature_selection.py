import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score, make_scorer, recall_score

RANDOM_STATE = 42
specificity = make_scorer(recall_score, pos_label=0)

def remove_correlated_features(X, threshold=0.7):
    """
    Remove highly correlated features
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr = [i for i in upper.columns if upper.loc[i, col] > threshold]
        if high_corr:
            to_drop.update(high_corr)
    
    X_filtered = X.drop(columns=list(to_drop), errors='ignore')
    return X_filtered

def select_features_statistical(X, y, score_func, k=10):
    """
    Select features using statistical methods (SelectKBest)
    """
    skb = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    skb.fit(X, y)
    selected_mask = skb.get_support()
    selected_features = X.columns[selected_mask].tolist()
    return selected_features

def evaluate_with_statistical_selection(X, y, score_func, n_features, 
                                        corr_threshold, estimator, cv_splits=3):
    """
    Evaluate model with statistical feature selection
    """
    try:
        # Remove correlated features first
        X_filtered = remove_correlated_features(X, threshold=corr_threshold)
        
        # Select top k features
        selected_features = select_features_statistical(
            X_filtered, y, score_func, k=n_features
        )
        
        if len(selected_features) == 0:
            return {
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'balanced_accuracy_mean': np.nan,
                'balanced_accuracy_std': np.nan,
                'recall_mean': np.nan,
                'recall_std': np.nan,
                'specificity_mean': np.nan,
                'specificity_std': np.nan,
                'n_features_selected': 0,
                'selected_features': ''
            }
        
        X_selected = X[selected_features]
        
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        
        # Compute scores
        accuracy_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring='accuracy', n_jobs=-1
        )
        balanced_accuracy_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring='balanced_accuracy', n_jobs=-1
        )
        recall_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring='recall', n_jobs=-1
        )
        specificity_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring=specificity, n_jobs=-1
        )
        
        results = {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'balanced_accuracy_mean': np.mean(balanced_accuracy_scores),
            'balanced_accuracy_std': np.std(balanced_accuracy_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'specificity_mean': np.mean(specificity_scores),
            'specificity_std': np.std(specificity_scores),
            'n_features_selected': len(selected_features),
            'selected_features': ','.join(selected_features[:10])
        }
        
        return results
        
    except Exception as e:
        # Statistical selection always worked in practise
        print(f"Error in statistical selection: {e}")
        return {
            'accuracy_mean': np.nan,
            'accuracy_std': np.nan,
            'balanced_accuracy_mean': np.nan,
            'balanced_accuracy_std': np.nan,
            'recall_mean': np.nan,
            'recall_std': np.nan,
            'specificity_mean': np.nan,
            'specificity_std': np.nan,
            'n_features_selected': 0,
            'selected_features': ''
        }

def evaluate_with_sequential_selection(X, y, method_name, method_config, 
                                       estimator, n_features, corr_threshold, 
                                       cv_splits=3):
    """
    Evaluate model with sequential feature selection
    """
    try:
        # Remove correlated features first
        X_filtered = remove_correlated_features(X, threshold=corr_threshold)
        
        if X_filtered.shape[1] < n_features:
            n_features = max(2, X_filtered.shape[1])
        
        # Create selector
        selector_class = method_config['class']
        selector_params = method_config['params'].copy()
        
        # Handle n_features parameter
        if method_name == 'RFECV':
            # RFECV determines n_features automatically
            selector_params['min_features_to_select'] = max(2, n_features // 2)
        else:
            selector_params['n_features_to_select'] = n_features
        
        # Create selector with estimator
        selector = selector_class(estimator=estimator, **selector_params)
        
        # Fit selector on filtered features
        selector.fit(X_filtered, y)
        
        # Get selected features
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            selected_features = X_filtered.columns[selected_mask].tolist()
            n_selected = len(selected_features)
        else:
            selected_features = []
            n_selected = 0
        
        # Evaluate on selected features
        X_selected = X[selected_features]
        
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        
        # Compute scores
        accuracy_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring='accuracy', n_jobs=-1
        )
        balanced_accuracy_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring='balanced_accuracy', n_jobs=-1
        )
        recall_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring='recall', n_jobs=-1
        )
        specificity_scores = cross_val_score(
            estimator, X_selected, y, cv=skf, 
            scoring=specificity, n_jobs=-1
        )
        
        results = {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'balanced_accuracy_mean': np.mean(balanced_accuracy_scores),
            'balanced_accuracy_std': np.std(balanced_accuracy_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'specificity_mean': np.mean(specificity_scores),
            'specificity_std': np.std(specificity_scores),
            'n_features_selected': n_selected,
            'selected_features': ','.join(selected_features[:10])  # Store first 10
        }
        
        return results
        
    except Exception as e:
        # It was for 'SVC', 'KNN', 'MLP' with rfe
        print(f"Error in {method_name}: {e}")
        return {
            'accuracy_mean': np.nan,
            'accuracy_std': np.nan,
            'balanced_accuracy_mean': np.nan,
            'balanced_accuracy_std': np.nan,
            'recall_mean': np.nan,
            'recall_std': np.nan,
            'specificity_mean': np.nan,
            'specificity_std': np.nan,
            'n_features_selected': 0,
            'selected_features': ''
        }