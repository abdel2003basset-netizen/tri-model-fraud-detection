import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

import warnings
warnings.filterwarnings("ignore")

def train_svm_model(data):
    
    # Préparation des données
    data_sample = data.sample(frac=0.1, random_state=42) if len(data) > 100000 else data
    X = data_sample.drop(['Time_scaled', 'Class', 'Amount_scaled'], axis=1)
    y = data_sample['Class']
    
    # Split des données
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Configuration du modèle SVM
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
    model = SVC(probability=True, class_weight='balanced', random_state=42)
    
    # GridSearch
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Prédictions
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Métriques
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    
    # Données pour les graphiques
    cm = confusion_matrix(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)
    
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        },
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr),
        'pr_data': (precision_curve, recall_curve),
        'classification_report': classification_report(y_test, y_test_pred, target_names=['Normal', 'Fraude']),
        'data_info': {
            'total_samples': len(data_sample),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'fraud_rate': y.mean()
        }
    }

def create_confusion_matrix_plot(confusion_matrix):
    """Crée le graphique de la matrice de confusion"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'], ax=ax)
    ax.set_title("Matrice de Confusion")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    return fig

def create_roc_curve_plot(roc_data, roc_auc):
    """Crée le graphique de la courbe ROC"""
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr = roc_data
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='blue', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def create_precision_recall_plot(pr_data, pr_auc):
    """Crée le graphique de la courbe Précision-Rappel"""
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_curve, recall_curve = pr_data
    ax.plot(recall_curve, precision_curve, label=f'AUC = {pr_auc:.4f}', color='orange', linewidth=2)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Précision-Rappel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig