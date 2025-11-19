import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

def train_xgboost_model(data):
    """
    Entraîne un modèle XGBoost pour la détection de fraude
    """
    # Préparation des données
    data_sample = data.sample(frac=0.1, random_state=42) if len(data) > 100000 else data
    X = data_sample.drop(['Class'], axis=1)
    y = data_sample['Class']
    
    # Split des données
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Configuration du modèle XGBoost
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Gérer le déséquilibre
    )
    
    # Entraînement avec validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Prédictions
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
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
    
    # Importance des features
    feature_importance = model.feature_importances_
    feature_names = X.columns.tolist()
    
    return {
        'model': model,
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
        'feature_importance': dict(zip(feature_names, feature_importance)),
        'data_info': {
            'total_samples': len(data_sample),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'fraud_rate': y.mean()
        }
    }

def create_xgboost_confusion_matrix_plot(confusion_matrix):
    """Crée le graphique de la matrice de confusion pour XGBoost"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Greens', 
               xticklabels=['Normal', 'Fraude'], yticklabels=['Normal', 'Fraude'], ax=ax)
    ax.set_title("Matrice de Confusion - XGBoost")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réel")
    return fig

def create_xgboost_roc_curve_plot(roc_data, roc_auc):
    """Crée le graphique de la courbe ROC pour XGBoost"""
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr = roc_data
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='green', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC - XGBoost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def create_xgboost_precision_recall_plot(pr_data, pr_auc):
    """Crée le graphique de la courbe Précision-Rappel pour XGBoost"""
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_curve, recall_curve = pr_data
    ax.plot(recall_curve, precision_curve, label=f'AUC = {pr_auc:.4f}', color='darkgreen', linewidth=2)
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Précision-Rappel - XGBoost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def create_feature_importance_plot(feature_importance, top_n=20):
    """Crée le graphique d'importance des features"""
    # Trier par importance décroissante
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(features))
    
    bars = ax.barh(y_pos, importances, color='lightgreen', edgecolor='darkgreen')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features les plus importantes - XGBoost')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs sur les barres
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig