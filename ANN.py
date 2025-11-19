import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve
)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def train_ann_model(data):
    """
    Entraîne un modèle de réseau de neurones pour la détection de fraude
    """
    # Préparation des données
    features = ["Time_scaled"] + [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]
    target = "Class"
    
    X = data[features]
    y = data[target]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Construction du modèle
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement avec Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calcul des métriques
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # Données pour les courbes
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    # Calcul AUC-PR
    from sklearn.metrics import average_precision_score
    metrics['pr_auc'] = float(average_precision_score(y_test, y_pred_proba))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Informations sur le modèle
    model_info = {
        'hidden_layers': '256 → 128 → 64',
        'activation': 'ReLU + Sigmoid',
        'epochs_trained': len(history.history['loss']),
        'batch_size': 64
    }
    
    # Informations sur les données
    data_info = {
        'total_samples': len(data),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'fraud_rate': float(y.mean())
    }
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr),
        'pr_data': (precision, recall),
        'predictions': {'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba},
        'model_info': model_info,
        'data_info': data_info,
        'scaler': scaler
    }

def create_ann_plots(results, plot_type):
    """
    Crée différents types de graphiques pour l'ANN
    """
    plt.style.use('default')
    
    if plot_type == 'learning_curves':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Courbe de perte
        ax1.plot(results['history'].history['loss'], label='Entraînement', color='blue')
        ax1.plot(results['history'].history['val_loss'], label='Validation', color='red')
        ax1.set_title('Évolution de la perte')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Courbe d'exactitude
        ax2.plot(results['history'].history['accuracy'], label='Entraînement', color='blue')
        ax2.plot(results['history'].history['val_accuracy'], label='Validation', color='red')
        ax2.set_title('Évolution de l\'exactitude')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Exactitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    elif plot_type == 'confusion_matrix':
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=results['confusion_matrix'],
            display_labels=['Normal', 'Fraude']
        )
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title('Matrice de Confusion - ANN')
        return fig
    
    elif plot_type == 'roc_curve':
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr = results['roc_data']
        roc_auc = results['metrics']['roc_auc']
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Courbe ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbe ROC - ANN')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        return fig
    
    elif plot_type == 'precision_recall':
        fig, ax = plt.subplots(figsize=(8, 6))
        precision, recall = results['pr_data']
        pr_auc = results['metrics']['pr_auc']
        
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'Courbe PR (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Rappel')
        ax.set_ylabel('Précision')
        ax.set_title('Courbe Précision-Rappel - ANN')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        return fig
    
    else:
        return None