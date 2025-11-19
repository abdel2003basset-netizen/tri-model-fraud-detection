# Importations essentielles
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st

# Import des modules
from SVM import train_svm_model, create_confusion_matrix_plot, create_roc_curve_plot, create_precision_recall_plot
from ANN import train_ann_model, create_ann_plots
from XGBoost import (train_xgboost_model, create_xgboost_confusion_matrix_plot, 
                     create_xgboost_roc_curve_plot, create_xgboost_precision_recall_plot, 
                     create_feature_importance_plot)

# Configuration Streamlit
st.set_page_config(page_title="Détection de Fraude", layout="wide")
st.title("🔍 Détection de Fraude Bancaire")
st.write("""
### Analyse des transactions avec Machine Learning
Cette application détecte les opérations suspectes en utilisant différents algorithmes.
""")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("creditcard_Nettoyé.csv")

# Cache pour les modèles
@st.cache_data
def get_svm_results(data):
    return train_svm_model(data)

@st.cache_data
def get_ann_results(data):
    return train_ann_model(data)

@st.cache_data
def get_xgboost_results(data):
    return train_xgboost_model(data)

# Interface principale
data = load_data()

# Sidebar pour les informations sur les données
with st.sidebar:
    st.header("📊 Informations sur les données")
    st.write(f"**Nombre total de transactions:** {len(data):,}")
    st.write(f"**Transactions normales:** {(data['Class'] == 0).sum():,}")
    st.write(f"**Transactions frauduleuses:** {(data['Class'] == 1).sum():,}")
    st.write(f"**Taux de fraude:** {(data['Class'] == 1).mean()*100:.2f}%")
    
    st.divider()
    
    st.header("🤖 Algorithmes disponibles")
    st.write("**SVM** - Support Vector Machine")
    st.write("**ANN** - Artificial Neural Network")
    st.write("**XGBoost** - Extreme Gradient Boosting")

# Section principale
col1, col2 = st.columns([2, 1])

with col1:
    # Affichage des données
    if st.checkbox("Afficher les données brutes"):
        st.dataframe(data.head(100), use_container_width=True)

with col2:
    st.write("### 🚀 Lancer l'analyse")
    
    # Boutons pour les différents algorithmes
    col_svm, col_ann, col_xgb = st.columns(3)
    
    with col_svm:
        if st.button("🔵 SVM", type="primary", use_container_width=True):
            with st.spinner('Entraînement SVM...'):
                try:
                    results = get_svm_results(data)
                    st.success("✅ SVM terminé!")
                    st.session_state.svm_results = results
                    # Nettoyer les autres résultats
                    for key in ['ann_results', 'xgboost_results', 'comparison_mode']:
                        if key in st.session_state:
                            del st.session_state[key]
                except Exception as e:
                    st.error(f"❌ Erreur SVM : {str(e)}")
    
    with col_ann:
        if st.button("🟢 ANN", type="secondary", use_container_width=True):
            with st.spinner('Entraînement ANN...'):
                try:
                    results = get_ann_results(data)
                    st.success("✅ ANN terminé!")
                    st.session_state.ann_results = results
                    # Nettoyer les autres résultats
                    for key in ['svm_results', 'xgboost_results', 'comparison_mode']:
                        if key in st.session_state:
                            del st.session_state[key]
                except Exception as e:
                    st.error(f"❌ Erreur ANN : {str(e)}")
    
    with col_xgb:
        if st.button("🟡 XGBoost", type="secondary", use_container_width=True):
            with st.spinner('Entraînement XGBoost...'):
                try:
                    results = get_xgboost_results(data)
                    st.success("✅ XGBoost terminé!")
                    st.session_state.xgboost_results = results
                    # Nettoyer les autres résultats
                    for key in ['svm_results', 'ann_results', 'comparison_mode']:
                        if key in st.session_state:
                            del st.session_state[key]
                except Exception as e:
                    st.error(f"❌ Erreur XGBoost : {str(e)}")
    
    # Bouton pour comparer tous les modèles
    st.divider()
    if st.button("⚖️ Comparer tous les modèles", use_container_width=True):
        with st.spinner('Entraînement de tous les modèles...'):
            try:
                svm_results = get_svm_results(data)
                ann_results = get_ann_results(data)
                xgboost_results = get_xgboost_results(data)
                
                # Stocker tous les résultats
                st.session_state.svm_results = svm_results
                st.session_state.ann_results = ann_results
                st.session_state.xgboost_results = xgboost_results
                st.session_state.comparison_mode = True
                
                st.success("✅ Comparaison terminée!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la comparaison : {str(e)}")

# Affichage des résultats SVM
if 'svm_results' in st.session_state and 'comparison_mode' not in st.session_state:
    results = st.session_state.svm_results
    
    st.write("## 🔵 Résultats de l'analyse SVM")
    
    # Métriques principales
    st.write("### 📈 Performances du modèle")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Exactitude", f"{results['metrics']['accuracy']:.3f}")
    with col2:
        st.metric("Précision", f"{results['metrics']['precision']:.3f}")
    with col3:
        st.metric("Rappel", f"{results['metrics']['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{results['metrics']['f1']:.3f}")
    
    # Graphiques SVM
    st.write("### 📊 Visualisations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Matrice de Confusion**")
        fig1 = create_confusion_matrix_plot(results['confusion_matrix'])
        st.pyplot(fig1)
    
    with col2:
        st.write("**Courbe ROC**")
        fig2 = create_roc_curve_plot(results['roc_data'], results['metrics']['roc_auc'])
        st.pyplot(fig2)
    
    with col3:
        st.write("**Courbe Précision-Rappel**")
        fig3 = create_precision_recall_plot(results['pr_data'], results['metrics']['pr_auc'])
        st.pyplot(fig3)

# Affichage des résultats ANN
if 'ann_results' in st.session_state and 'comparison_mode' not in st.session_state:
    results = st.session_state.ann_results
    
    st.write("## 🟢 Résultats de l'analyse ANN (Réseau de Neurones)")
    
    # Métriques principales
    st.write("### 📈 Performances du modèle")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Exactitude", f"{results['metrics']['accuracy']:.3f}")
    with col2:
        st.metric("Précision", f"{results['metrics']['precision']:.3f}")
    with col3:
        st.metric("Rappel", f"{results['metrics']['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{results['metrics']['f1']:.3f}")

    # Informations sur le modèle et les données
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🧠 Architecture du modèle")
        st.write(f"**Couches cachées:** {results['model_info']['hidden_layers']}")
        st.write(f"**Fonction d'activation:** {results['model_info']['activation']}")
        st.write(f"**Époques d'entraînement:** {results['model_info']['epochs_trained']}")
        st.write(f"**Taille de batch:** {results['model_info']['batch_size']}")
    
    with col2:
        st.write("### 📋 Informations d'entraînement")
        st.write(f"**Échantillons totaux:** {results['data_info']['total_samples']:,}")
        st.write(f"**Échantillons d'entraînement:** {results['data_info']['train_samples']:,}")
        st.write(f"**Échantillons de validation:** {results['data_info']['val_samples']:,}")
        st.write(f"**Échantillons de test:** {results['data_info']['test_samples']:,}")

    # Métriques avancées
    st.write("### 🎯 Métriques avancées")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AUC-ROC", f"{results['metrics']['roc_auc']:.4f}")
    with col2:
        st.metric("AUC-PR", f"{results['metrics']['pr_auc']:.4f}")
    with col3:
        st.write(f"**Taux de fraude:** {results['data_info']['fraud_rate']*100:.2f}%")
    
    # Graphiques ANN
    st.write("### 📊 Visualisations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Courbes d'apprentissage**")
        fig1 = create_ann_plots(results, 'learning_curves')
        st.pyplot(fig1)
    
    with col2:
        st.write("**Matrice de Confusion**")
        fig2 = create_ann_plots(results, 'confusion_matrix')
        st.pyplot(fig2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Courbe ROC**")
        fig3 = create_ann_plots(results, 'roc_curve')
        st.pyplot(fig3)
    
    with col2:
        st.write("**Courbe Précision-Rappel**")
        fig4 = create_ann_plots(results, 'precision_recall')
        st.pyplot(fig4)

# Affichage des résultats XGBoost
if 'xgboost_results' in st.session_state and 'comparison_mode' not in st.session_state:
    results = st.session_state.xgboost_results
    
    st.write("## 🟡 Résultats de l'analyse XGBoost")
    
    # Métriques principales
    st.write("### 📈 Performances du modèle")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Exactitude", f"{results['metrics']['accuracy']:.3f}")
    with col2:
        st.metric("Précision", f"{results['metrics']['precision']:.3f}")
    with col3:
        st.metric("Rappel", f"{results['metrics']['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{results['metrics']['f1']:.3f}")
    
    # Informations sur les données utilisées
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📋 Informations d'entraînement")
        st.write(f"**Échantillons totaux:** {results['data_info']['total_samples']:,}")
        st.write(f"**Échantillons d'entraînement:** {results['data_info']['train_samples']:,}")
        st.write(f"**Échantillons de validation:** {results['data_info']['val_samples']:,}")
        st.write(f"**Échantillons de test:** {results['data_info']['test_samples']:,}")
    
    with col2:
        st.write("### 🎯 Métriques avancées")
        st.metric("AUC-ROC", f"{results['metrics']['roc_auc']:.4f}")
        st.metric("AUC-PR", f"{results['metrics']['pr_auc']:.4f}")
        st.write(f"**Taux de fraude:** {results['data_info']['fraud_rate']*100:.2f}%")
    
    # Graphiques XGBoost
    st.write("### 📊 Visualisations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Matrice de Confusion**")
        fig1 = create_xgboost_confusion_matrix_plot(results['confusion_matrix'])
        st.pyplot(fig1)
    
    with col2:
        st.write("**Courbe ROC**")
        fig2 = create_xgboost_roc_curve_plot(results['roc_data'], results['metrics']['roc_auc'])
        st.pyplot(fig2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Courbe Précision-Rappel**")
        fig3 = create_xgboost_precision_recall_plot(results['pr_data'], results['metrics']['pr_auc'])
        st.pyplot(fig3)
    
    with col2:
        st.write("**Importance des Features**")
        fig4 = create_feature_importance_plot(results['feature_importance'])
        st.pyplot(fig4)
    
    # Rapport de classification détaillé
    st.write("### 📄 Rapport de classification détaillé")
    st.text(results['classification_report'])

# Mode comparaison
if 'comparison_mode' in st.session_state:
    st.write("## ⚖️ Comparaison des modèles")
    
    svm_results = st.session_state.svm_results
    ann_results = st.session_state.ann_results
    xgboost_results = st.session_state.xgboost_results
    
    # Tableau de comparaison des métriques
    st.write("### 📊 Comparaison des performances")
    
    comparison_df = pd.DataFrame({
        'Métrique': ['Exactitude', 'Précision', 'Rappel', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
        'SVM': [
            svm_results['metrics']['accuracy'],
            svm_results['metrics']['precision'],
            svm_results['metrics']['recall'],
            svm_results['metrics']['f1'],
            svm_results['metrics']['roc_auc'],
            svm_results['metrics']['pr_auc']
        ],
        'ANN': [
            ann_results['metrics']['accuracy'],
            ann_results['metrics']['precision'],
            ann_results['metrics']['recall'],
            ann_results['metrics']['f1'],
            ann_results['metrics']['roc_auc'],
            ann_results['metrics']['pr_auc']
        ],
        'XGBoost': [
            xgboost_results['metrics']['accuracy'],
            xgboost_results['metrics']['precision'],
            xgboost_results['metrics']['recall'],
            xgboost_results['metrics']['f1'],
            xgboost_results['metrics']['roc_auc'],
            xgboost_results['metrics']['pr_auc']
        ]
    })
    
    # Ajouter une colonne "Meilleur"
    comparison_df['Meilleur'] = comparison_df.apply(
        lambda row: 'SVM' if row['SVM'] >= max(row['ANN'], row['XGBoost']) 
        else 'ANN' if row['ANN'] >= row['XGBoost'] 
        else 'XGBoost', axis=1
    )
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Graphiques comparatifs
    st.write("### 📈 Courbes ROC comparatives")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**SVM**")
        fig1 = create_roc_curve_plot(svm_results['roc_data'], svm_results['metrics']['roc_auc'])
        st.pyplot(fig1)
    
    with col2:
        st.write("**ANN**")
        fig2 = create_ann_plots(ann_results, 'roc_curve')
        st.pyplot(fig2)
    
    with col3:
        st.write("**XGBoost**")
        fig3 = create_xgboost_roc_curve_plot(xgboost_results['roc_data'], xgboost_results['metrics']['roc_auc'])
        st.pyplot(fig3)
    
    # Recommandation basée sur les résultats
    st.write("### 🎯 Recommandation")
    
    # Calculer le score moyen pour chaque modèle
    svm_avg = sum([svm_results['metrics'][m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]) / 5
    ann_avg = sum([ann_results['metrics'][m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]) / 5
    xgb_avg = sum([xgboost_results['metrics'][m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]) / 5
    
    best_model = max([('SVM', svm_avg), ('ANN', ann_avg), ('XGBoost', xgb_avg)], key=lambda x: x[1])
    
    st.success(f"🏆 **Modèle recommandé : {best_model[0]}** avec un score moyen de {best_model[1]:.4f}")
    
    if st.button("🗑️ Effacer la comparaison"):
        for key in ['svm_results', 'ann_results', 'xgboost_results', 'comparison_mode']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Bouton de nettoyage général
if any(key in st.session_state for key in ['svm_results', 'ann_results', 'xgboost_results']):
    if st.button("🗑️ Effacer tous les résultats"):
        for key in ['svm_results', 'ann_results', 'xgboost_results', 'comparison_mode']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()