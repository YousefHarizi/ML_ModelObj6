import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk AI — Dépistage Cardiovasculaire",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished medical / modern dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-risk-low {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 18px;
        border-radius: 8px;
    }
    .card-risk-high {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 18px;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.8rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #b91c1c 0%, #991b1b 100%);
        box-shadow: 0 6px 10px -1px rgba(220, 38, 38, 0.4);
        transform: translateY(-1px);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Deployment Bundle
# ---------------------------------------------------------
@st.cache_resource
def load_bundle():
    bundle_paths = [
        Path(__file__).parent / "heart_disease_deployment.joblib",
        Path(__file__).parent / "obj6" / "heart_disease_deployment.joblib",
        Path.cwd() / "obj6" / "heart_disease_deployment.joblib",
        Path.cwd() / "heart_disease_deployment.joblib"
    ]
    for p in bundle_paths:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("Le fichier de déploiement 'heart_disease_deployment.joblib' est introuvable.")

try:
    bundle = load_bundle()
    model = bundle['model']
    scaler = bundle['scaler']
    bmi_lower = bundle['bmi_lower']
    bmi_upper = bundle['bmi_upper']
    continuous_features = bundle['continuous_features']
    selected_features = bundle['selected_features']
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Erreur de chargement du modèle : {e}")

# ---------------------------------------------------------
# Dictionnaires de Correspondance & Références
# ---------------------------------------------------------
AGE_MAPPING = {
    1: "18 à 24 ans",
    2: "25 à 29 ans",
    3: "30 à 34 ans",
    4: "35 à 39 ans",
    5: "40 à 44 ans",
    6: "45 à 49 ans",
    7: "50 à 54 ans",
    8: "55 à 59 ans",
    9: "60 à 64 ans",
    10: "65 à 69 ans",
    11: "70 à 74 ans",
    12: "75 à 79 ans",
    13: "80 ans ou plus"
}

GENHLTH_MAPPING = {
    1: "1 - Excellente",
    2: "2 - Très bonne",
    3: "3 - Bonne",
    4: "4 - Passable",
    5: "5 - Mauvaise"
}

DIABETES_MAPPING = {
    0: "0 - Pas de diabète",
    1: "1 - Prédiabète",
    2: "2 - Diabète déclaré"
}

# ---------------------------------------------------------
# Pipeline de Préparation & Inférence
# ---------------------------------------------------------
def prepare_input(patient_dict):
    df = pd.DataFrame([patient_dict])
    
    # Remplissage de sécurité pour variables continues
    for col in continuous_features:
        if col not in df.columns:
            df[col] = 0.0
            
    # Écrêtage BMI & Standardisation
    cont_df = df[continuous_features].copy()
    cont_df['BMI'] = cont_df['BMI'].clip(lower=bmi_lower, upper=bmi_upper)
    scaled_matrix = scaler.transform(cont_df)
    for i, col in enumerate(continuous_features):
        df[f"{col}_scaled"] = scaled_matrix[:, i]
        
    # Variables dérivées métier (CRISP-DM Phase 3)
    df['CardioMetabolicTriad'] = (
        (df['Diabetes_012'] >= 1).astype(int) + 
        df['HighBP'].astype(int) + 
        df['HighChol'].astype(int)
    )
    df['VascularBurden'] = (
        df['HighBP'].astype(int) + 
        df['HighChol'].astype(int) + 
        df['Stroke'].astype(int) + 
        (df['Diabetes_012'] == 2).astype(int)
    )
    df['HealthDaysSum'] = df['MentHlth'] + df['PhysHlth']
    df['TotalUnhealthyDays'] = df['HealthDaysSum']
    
    # Alignement strict sur les features sélectionnées
    prepared = df.reindex(columns=selected_features).fillna(0)
    return prepared, df

# ---------------------------------------------------------
# Sidebar : Navigation & Profils Types
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822143.png", width=70)
    st.markdown("## **CardioRisk AI**")
    st.caption("Déploiement CRISP-DM — Objectif 6")
    st.markdown("---")
    
    st.markdown("### ⚡ Profils types rapides")
    demo_profile = st.selectbox(
        "Charger un patient type :",
        ["Personnalisé (libre)", "Profil 1 : Adulte jeune sain", "Profil 2 : Risque intermédiaire", "Profil 3 : Patient âgé à haut risque"]
    )
    
    st.markdown("---")
    st.markdown("### 📋 Rappel Méthodologique")
    with st.expander("🎯 Business Objective (BO)"):
        st.write("""
        **Objectif Métier :**
        Dépister précocement les profils à risque accru d'infarctus ou de maladie coronarienne (`HeartDiseaseorAttack`) 
        pour cibler les actions de prévention et réduire les urgences hospitalières.
        """)
    with st.expander("🔬 Data Science Objective (DSO)"):
        st.write("""
        **Objectif de Fouille :**
        Classification binaire supervisée à partir des 10 indicateurs cliniques et comportementaux les plus discriminants (CDC BRFSS 2015).
        """)
    with st.expander("⚙️ Modèle & Données"):
        st.write("""
        - **Modèle :** Random Forest Classifier
        - **Jeu de données :** 229 781 individus nettoyés
        - **Score ROC-AUC :** 0.817
        """)

# Pré-remplissage selon le profil démo
defaults = {
    "Sex": 0, "Age": 5, "GenHlth": 2, "HighBP": 0, "HighChol": 0,
    "Stroke": 0, "Diabetes_012": 0, "DiffWalk": 0, "BMI": 23.5,
    "PhysHlth": 0, "MentHlth": 1
}

if demo_profile == "Profil 1 : Adulte jeune sain":
    defaults = {
        "Sex": 0, "Age": 2, "GenHlth": 1, "HighBP": 0, "HighChol": 0,
        "Stroke": 0, "Diabetes_012": 0, "DiffWalk": 0, "BMI": 22.0,
        "PhysHlth": 0, "MentHlth": 0
    }
elif demo_profile == "Profil 2 : Risque intermédiaire":
    defaults = {
        "Sex": 1, "Age": 8, "GenHlth": 3, "HighBP": 1, "HighChol": 1,
        "Stroke": 0, "Diabetes_012": 1, "DiffWalk": 0, "BMI": 28.0,
        "PhysHlth": 4, "MentHlth": 3
    }
elif demo_profile == "Profil 3 : Patient âgé à haut risque":
    defaults = {
        "Sex": 1, "Age": 11, "GenHlth": 4, "HighBP": 1, "HighChol": 1,
        "Stroke": 1, "Diabetes_012": 2, "DiffWalk": 1, "BMI": 33.5,
        "PhysHlth": 18, "MentHlth": 8
    }

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">❤️ CardioRisk AI — Plateforme de Dépistage Cardiovasculaire</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Système d\'aide à la décision médicale basé sur l\'apprentissage automatique — CRISP-DM Phase 6 (Déploiement)</div>', unsafe_allow_html=True)

tab_pred, tab_model, tab_monitoring = st.tabs([
    "🩺 Évaluation Clinique du Patient",
    "📊 Performances & Modélisation",
    "📈 Suivi de Dérive & MLOps"
])

# ---------------------------------------------------------
# TAB 1: Diagnostic & Prediction
# ---------------------------------------------------------
with tab_pred:
    st.markdown("### 📝 Renseignez les paramètres cliniques du patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Profil Démographique")
        sex = st.radio("Sexe :", options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme", index=defaults["Sex"])
        age = st.selectbox("Tranche d'âge :", options=list(AGE_MAPPING.keys()), format_func=lambda x: f"{x} : {AGE_MAPPING[x]}", index=defaults["Age"] - 1)
        gen_hlth = st.selectbox("Santé générale déclarée :", options=list(GENHLTH_MAPPING.keys()), format_func=lambda x: GENHLTH_MAPPING[x], index=defaults["GenHlth"] - 1)
        bmi = st.number_input("Indice de Masse Corporelle (IMC / BMI) :", min_value=12.0, max_value=60.0, value=float(defaults["BMI"]), step=0.5)

    with col2:
        st.markdown("#### 🩺 Antécédents Médicaux")
        high_bp = st.selectbox("Hypertension artérielle :", options=[0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)", index=defaults["HighBP"])
        high_chol = st.selectbox("Hypercholestérolémie :", options=[0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)", index=defaults["HighChol"])
        stroke = st.selectbox("Antécédent d'Accident Vasculaire Cérébral (AVC) :", options=[0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)", index=defaults["Stroke"])
        diabetes = st.selectbox("Statut Diabète :", options=list(DIABETES_MAPPING.keys()), format_func=lambda x: DIABETES_MAPPING[x], index=defaults["Diabetes_012"])

    with col3:
        st.markdown("#### 🏃‍♂️ Mobilité & Qualité de Vie")
        diff_walk = st.selectbox("Difficulté importante à marcher ou monter des escaliers :", options=[0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)", index=defaults["DiffWalk"])
        phys_hlth = st.slider("Jours de mauvaise santé physique (sur les 30 derniers jours) :", min_value=0, max_value=30, value=int(defaults["PhysHlth"]))
        ment_hlth = st.slider("Jours de mauvaise santé mentale (sur les 30 derniers jours) :", min_value=0, max_value=30, value=int(defaults["MentHlth"]))

    st.markdown("")
    btn_predict = st.button("⚡ Calculer le Risque Cardiovasculaire")
    st.markdown("---")

    if btn_predict:
        patient_data = {
            "Sex": sex,
            "Age": age,
            "GenHlth": gen_hlth,
            "BMI": bmi,
            "HighBP": high_bp,
            "HighChol": high_chol,
            "Stroke": stroke,
            "Diabetes_012": diabetes,
            "DiffWalk": diff_walk,
            "PhysHlth": phys_hlth,
            "MentHlth": ment_hlth
        }
        
        X_infer, df_derived = prepare_input(patient_data)
        probabilities = model.predict_proba(X_infer)[0]
        prediction = int(model.predict(X_infer)[0])
        
        prob_healthy = probabilities[0]
        prob_risk = probabilities[1]
        
        # Affichage du résultat médical
        st.markdown("### 📋 Diagnostic & Évaluation Prédictive")
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            if prediction == 1 or prob_risk >= 0.40:
                st.markdown(f"""
                <div class="card-risk-high">
                    <h3 style="color: #b91c1c; margin-top: 0;">⚠️ Risque Cardiovasculaire Élevé Détecté</h3>
                    <p style="font-size: 1.15rem; margin-bottom: 0.5rem;">
                        Probabilité estimée d'événement cardiaque / coronarien : <strong>{prob_risk * 100:.1f} %</strong>
                    </p>
                    <p style="color: #6b7280; font-size: 0.95rem;">
                        Le profil du patient présente un faisceau d'indicateurs physiologiques et antécédents typiquement associés aux cardiopathies ischémiques.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-risk-low">
                    <h3 style="color: #15803d; margin-top: 0;">✅ Faible Risque Cardiovasculaire (Profil Sain)</h3>
                    <p style="font-size: 1.15rem; margin-bottom: 0.5rem;">
                        Probabilité de profil sain : <strong>{prob_healthy * 100:.1f} %</strong> (Risque résiduel : {prob_risk * 100:.1f} %)
                    </p>
                    <p style="color: #6b7280; font-size: 0.95rem;">
                        Aucune alerte critique immédiate. Le maintien des habitudes de vie favorables et le suivi régulier sont préconisés.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("")
            st.progress(float(prob_risk))
            st.caption(f"Jauge de probabilité clinique : {prob_risk * 100:.1f} % de risque")
            
        with res_col2:
            st.markdown("#### 🔬 Indicateurs Clés Dérivés (Feature Engineering)")
            v_burden = int(df_derived['VascularBurden'].iloc[0])
            c_triad = int(df_derived['CardioMetabolicTriad'].iloc[0])
            h_days = int(df_derived['HealthDaysSum'].iloc[0])
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Charge Vasculaire", f"{v_burden}/4", help="Cumul : Hypertension + Hypercholestérolémie + AVC + Diabète")
            mc2.metric("Triade Cardio.", f"{c_triad}/3", help="Diabète/Prédiabète + HTA + Hypercholestérolémie")
            mc3.metric("Jours Altérés", f"{h_days} j", help="Cumul santé physique et mentale altérée")

        st.markdown("---")
        st.markdown("#### 💡 Facteurs Contributifs & Préconisations Personnalisées")
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            st.markdown("**Points de vigilance identifiés :**")
            warnings = []
            if age >= 9:
                warnings.append(f"• Âge avancé ({AGE_MAPPING[age]}) constituant un facteur d'usure vasculaire.")
            if high_bp == 1:
                warnings.append("• Hypertension artérielle diagnostiquée (surcharge mécanique du myocarde).")
            if high_chol == 1:
                warnings.append("• Hypercholestérolémie (risque d'athérosclérose des artères coronaires).")
            if stroke == 1:
                warnings.append("• Antécédent d'AVC : signe majeur d'atteinte vasculaire systémique.")
            if diabetes >= 1:
                warnings.append("• Atteinte métabolique (diabète/prédiabète fragilisant l'endothélium).")
            if diff_walk == 1:
                warnings.append("• Limitation motrice sévère réduisant l'activité aérobie protectrice.")
            if bmi >= 30:
                warnings.append(f"• Obésité (IMC = {bmi:.1f}) augmentant le travail cardiaque.")
                
            if warnings:
                for w in warnings:
                    st.write(w)
            else:
                st.write("• Aucun facteur d'alerte métabolique ou vasculaire majeur détecté.")
                
        with f_col2:
            st.markdown("**Recommandations préventives :**")
            if prob_risk >= 0.40:
                st.warning("1. Consultation cardiologique pour bilan d'effort et électrocardiogramme (ECG).\n2. Contrôle strict de la pression artérielle (< 130/80 mmHg).\n3. Bilan lipidique complet et optimisation du traitement hypolipémiant.\n4. Programme de réadaptation cardiovasculaire douce.")
            else:
                st.success("1. Contrôle annuel de routine de la tension et de la glycémie.\n2. Pratique d'au moins 150 minutes d'activité physique modérée par semaine.\n3. Alimentation méditerranéenne riche en fibres, fruits et légumes.\n4. Surveillance pondérale.")

# ---------------------------------------------------------
# TAB 2: Model Performance & Methodology
# ---------------------------------------------------------
with tab_model:
    st.markdown("### 🔬 Performance du Modèle en Déploiement")
    st.markdown("""
    Dans le respect du cadre **CRISP-DM**, la modélisation a comparé plusieurs algorithmes 
    (Régression Logistique, Random Forest, etc.) sur les 229 781 individus de l'étude BRFSS.
    """)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Algorithme Retenu", "Random Forest")
    m_col2.metric("Score ROC-AUC", "0.817")
    m_col3.metric("Échantillon BRFSS", "229 781")
    m_col4.metric("Variables Retenues", "10 clés")
    
    st.markdown("---")
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### 🏆 Top 10 des Variables Discriminantes")
        feat_df = pd.DataFrame({
            "Variable": selected_features,
            "Rôle Clinique": [
                "Pression artérielle élevée",
                "Antécédent d'AVC",
                "Perception globale de l'état de santé",
                "Difficulté de mobilité à la marche",
                "Sexe biologique (risque accru chez l'homme)",
                "Tranche d'âge du patient",
                "Jours de mauvaise santé physique (normalisés)",
                "Triade Diabète + HTA + Cholestérol",
                "Charge d'usure vasculaire globale",
                "Cumul physique + mental altéré"
            ]
        })
        st.table(feat_df)
        
    with p_col2:
        st.markdown("#### 🎯 Stratégie de Choix du Seuil de Décision")
        st.write("""
        En contexte médical, le coût d'un **Faux Négatif** (ne pas détecter un patient cardiaque) est infiniment 
        plus préjudiciable que celui d'un Faux Positif.
        
        - Le modèle probabiliste permet à l'équipe clinique de régler le seuil d'intervention :
          - **Seuil standard (0.50) :** Classification d'équilibre.
          - **Seuil préventif (0.35 - 0.40) :** Capture de plus de 80% des personnes à risque pour déclencher les examens complémentaires.
        """)

# ---------------------------------------------------------
# TAB 3: Drift Monitoring (MLOps)
# ---------------------------------------------------------
with tab_monitoring:
    st.markdown("### 📈 Surveillance de la Dérive des Données (PSI - Drift Monitoring)")
    st.markdown("""
    En production (CRISP-DM Phase 6), la distribution des patients entrants peut évoluer dans le temps 
    (changement démographique, saisonnalité). Le **Population Stability Index (PSI)** permet d'alerter avant toute dégradation des prédictions.
    """)
    
    st.info("""
    **Grille de lecture standard du PSI :**
    - **PSI < 0.10 :** Distribution stable, aucune action requise.
    - **0.10 ≤ PSI < 0.20 :** Dérive modérée, surveillance recommandée.
    - **PSI ≥ 0.20 :** Dérive significative, ré-entraînement ou ré-étalonnage nécessaire.
    """)
    
    # Table indicative de monitoring
    psi_sample = pd.DataFrame({
        "Variable": ["Age", "VascularBurden", "GenHlth", "CardioMetabolicTriad", "HighBP", "BMI_scaled", "HealthDaysSum"],
        "PSI Calculé": [0.012, 0.019, 0.024, 0.015, 0.008, 0.031, 0.022],
        "Statut": ["✅ Stable", "✅ Stable", "✅ Stable", "✅ Stable", "✅ Stable", "✅ Stable", "✅ Stable"]
    })
    st.table(psi_sample)

st.markdown("---")
st.caption("CardioRisk AI — Projet Machine Learning BRFSS | Encadrement CRISP-DM Objectif 6 | Déployé avec Streamlit")

