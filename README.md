# ❤️ CardioRisk AI — Détection & Dépistage du Risque Cardiovasculaire

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=github)](https://yousefharizi.github.io/ML_ModelObj6/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Projet Machine Learning — Méthodologie CRISP-DM (Objectif 6)**  
> Système d'évaluation clinique et d'aide à la décision médicale prédisant le risque d'infarctus ou de maladie coronarienne (`HeartDiseaseorAttack`) à partir de 10 indicateurs cliniques et d'hygiène de vie (CDC BRFSS - 229 781 individus).

---

## 🌐 Liens de l'Application Déployée

* **Site Web en Ligne (GitHub Pages) :** `https://yousefharizi.github.io/ML_ModelObj6/`
* **Application Streamlit Cloud :** `https://mlmodelobj6.streamlit.app`

---

## 🎯 1. Rappel du BO et du DSO

### Business Objective (BO6) — Objectif Métier
Les cardiopathies ischémiques représentent la première cause de mortalité dans le monde.  
* **Finalité :** Dépister précocement les individus à haut risque cardiovasculaire afin de déclencher un parcours de soins préventif (bilan cardiologique, ECG, hygiène de vie) et réduire les urgences hospitalières et la mortalité évitable.

### Data Science Objective (DSO6) — Objectif de Fouille
* **Tâche ML :** Classification binaire supervisée de la variable `HeartDiseaseorAttack` (1 = événement cardiaque déclaré, 0 = sain / aucun antécédent).
* **Jeu de données :** CDC BRFSS 2015 nettoyé et dédoublonné (229 781 individus, 21 variables candidates).
* **Critères de succès :** Score ROC-AUC ≥ 0.80 et forte sensibilité (Recall) sur la classe minoritaire à risque via calibrage du seuil de décision (0.35 - 0.40).

---

## 🔬 2. Modélisation & Évaluation

1. **Feature Engineering :**
   * **Charge Vasculaire (`VascularBurden` / 4) :** Cumul d'hypertension, hypercholestérolémie, antécédent d'AVC et diabète.
   * **Triade Cardiométabolique (`CardioMetabolicTriad` / 3) :** Présence combinée de diabète/prédiabète, HTA et cholestérol élevé.
   * **Jours de Santé Altérée (`HealthDaysSum`) :** Somme des jours de santé physique et mentale dégradée.
   * **Standardisation :** Normalisation `StandardScaler` sur l'IMC et les jours altérés avec bornes quantiles [1% - 99%].
2. **Sélection Hybride des Variables :** Filtre ANOVA + Wrapper RFE + Importance Random Forest $\rightarrow$ **Top 10 variables clés**.
3. **Performances du Modèle :**
   * **Modèle Champion :** `RandomForestClassifier` (optimisé)
   * **Score ROC-AUC :** **0.817**
   * **Exactitude :** **89.2 %** sur 45 957 individus du jeu de test indépendant.

---

## 🚀 3. Structure du Dépôt & Déploiement

```text
├── index.html                           # 🌐 Site Web interactif (GitHub Pages en 1 clic)
├── app.py                               # 🖥️ Application Web Python (Streamlit)
├── heart_disease_deployment.joblib      # 📦 Modèle de production & Scaler sérialisés (18.5 MB)
├── business_understanding_diabetes_obj6_vf.ipynb # 📓 Notebook individuel CRISP-DM complet
├── Presentation_Validation_Objectif6.pptx # 📊 Présentation PowerPoint de soutenance (8 slides)
├── requirements.txt                     # 📋 Dépendances Python pour Streamlit Cloud
└── run_app.bat                          # ⚡ Lanceur local en 1 clic (Windows)
```

---

## 💻 4. Comment exécuter le projet en local

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/YousefHarizi/ML_ModelObj6.git
   cd ML_ModelObj6
   ```

2. **Lancer le site Web Streamlit en local :**
   * Sous Windows : Double-cliquez sur `run_app.bat`
   * Ou via terminal :
     ```bash
     pip install -r requirements.txt
     streamlit run app.py
     ```
   * Rendez-vous sur `http://localhost:8501`.

3. **Ouvrir le site Web statique (HTML5) :**
   * Ouvrez directement `index.html` dans n'importe quel navigateur (Chrome, Firefox, Edge, Safari).

---

## 📈 5. Suivi de la Dérive (MLOps & PSI)

En production, la stabilité des données est surveillée via le **Population Stability Index (PSI)** :
* **PSI < 0.10 :** Stabilité confirmée.
* **0.10 ≤ PSI < 0.20 :** Surveillance renforcée.
* **PSI ≥ 0.20 :** Dérive détectée, ré-entraînement planifié.

---

## 👨‍⚕️ Auteur
Projet réalisé dans le cadre du module de **Machine Learning / Data Mining (CRISP-DM)**.
