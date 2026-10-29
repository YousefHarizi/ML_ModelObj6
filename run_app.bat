@echo off
title CardioRisk AI - Deploiement Web Objectif 6
echo ========================================================
echo    CardioRisk AI - Deploiement CRISP-DM Objectif 6
echo    Demarrage du serveur Web Streamlit...
echo ========================================================
cd /d "%~dp0"
start "" http://localhost:8501
python -m streamlit run app.py
pause

