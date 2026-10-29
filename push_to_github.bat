@echo off
setlocal enabledelayedexpansion
title Publication sur GitHub - CardioRisk AI (deployweb)
echo =====================================================================
echo       CardioRisk AI - Envoi du dossier deployweb vers GitHub
echo =====================================================================
echo.
cd /d "%~dp0"

echo [1/3] Preparation du repertoire Git local...
if not exist ".git" (
    git init
    git branch -M main
)
git add .
git commit -m "Initial commit - Deploiement Web et Modele Objectif 6"
echo Fichiers prets pour l'envoi.
echo.

echo [2/3] Liaison avec votre depot GitHub :
echo ---------------------------------------------------------------------
echo Veuillez coller ci-dessous l'URL de votre depot GitHub :
echo Exemple : https://github.com/VotrePseudo/cardiorisk-ai.git
echo ---------------------------------------------------------------------
echo.
set /p REPO_URL="URL du depot GitHub : "

if "%REPO_URL%"=="" (
    echo Aucune URL fournie. Relancez le script en collant votre URL.
    pause
    exit /b
)

git remote remove origin 2>nul
git remote add origin %REPO_URL%
echo.

echo [3/3] Envoi des fichiers (git push)...
git push -u origin main --force

if %errorlevel% equ 0 (
    echo.
    echo =====================================================================
    echo FELICITATIONS ! Vos fichiers sont en ligne sur GitHub !
    echo.
    echo DERNIERE ETAPE POUR METTRE LE SITE WEB EN LIGNE (30 secondes) :
    echo 1. Allez sur votre depot GitHub.
    echo 2. Cliquez sur "Settings" (en haut a droite).
    echo 3. Cliquez sur "Pages" (dans le menu a gauche).
    echo 4. Sous "Build and deployment", choisissez Branch: "main" et "/(root)".
    echo 5. Cliquez sur "Save".
    echo.
    echo Votre site sera en direct a l'adresse :
    echo https://^<votre-pseudo^>.github.io/^<nom-du-repo^>/
    echo =====================================================================
) else (
    echo.
    echo Erreur lors du push. Verifiez vos identifiants GitHub.
)
echo.
pause

