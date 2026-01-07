#!/bin/bash

# Script simple pour déployer le serveur sur Hugging Face
# Usage: ./deploy_server.sh

BRANCH="deploy-hf"

echo "🚀 Début du déploiement..."

# 1. Nettoyage préventif
git branch -D $BRANCH 2>/dev/null

# 2. Création de la branche propre (Server seulement)
echo "✂️  Préparation du serveur..."
git subtree split --prefix ChessServer -b $BRANCH

# 3. Envoi forcé
echo "⬆️  Envoi vers Hugging Face..."
git push https://huggingface.co/spaces/raphalp/chess-fen-detection $BRANCH:main --force

# 4. Nettoyage final
git branch -D $BRANCH
echo "✅ Terminé !"
