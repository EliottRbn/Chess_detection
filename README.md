# ♟️ Chess Vision AI

Une solution complète de digitalisation d'échiquier utilisant l'Intelligence Artificielle. Prenez une photo de votre plateau d'échecs réel, et obtenez instantanément la position FEN numérique et le meilleur coup à jouer.

**Architecture Monorepo unifiée :**
- **📱 Frontend :** Application mobile (React Native / Expo)
- **🧠 Backend :** Serveur d'inférence IA (Python / FastAPI / YOLOv8)

---

## 🚀 Fonctionnalités Clés

- **Détection Instantanée** : Analyse d'une photo d'échiquier en < 1 seconde.
- **IA Hybride (Niveau 3)** : 
  - Détection d'objets (YOLO8m/11n) pour localiser les pièces.
  - Segmentation (YOLO-seg) pour détourer le plateau.
  - Classificateur spécialisé (MobileNetV3) pour valider les pièces ambiguës (ex: Roi vs Reine).
  - Validation logique par règles d'échecs (1 roi max, pions max, etc.).
- **Calcul du Coup** : Intégration de Stockfish pour suggérer le meilleur coup.
- **Interface Mobile** : UX moderne, visualisation 2D, gestion du tour (Blancs/Noirs).

---

## 📂 Structure du Projet

```
Chess/
├── 📱 ChessExpo/           # Application Mobile (React Native)
│   ├── app/                # Pages (Expo Router)
│   ├── assets/             # Images & Icônes
│   └── src/                # Composants & Logique business
│
└── 🧠 ChessServer/         # Backend IA (Python)
    ├── src/                # Code source serveur
    │   ├── server_fen.py   # API FastAPI principale
    │   └── scripts/        # Outils entrainement/debug
    ├── models/             # Poids des modèles IA (.pt/.onnx)
    └── assets/             # Données de test
```

---

## 🛠️ Installation & Démarrage

### 1️⃣ Pré-requis
- Node.js & npm
- Python 3.9+
- Expo Go (sur votre téléphone)

### 2️⃣ Lancer le Serveur IA (Backend)

```bash
cd ChessServer

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
cd src
python server_fen.py
```
*Le serveur démarrera sur `http://0.0.0.0:7860`*

### 3️⃣ Lancer l'Application Mobile (Frontend)

```bash
cd ChessExpo

# Installer les dépendances
npm install

# Lancer Expo
npx expo start
```
*Scannez le QR code avec l'application "Expo Go" sur votre téléphone.*

**Note :** Assurez-vous que votre téléphone et votre ordinateur sont sur le même réseau WiFi. Modifiez l'IP du serveur dans `ChessExpo/app/(tabs)/index.tsx` si nécessaire.

---

## 🧠 Détails Techniques IA (Backend)

Le pipeline de reconnaissance se déroule en 5 étapes :

1.  **Plateau (Segmentation)** : Un modèle YOLOv8-seg détecte les 4 coins de l'échiquier et applique une transformation de perspective (Homographie) pour obtenir une vue "à plat" (640x640).
2.  **Pièces (Détection)** : Un modèle YOLOv8m (ou YOLO11n) détecte toutes les pièces sur le plateau redressé.
3.  **Filtrage (Confiance & IoU)** : Suppression des détections multiples ou superposées.
4.  **Classification Secondaire (Validation)** : Si une détection a une confiance faible (< 65%), un classificateur CNN rapide (MobileNetV3) vérifie la pièce (ex: confirme que c'est bien un Roi et pas une Reine).
5.  **Logique Échecs** : Algorithme de validation qui corrige les erreurs impossibles (ex: "3 Rois détectés" -> garde le plus probable).
6.  **FEN & Engine** : Conversion de la grille 8x8 détectée en chaîne FEN, puis envoi à Stockfish pour analyse.

---

## 👥 Auteurs & Crédits

Projet réalisé dans le cadre du cours "Conception des systèmes - Prototypage rapide" à l'IPSA.

**Datasets utilisés :**
- Roboflow Universe (Chess Pieces by Bruno Leonardo)
- Génération synthétique (Data Augmentation)

---
© 2026 - Chess Vision Team