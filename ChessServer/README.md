---
title: Chess Vision - FEN Detection
emoji: ♟️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Chess Vision - FEN Detection API

Détecte les pièces d'échecs sur une photo et retourne la position en notation FEN + le meilleur coup.

## API Endpoint

**POST** `/detect_fen`

### Parameters
- `file`: Image du plateau d'échecs (multipart/form-data)
- `turn`: `w` (blancs) ou `b` (noirs) - qui joue

### Response
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
  "pieces_count": 32,
  "board_detected": true,
  "best_move": "e2e4"
}
```

## Usage

```bash
curl -X POST "https://YOUR-SPACE.hf.space/detect_fen?turn=w" \
  -F "file=@chess_photo.jpg"
```