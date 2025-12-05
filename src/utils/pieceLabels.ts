/**
 * Piece labels mapping for YOLO model (12 classes)
 * Order matches the training data.yaml exactly
 */

export const PIECE_LABELS = [
  "black-bishop",   // 0
  "black-king",     // 1
  "black-knight",   // 2
  "black-pawn",     // 3
  "black-queen",    // 4
  "black-rook",     // 5
  "white-bishop",   // 6
  "white-king",     // 7
  "white-knight",   // 8
  "white-pawn",     // 9
  "white-queen",    // 10
  "white-rook",     // 11
] as const;

export type PieceLabel = (typeof PIECE_LABELS)[number];

/**
 * Unicode symbols for each piece (chess notation style)
 */
export const PIECE_SYMBOLS: Record<PieceLabel, string> = {
  "black-bishop": "♝",
  "black-king": "♚",
  "black-knight": "♞",
  "black-pawn": "♟",
  "black-queen": "♛",
  "black-rook": "♜",
  "white-bishop": "♗",
  "white-king": "♔",
  "white-knight": "♘",
  "white-pawn": "♙",
  "white-queen": "♕",
  "white-rook": "♖",
};

/**
 * Short labels for display (FEN notation)
 */
export const PIECE_SHORT: Record<PieceLabel, string> = {
  "black-bishop": "b",
  "black-king": "k",
  "black-knight": "n",
  "black-pawn": "p",
  "black-queen": "q",
  "black-rook": "r",
  "white-bishop": "B",
  "white-king": "K",
  "white-knight": "N",
  "white-pawn": "P",
  "white-queen": "Q",
  "white-rook": "R",
};

/**
 * Colors for drawing bounding boxes (black pieces = darker, white pieces = lighter)
 */
export const PIECE_COLORS: Record<PieceLabel, string> = {
  "black-bishop": "#2E7D32",  // dark green
  "black-king": "#C62828",    // dark red
  "black-knight": "#6A1B9A",  // dark purple
  "black-pawn": "#37474F",    // dark gray
  "black-queen": "#AD1457",   // dark pink
  "black-rook": "#1565C0",    // dark blue
  "white-bishop": "#81C784",  // light green
  "white-king": "#EF5350",    // light red
  "white-knight": "#BA68C8",  // light purple
  "white-pawn": "#B0BEC5",    // light gray
  "white-queen": "#F48FB1",   // light pink
  "white-rook": "#64B5F6",    // light blue
};

/**
 * Get label from classId
 */
export function getPieceLabel(classId: number): PieceLabel | null {
  if (classId < 0 || classId >= PIECE_LABELS.length) return null;
  return PIECE_LABELS[classId];
}

/**
 * Get display info for a piece
 */
export function getPieceDisplay(classId: number) {
  const label = getPieceLabel(classId);
  if (!label) return { label: "unknown", symbol: "?", short: "?", color: "#FF0000" };
  
  return {
    label,
    symbol: PIECE_SYMBOLS[label],
    short: PIECE_SHORT[label],
    color: PIECE_COLORS[label],
  };
}

