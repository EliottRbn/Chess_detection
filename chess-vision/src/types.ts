// Types for Chess Vision App

// Bounding box from YOLO detection (in pixels, 640x640 space)
export interface BoundingBox {
  x: number;      // center x in pixels
  y: number;      // center y in pixels
  width: number;  // width in pixels
  height: number; // height in pixels
}

// Single piece detection
export interface PieceDetection {
  box: BoundingBox;
  classId: number;
  className: string;
  confidence: number;
}

// Board detection result
export interface BoardDetection {
  box: BoundingBox;
  confidence: number;
}

// Chess piece on a square
export type PieceType = 'K' | 'Q' | 'R' | 'B' | 'N' | 'P' | 'k' | 'q' | 'r' | 'b' | 'n' | 'p' | null;

// Board state: 8x8 array, row 0 = rank 8, row 7 = rank 1
export type BoardState = PieceType[][];

// Piece class mapping (12 classes) - CORRECT ORDER from model metadata
// {0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 
//  4: 'black-queen', 5: 'black-rook', 6: 'white-bishop', 7: 'white-king', 
//  8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'}
export const PIECE_CLASSES: Record<number, { name: string; piece: PieceType }> = {
  0: { name: 'black-bishop', piece: 'b' },
  1: { name: 'black-king', piece: 'k' },
  2: { name: 'black-knight', piece: 'n' },
  3: { name: 'black-pawn', piece: 'p' },
  4: { name: 'black-queen', piece: 'q' },
  5: { name: 'black-rook', piece: 'r' },
  6: { name: 'white-bishop', piece: 'B' },
  7: { name: 'white-king', piece: 'K' },
  8: { name: 'white-knight', piece: 'N' },
  9: { name: 'white-pawn', piece: 'P' },
  10: { name: 'white-queen', piece: 'Q' },
  11: { name: 'white-rook', piece: 'R' },
};

// Unicode chess pieces for display
export const PIECE_UNICODE: Record<NonNullable<PieceType>, string> = {
  'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
  'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
};

// Create empty board
export function createEmptyBoard(): BoardState {
  return Array(8).fill(null).map(() => Array(8).fill(null));
}

// Model input size
export const MODEL_INPUT_SIZE = 640;
