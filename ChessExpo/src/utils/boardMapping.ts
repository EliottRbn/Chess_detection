import { PieceDetection, BoardState, createEmptyBoard, PIECE_CLASSES, MODEL_INPUT_SIZE } from '../types';
import { BoardCorners } from './yoloParser';

/**
 * Matrix 3x3 for Homography
 */
type Matrix3x3 = [
  number, number, number,
  number, number, number,
  number, number, number
];

/**
 * Compute the homography matrix that maps srcPoints to dstPoints
 * srcPoints: [x0, y0, x1, y1, x2, y2, x3, y3] (TL, TR, BR, BL)
 * dstPoints: [0, 0, 1, 0, 1, 1, 0, 1] (Unit square)
 */
function computeHomography(src: { x: number; y: number }[]): Matrix3x3 | null {
  if (src.length !== 4) return null;

  // Destination points: (0,0), (1,0), (1,1), (0,1)
  const x0 = src[0].x, y0 = src[0].y;
  const x1 = src[1].x, y1 = src[1].y;
  const x2 = src[2].x, y2 = src[2].y;
  const x3 = src[3].x, y3 = src[3].y;

  const u0 = 0, v0 = 0;
  const u1 = 1, v1 = 0;
  const u2 = 1, v2 = 1;
  const u3 = 0, v3 = 1;

  // Gaussian elimination to solve for H
  // We have 8 equations for 8 unknowns (h33 = 1)
  // Ah = b
  
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const x = src[i].x;
    const y = src[i].y;
    const u = [0, 1, 1, 0][i];
    const v = [0, 0, 1, 1][i];

    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }

  // Solve Ax = b using Gaussian elimination
  const h = solveLinearSystem(A, b);
  if (!h) return null;

  return [
    h[0], h[1], h[2],
    h[3], h[4], h[5],
    h[6], h[7], 1
  ];
}

/**
 * Solve linear system Ax = b using Gaussian elimination
 */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  // Augment A with b
  const M = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // Pivot
    let maxEl = Math.abs(M[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > maxEl) {
        maxEl = Math.abs(M[k][i]);
        maxRow = k;
      }
    }

    if (maxEl === 0) return null; // Singular matrix

    // Swap rows
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    // Normalize pivot row
    for (let k = i + 1; k < n + 1; k++) {
      M[i][k] /= M[i][i];
    }
    M[i][i] = 1;

    // Eliminate other rows
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = M[k][i];
        for (let j = i; j < n + 1; j++) {
          M[k][j] -= factor * M[i][j];
        }
      }
    }
  }

  return M.map(row => row[n]);
}

/**
 * Apply homography to a point
 */
function applyHomography(h: Matrix3x3, x: number, y: number): { x: number; y: number } {
  const w = h[6] * x + h[7] * y + h[8];
  const px = (h[0] * x + h[1] * y + h[2]) / w;
  const py = (h[3] * x + h[4] * y + h[5]) / w;
  return { x: px, y: py };
}

/**
 * Map piece detections to board squares using perspective transformation (Homography)
 */
export function mapPiecesToBoard(
  pieces: PieceDetection[],
  corners: BoardCorners | null
): BoardState {
  const board = createEmptyBoard();

  if (pieces.length === 0) {
    console.log('[BoardMapping] No pieces to map');
    return board;
  }

  // If no corners detected, fallback to simple grid
  if (!corners) {
    console.log('[BoardMapping] No board corners, using fallback mapping');
    return mapPiecesToBoardSimple(pieces);
  }

  console.log('[BoardMapping] Using Homography mapping with corners:');
  console.log(`  TL: (${corners.topLeft.x.toFixed(1)}, ${corners.topLeft.y.toFixed(1)})`);
  console.log(`  TR: (${corners.topRight.x.toFixed(1)}, ${corners.topRight.y.toFixed(1)})`);
  console.log(`  BR: (${corners.bottomRight.x.toFixed(1)}, ${corners.bottomRight.y.toFixed(1)})`); // Order: TL, TR, BR, BL
  console.log(`  BL: (${corners.bottomLeft.x.toFixed(1)}, ${corners.bottomLeft.y.toFixed(1)})`);

  // Compute Homography Matrix
  // Order MUST be TL, TR, BR, BL to match (0,0), (1,0), (1,1), (0,1)
  const srcPoints = [
    corners.topLeft,
    corners.topRight,
    corners.bottomRight,
    corners.bottomLeft
  ];
  
  const H = computeHomography(srcPoints);
  
  if (!H) {
    console.error('[BoardMapping] Failed to compute homography matrix');
    return mapPiecesToBoardSimple(pieces);
  }

  // Map each piece to a square using homography
  for (const piece of pieces) {
    // Get piece position (bottom center of box is best for standing pieces)
    const px = piece.box.x;
    const py = piece.box.y + piece.box.height * 0.4; // 40% down from center (near bottom)

    // Transform to board coordinates [0,1]
    const boardPos = applyHomography(H, px, py);
    
    // Check if within board (with slight margin)
    const margin = 0.1;
    if (boardPos.x < -margin || boardPos.x > 1 + margin || boardPos.y < -margin || boardPos.y > 1 + margin) {
      console.log(`[BoardMapping] ${piece.className} at (${px.toFixed(0)}, ${py.toFixed(0)}) -> (${boardPos.x.toFixed(2)}, ${boardPos.y.toFixed(2)}) OUT`);
      continue;
    }

    // Convert from [0,1] to board indices [0-7]
    const col = Math.floor(boardPos.x * 8);
    const row = Math.floor(boardPos.y * 8);

    // Clamp to valid range
    const validCol = Math.max(0, Math.min(7, col));
    const validRow = Math.max(0, Math.min(7, row));

    // Get piece type
    const pieceInfo = PIECE_CLASSES[piece.classId];
    if (pieceInfo) {
      // If multiple pieces map to same square, keep the one closest to center of square?
      // For now, just overwrite (or maybe check confidence)
      // Better: check if square is empty or if new piece has higher confidence
      if (board[validRow][validCol] === null) {
        board[validRow][validCol] = pieceInfo.piece;
        const square = `${String.fromCharCode(97 + validCol)}${8 - validRow}`;
        console.log(`[BoardMapping] ${pieceInfo.name} -> ${square} (u=${boardPos.x.toFixed(2)}, v=${boardPos.y.toFixed(2)})`);
      }
    }
  }

  // Count pieces placed
  let count = 0;
  for (const row of board) {
    for (const cell of row) {
      if (cell !== null) count++;
    }
  }
  console.log(`[BoardMapping] Total: ${count} pieces mapped`);

  return board;
}

/**
 * Simple fallback mapping when no board is detected
 */
function mapPiecesToBoardSimple(pieces: PieceDetection[]): BoardState {
  const board = createEmptyBoard();
  // ... (same as before)
  // Find bounding box of all pieces
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const piece of pieces) {
    minX = Math.min(minX, piece.box.x);
    maxX = Math.max(maxX, piece.box.x);
    minY = Math.min(minY, piece.box.y);
    maxY = Math.max(maxY, piece.box.y);
  }

  const width = maxX - minX || MODEL_INPUT_SIZE;
  const height = maxY - minY || MODEL_INPUT_SIZE;

  for (const piece of pieces) {
    const relX = (piece.box.x - minX) / width;
    const relY = (piece.box.y - minY) / height;

    const col = Math.floor(relX * 8);
    const row = Math.floor(relY * 8);

    const validCol = Math.max(0, Math.min(7, col));
    const validRow = Math.max(0, Math.min(7, row));

    const pieceInfo = PIECE_CLASSES[piece.classId];
    if (pieceInfo && board[validRow][validCol] === null) {
      board[validRow][validCol] = pieceInfo.piece;
    }
  }

  return board;
}

export function squareToAlgebraic(row: number, col: number): string {
  const file = String.fromCharCode(97 + col); // a-h
  const rank = 8 - row; // 1-8
  return `${file}${rank}`;
}
