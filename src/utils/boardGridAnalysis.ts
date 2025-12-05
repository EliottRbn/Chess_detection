/**
 * Board Grid Analysis
 * Precise calculation of 64 squares from 4 corners
 * with color-based validation and orientation detection
 */

import { 
  getPerspectiveTransform, 
  perspectiveTransform, 
  type Point 
} from "./perspectiveTransform";

const INPUT_SIZE = 640;

export type GridSquare = {
  file: number;  // 0-7 (a-h)
  rank: number;  // 0-7 (1-8 from bottom)
  center: Point;
  corners: [Point, Point, Point, Point]; // TL, TR, BR, BL
  isLightSquare: boolean;
};

export type BoardGrid = {
  squares: GridSquare[][];  // [rank][file]
  corners: [Point, Point, Point, Point];  // Board corners: TL, TR, BR, BL
  perspectiveMatrix: number[][];
  inversePerspectiveMatrix: number[][];
  orientation: "white-bottom" | "black-bottom";
};

/**
 * Compute the 64 squares of the board from 4 corners
 * Corners should be ordered: top-left, top-right, bottom-right, bottom-left
 */
export function computeBoardGrid(
  corners: Array<{ x: number; y: number }>
): BoardGrid | null {
  if (corners.length < 4) return null;
  
  const orderedCorners = corners.slice(0, 4) as [Point, Point, Point, Point];
  const [tl, tr, br, bl] = orderedCorners;
  const quadArea = polygonArea(orderedCorners);
  if (!Number.isFinite(quadArea) || quadArea < 1) {
    return null;
  }
  
  // Compute perspective transform from board to normalized square
  const srcCorners: [Point, Point, Point, Point] = [tl, tr, br, bl];
  const dstCorners: [Point, Point, Point, Point] = [
    { x: 0, y: 0 },
    { x: INPUT_SIZE, y: 0 },
    { x: INPUT_SIZE, y: INPUT_SIZE },
    { x: 0, y: INPUT_SIZE },
  ];
  
  const M = getPerspectiveTransform(srcCorners, dstCorners);
  const Minv = getPerspectiveTransform(dstCorners, srcCorners);
  
  // Compute all 64 squares
  const squareSize = INPUT_SIZE / 8;
  const squares: GridSquare[][] = [];
  
  for (let rank = 0; rank < 8; rank++) {  // rank 0 = top of image = rank 8 in chess
    const row: GridSquare[] = [];
    for (let file = 0; file < 8; file++) {
      // Square corners in normalized (top-down) space
      const normTL = { x: file * squareSize, y: rank * squareSize };
      const normTR = { x: (file + 1) * squareSize, y: rank * squareSize };
      const normBR = { x: (file + 1) * squareSize, y: (rank + 1) * squareSize };
      const normBL = { x: file * squareSize, y: (rank + 1) * squareSize };
      const normCenter = { x: (file + 0.5) * squareSize, y: (rank + 0.5) * squareSize };
      
      // Transform back to image space
      const imgTL = perspectiveTransform(normTL, Minv);
      const imgTR = perspectiveTransform(normTR, Minv);
      const imgBR = perspectiveTransform(normBR, Minv);
      const imgBL = perspectiveTransform(normBL, Minv);
      const imgCenter = perspectiveTransform(normCenter, Minv);
      
      // Chess board pattern: (file + rank) % 2 == 0 means light square
      // But we need to determine orientation first
      const isLightSquare = (file + rank) % 2 === 0;
      
      row.push({
        file,
        rank: 7 - rank,  // Convert to chess rank (0 = rank 1, bottom)
        center: imgCenter,
        corners: [imgTL, imgTR, imgBR, imgBL],
        isLightSquare,
      });
    }
    squares.push(row);
  }
  
  return {
    squares,
    corners: orderedCorners,
    perspectiveMatrix: M,
    inversePerspectiveMatrix: Minv,
    orientation: "white-bottom",  // Default, can be updated by color validation
  };
}

function polygonArea(points: [Point, Point, Point, Point]): number {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    area += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(area) / 2;
}

/**
 * Get the square at a given image coordinate
 */
export function getSquareAtPoint(
  point: Point,
  grid: BoardGrid
): GridSquare | null {
  // Transform point to normalized space
  const normPoint = perspectiveTransform(point, grid.perspectiveMatrix);
  console.log(`    [getSquareAtPoint] Input: (${point.x.toFixed(1)}, ${point.y.toFixed(1)}) → Normalized: (${normPoint.x.toFixed(1)}, ${normPoint.y.toFixed(1)})`);
  
  if (!Number.isFinite(normPoint.x) || !Number.isFinite(normPoint.y)) {
    console.log(`    [getSquareAtPoint] ❌ Non-finite normalized coords`);
    return null;
  }
  
  const squareSize = INPUT_SIZE / 8;
  const file = Math.floor(normPoint.x / squareSize);
  const rank = Math.floor(normPoint.y / squareSize);
  
  console.log(`    [getSquareAtPoint] Computed file=${file}, rank=${rank} (squareSize=${squareSize})`);
  
  if (file < 0 || file > 7 || rank < 0 || rank > 7) {
    console.log(`    [getSquareAtPoint] ❌ Out of range`);
    return null;
  }
  
  if (!grid.squares[rank]) {
    console.log(`    [getSquareAtPoint] ❌ Grid rank missing`);
    return null;
  }
  
  return grid.squares[rank][file];
}

/**
 * Convert file/rank to chess notation
 */
export function squareToNotation(file: number, rank: number): string {
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  return `${files[file]}${rank + 1}`;
}

/**
 * Validate board orientation using piece positions
 * If we detect more white pieces on ranks 1-2 and black on 7-8, orientation is correct
 * Otherwise, we need to rotate
 */
export function validateOrientation(
  grid: BoardGrid,
  pieces: Array<{ center: Point; isWhite: boolean }>
): "white-bottom" | "black-bottom" {
  let whiteLow = 0;  // White pieces on ranks 1-2
  let whiteHigh = 0; // White pieces on ranks 7-8
  let blackLow = 0;
  let blackHigh = 0;
  
  for (const piece of pieces) {
    const square = getSquareAtPoint(piece.center, grid);
    if (!square) continue;
    
    if (square.rank <= 1) {  // Ranks 1-2
      if (piece.isWhite) whiteLow++;
      else blackLow++;
    } else if (square.rank >= 6) {  // Ranks 7-8
      if (piece.isWhite) whiteHigh++;
      else blackHigh++;
    }
  }
  
  // If more white pieces are low and black are high, orientation is correct
  // If reversed, we need to flip
  const whiteBottomScore = whiteLow + blackHigh;
  const blackBottomScore = blackLow + whiteHigh;
  
  return whiteBottomScore >= blackBottomScore ? "white-bottom" : "black-bottom";
}

/**
 * Rotate board 180 degrees (swap a1<->h8, etc.)
 */
export function rotateBoard(grid: BoardGrid): BoardGrid {
  const rotatedSquares: GridSquare[][] = [];
  
  for (let rank = 7; rank >= 0; rank--) {
    const row: GridSquare[] = [];
    for (let file = 7; file >= 0; file--) {
      const original = grid.squares[7 - rank][7 - file];
      row.push({
        ...original,
        file: 7 - original.file,
        rank: 7 - original.rank,
      });
    }
    rotatedSquares.push(row);
  }
  
  return {
    ...grid,
    squares: rotatedSquares,
    orientation: grid.orientation === "white-bottom" ? "black-bottom" : "white-bottom",
  };
}

/**
 * Sample color at a point (for color-based validation)
 * Returns brightness (0-255)
 */
export function sampleBrightness(
  imageData: Uint8Array,
  point: Point,
  imageWidth: number,
  imageHeight: number
): number {
  const x = Math.round(Math.max(0, Math.min(imageWidth - 1, point.x)));
  const y = Math.round(Math.max(0, Math.min(imageHeight - 1, point.y)));
  
  // Assuming RGBA format
  const idx = (y * imageWidth + x) * 4;
  const r = imageData[idx];
  const g = imageData[idx + 1];
  const b = imageData[idx + 2];
  
  // Luminance approximation
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Validate checker pattern by sampling colors
 * Returns true if the pattern matches expected light/dark alternation
 */
export function validateCheckerPattern(
  imageData: Uint8Array,
  grid: BoardGrid,
  imageWidth: number,
  imageHeight: number
): { isValid: boolean; matchScore: number } {
  let matches = 0;
  let total = 0;
  
  // Sample center of each square
  const brightnesses: number[][] = [];
  
  for (let rank = 0; rank < 8; rank++) {
    const row: number[] = [];
    for (let file = 0; file < 8; file++) {
      const square = grid.squares[rank][file];
      const brightness = sampleBrightness(imageData, square.center, imageWidth, imageHeight);
      row.push(brightness);
    }
    brightnesses.push(row);
  }
  
  // Find threshold (median brightness)
  const allBrightnesses = brightnesses.flat().sort((a, b) => a - b);
  const threshold = allBrightnesses[32]; // Median of 64 values
  
  // Check if pattern matches expected alternation
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const brightness = brightnesses[rank][file];
      const expectedLight = (file + rank) % 2 === 0;
      const isLight = brightness > threshold;
      
      if (expectedLight === isLight) matches++;
      total++;
    }
  }
  
  const matchScore = matches / total;
  
  // If less than 50% match, pattern is inverted
  // This helps detect correct orientation
  return {
    isValid: matchScore > 0.7 || matchScore < 0.3,
    matchScore,
  };
}
