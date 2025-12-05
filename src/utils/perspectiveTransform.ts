/**
 * Perspective transform utilities
 * Implements cv2.getPerspectiveTransform and cv2.perspectiveTransform in TypeScript
 */

export type Point = { x: number; y: number };

/**
 * Compute 3x3 perspective transform matrix from 4 source points to 4 destination points
 * Equivalent to cv2.getPerspectiveTransform(src, dst)
 */
export function getPerspectiveTransform(
  src: [Point, Point, Point, Point],
  dst: [Point, Point, Point, Point]
): number[][] {
  // Build the 8x8 matrix for solving the system of equations
  // For each point pair (xi, yi) -> (ui, vi):
  // ui = (a*xi + b*yi + c) / (g*xi + h*yi + 1)
  // vi = (d*xi + e*yi + f) / (g*xi + h*yi + 1)
  
  const A: number[][] = [];
  const b: number[] = [];
  
  for (let i = 0; i < 4; i++) {
    const sx = src[i].x;
    const sy = src[i].y;
    const dx = dst[i].x;
    const dy = dst[i].y;
    
    A.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
    b.push(dx);
    
    A.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
    b.push(dy);
  }
  
  // Solve the system using Gaussian elimination
  const h = solveLinearSystem(A, b);
  
  // Build 3x3 matrix
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

/**
 * Solve linear system Ax = b using Gaussian elimination with partial pivoting
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const augmented = A.map((row, i) => [...row, b[i]]);
  
  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[maxRow][col])) {
        maxRow = row;
      }
    }
    [augmented[col], augmented[maxRow]] = [augmented[maxRow], augmented[col]];
    
    // Eliminate column
    for (let row = col + 1; row < n; row++) {
      const factor = augmented[row][col] / augmented[col][col];
      for (let j = col; j <= n; j++) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }
  
  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= augmented[i][j] * x[j];
    }
    x[i] /= augmented[i][i];
  }
  
  return x;
}

/**
 * Apply perspective transform to a point
 * Equivalent to cv2.perspectiveTransform
 */
export function perspectiveTransform(point: Point, M: number[][]): Point {
  const x = point.x;
  const y = point.y;
  
  const w = M[2][0] * x + M[2][1] * y + M[2][2];
  const px = (M[0][0] * x + M[0][1] * y + M[0][2]) / w;
  const py = (M[1][0] * x + M[1][1] * y + M[1][2]) / w;
  
  return { x: px, y: py };
}

/**
 * Get board square from warped coordinates (0-640 -> 0-7)
 */
export function getSquareFromWarped(
  warpedX: number, 
  warpedY: number, 
  boardSize: number = 640
): { file: number; rank: number } | null {
  const squareSize = boardSize / 8;
  
  const file = Math.floor(warpedX / squareSize);
  const rank = Math.floor(warpedY / squareSize);
  
  if (file < 0 || file > 7 || rank < 0 || rank > 7) {
    return null;
  }
  
  return { file, rank };
}
