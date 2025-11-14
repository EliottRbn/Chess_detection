// src/utils/polyUtils.ts

export type Point = [number, number];

/**
 * Retourne tous les contours trouvés dans une image binaire (0/1)
 * Format image : number[][] (H × W)
 */
export function findContours(binary: number[][]): Point[][] {
  const H = binary.length;
  const W = binary[0].length;

  const contours: Point[][] = [];

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      if (binary[y][x] === 1 && hasZeroAround(binary, x, y)) {
        const contour = traceContour(binary, x, y);
        if (contour.length > 10) {
          contours.push(contour);
        }
      }
    }
  }

  return contours;
}

/**
 * Vérifie si un pixel 1 a au moins un voisin 0 (= bord)
 */
function hasZeroAround(binary: number[][], x: number, y: number): boolean {
  return (
    binary[y - 1][x] === 0 ||
    binary[y + 1][x] === 0 ||
    binary[y][x - 1] === 0 ||
    binary[y][x + 1] === 0
  );
}

/**
 * Suivi simple d'un contour 1-pixel
 */
function traceContour(binary: number[][], startX: number, startY: number): Point[] {
  const H = binary.length;
  const W = binary[0].length;

  const contour: Point[] = [];
  const visited = new Set<string>();

  let x = startX;
  let y = startY;

  const dirs: Point[] = [
    [1, 0],  // droite
    [0, 1],  // bas
    [-1, 0], // gauche
    [0, -1], // haut
  ];

  while (true) {
    const key = `${x},${y}`;
    if (visited.has(key)) break;
    visited.add(key);

    contour.push([x, y]);

    let moved = false;

    for (const [dx, dy] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx >= 0 && nx < W && ny >= 0 && ny < H && binary[ny][nx] === 1) {
        x = nx;
        y = ny;
        moved = true;
        break;
      }
    }

    if (!moved) break;
  }

  return contour;
}

/**
 * Aire d'un contour par formule de Shoelace
 */
export function contourArea(contour: Point[]): number {
  let area = 0;
  const n = contour.length;

  for (let i = 0; i < n; i++) {
    const [x1, y1] = contour[i];
    const [x2, y2] = contour[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }

  return Math.abs(area) / 2;
}

/**
 * Approximation naïve de polygone : échantillonnage
 */
export function approxPolyDP(contour: Point[], epsilonRatio = 0.02): Point[] {
  const simplified: Point[] = [];

  const step = Math.max(1, Math.floor(contour.length * epsilonRatio));

  for (let i = 0; i < contour.length; i += step) {
    simplified.push(contour[i]);
  }

  // On veut 4 points max (forme plateau carré)
  if (simplified.length > 4) {
    return simplified.slice(0, 4);
  }

  return simplified;
}