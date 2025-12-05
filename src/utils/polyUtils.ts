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
 * Approximate polygon like cv2.approxPolyDP using Douglas–Peucker.
 * contour : array of [x,y]
 * epsilonRatio : ~0.01–0.02 is typical
 */
export function approxPolyDP(contour: Point[], epsilonRatio = 0.02): Point[] {
  if (contour.length < 3) return contour;

  const eps = epsilonRatio * contourPerimeter(contour);

  function perpendicularDist(p: Point, a: Point, b: Point): number {
    const [x, y] = p;
    const [x1, y1] = a;
    const [x2, y2] = b;

    const num = Math.abs(
      (y2 - y1) * x -
      (x2 - x1) * y +
      x2 * y1 -
      y2 * x1
    );

    const den = Math.hypot(x2 - x1, y2 - y1);
    return den === 0 ? 0 : num / den;
  }

  function dp(pts: Point[]): Point[] {
    let maxDist = 0;
    let index = 0;

    for (let i = 1; i < pts.length - 1; i++) {
      const d = perpendicularDist(pts[i], pts[0], pts[pts.length - 1]);
      if (d > maxDist) {
        maxDist = d;
        index = i;
      }
    }

    if (maxDist > eps) {
      const left = dp(pts.slice(0, index + 1));
      const right = dp(pts.slice(index));
      return [...left.slice(0, -1), ...right];
    }

    return [pts[0], pts[pts.length - 1]];
  }

  const simplified = dp(contour);

  // cv2.approxPolyDP tends to return 4 points for rectangles => enforce max 4
  if (simplified.length > 4) {
    // keep 4 farthest points from centroid
    const cx = simplified.reduce((s, p) => s + p[0], 0) / simplified.length;
    const cy = simplified.reduce((s, p) => s + p[1], 0) / simplified.length;

    return simplified
      .sort((a, b) =>
        Math.hypot(b[0] - cx, b[1] - cy) -
        Math.hypot(a[0] - cx, a[1] - cy)
      )
      .slice(0, 4);
  }

  return simplified;
}

function contourPerimeter(pts: Point[]): number {
  let p = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    p += Math.hypot(x2 - x1, y2 - y1);
  }
  return p;
}