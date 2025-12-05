import { sigmoid } from "./sigmoid";
import { findContours, approxPolyDP, contourArea } from "./polyUtils";

export type BoardPoint = { x: number; y: number };   // ← ajoute ça

type BoundingBox = { x1: number; y1: number; x2: number; y2: number };

const INPUT_SIZE = 640;

export function postprocessBoard(outputs: Record<string, any>) {
  console.log("[postprocessBoard] Starting board detection...");
  const out0 = outputs[Object.keys(outputs)[0]]; // [1,38,8400]
  const proto = outputs[Object.keys(outputs)[1]]; // [1,32,160,160]

  if (!out0 || !proto) {
    console.warn("[postprocessBoard] ❌ Invalid YOLO-seg outputs");
    return null;
  }

  const pred = out0.data as Float32Array;
  const [_, C, N] = out0.dims; // 38,8400

  const numMasks = 32; // Standard for YOLOv8-seg
  const numClasses = C - 4 - numMasks;
  console.log(`[postprocessBoard] Dims: C=${C}, N=${N}, numClasses=${numClasses}, numMasks=${numMasks}`);
  
  const maskProto = proto.data as Float32Array;
  const [__, M, H, W] = proto.dims; // 32,160,160

  let bestIndex = -1;
  let bestScore = 0;

  // YOLOv8 output: [x, y, w, h, class0, class1, ..., mask0, mask1...]
  // No objectness score!
  
  for (let i = 0; i < N; i++) {
    // Find best class score for this anchor
    let maxClassScore = 0;
    
    for (let c = 0; c < numClasses; c++) {
      const score = pred[(4 + c) * N + i];
      if (score > maxClassScore) {
        maxClassScore = score;
      }
    }

    if (maxClassScore > bestScore) {
      bestScore = maxClassScore;
      bestIndex = i;
    }
  }

  console.log(`[postprocessBoard] Best score: ${bestScore.toFixed(3)}, index: ${bestIndex}`);
  if (bestScore < 0.2 || bestIndex === -1) {
    console.warn(`[postprocessBoard] ❌ Score too low (${bestScore.toFixed(3)} < 0.2) or no detection`);
    return null;
  }

  const at = (d: number, i: number) => pred[d * N + i];

  const cx = at(0, bestIndex);
  const cy = at(1, bestIndex);
  const w = at(2, bestIndex);
  const h = at(3, bestIndex);

  const x1 = cx - w / 2;
  const y1 = cy - h / 2;
  const x2 = cx + w / 2;
  const y2 = cy + h / 2;
  const rawBox: BoundingBox = { x1, y1, x2, y2 };

  // coefficients de masque
  const coeffs = [];
  const maskStartIndex = 4 + numClasses;
  for (let m = 0; m < numMasks; m++) {
    coeffs.push(at(maskStartIndex + m, bestIndex));
  }

  // reconstitution masque
  const mask = new Array(H * W).fill(0);
  for (let m = 0; m < numMasks; m++) {
    const k = coeffs[m];
    for (let i = 0; i < H * W; i++) {
      mask[i] += k * maskProto[m * H * W + i];
    }
  }

  // sigmoid
  for (let i = 0; i < H * W; i++) {
    mask[i] = sigmoid(mask[i]);
  }

  // binarisation
  const binary = [];
  for (let y = 0; y < H; y++) {
    const row = [];
    for (let x = 0; x < W; x++) {
      row.push(mask[y * W + x] > 0.5 ? 1 : 0);
    }
    binary.push(row);
  }

  // Try to get polygon from mask contours
  let scaledCorners: BoardPoint[] | null = null;
  
  const contours = findContours(binary);
  console.log(`[postprocessBoard] Found ${contours.length} contours`);
  
  if (contours.length > 0) {
    const biggest = contours.reduce((a, b) =>
      contourArea(b) > contourArea(a) ? b : a
    );
    console.log(`[postprocessBoard] Biggest contour: ${biggest.length} points, area: ${contourArea(biggest).toFixed(1)}`);
    
    // Try polygon approximation
    let polygonRaw = approxPolyDP(biggest, 0.02);
    console.log(`[postprocessBoard] Polygon approximation: ${polygonRaw?.length || 0} points`);
    
    if (!polygonRaw || polygonRaw.length < 4) {
      // Try relaxed approximation
      polygonRaw = approxPolyDP(biggest, 0.05);
      console.log(`[postprocessBoard] Relaxed approximation: ${polygonRaw?.length || 0} points`);
    }
    
    if (polygonRaw && polygonRaw.length >= 4) {
      // Convert to uniform format
      let points: BoardPoint[] = polygonRaw.map(
        (p: [number, number] | { x: number; y: number }) => {
          if (Array.isArray(p)) {
            return { x: p[0], y: p[1] };
          }
          return { x: p.x, y: p.y };
        }
      );

      if (points.length > 4) {
        points = findFourCorners(points);
      }

      const orderedCorners = orderCornersClockwise(points);
      const scaledBox = scaleBoxToMask(rawBox, W, H);
      const biggestPoints: BoardPoint[] = biggest.map(([px, py]) => ({ x: px, y: py }));
      
      const refinedCorners = refineBoardPolygon({
        corners: orderedCorners,
        contour: biggestPoints,
        bbox: scaledBox,
        maskWidth: W,
        maskHeight: H,
      });
      
      const scaleX = INPUT_SIZE / W;
      const scaleY = INPUT_SIZE / H;
      
      scaledCorners = refinedCorners.map(p => ({
        x: p.x * scaleX,
        y: p.y * scaleY,
      }));
      console.log("[postprocessBoard] ✅ Using polygon from mask");
    }
  }
  
  // FALLBACK: Use bounding box if polygon failed
  if (!scaledCorners) {
    console.log("[postprocessBoard] ⚠️ Using bounding box as fallback");
    scaledCorners = [
      { x: x1, y: y1 },  // top-left
      { x: x2, y: y1 },  // top-right
      { x: x2, y: y2 },  // bottom-right
      { x: x1, y: y2 },  // bottom-left
    ];
  }

  return {
    class: "chessboard",
    confidence: bestScore,
    bbox: [x1, y1, x2, y2],
    polygon: scaledCorners,
    corners: scaledCorners,
  };
}

function scaleBoxToMask(box: BoundingBox, maskWidth: number, maskHeight: number): BoundingBox {
  const scaleX = maskWidth / INPUT_SIZE;
  const scaleY = maskHeight / INPUT_SIZE;
  return {
    x1: clamp(box.x1 * scaleX, 0, maskWidth),
    y1: clamp(box.y1 * scaleY, 0, maskHeight),
    x2: clamp(box.x2 * scaleX, 0, maskWidth),
    y2: clamp(box.y2 * scaleY, 0, maskHeight),
  };
}

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function polygonArea(points: BoardPoint[]): number {
  if (points.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const { x: x1, y: y1 } = points[i];
    const { x: x2, y: y2 } = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

function edgeLengths(points: BoardPoint[]): number[] {
  if (points.length < 2) return [];
  const lengths: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    lengths.push(Math.hypot(dx, dy));
  }
  return lengths;
}

function cornersFromBox(box: BoundingBox, width: number, height: number, paddingRatio = 0.06): BoardPoint[] {
  const padX = (box.x2 - box.x1) * paddingRatio;
  const padY = (box.y2 - box.y1) * paddingRatio;
  const x1 = clamp(box.x1 - padX, 0, width);
  const y1 = clamp(box.y1 - padY, 0, height);
  const x2 = clamp(box.x2 + padX, 0, width);
  const y2 = clamp(box.y2 + padY, 0, height);

  return orderCornersClockwise([
    { x: x1, y: y1 },
    { x: x2, y: y1 },
    { x: x2, y: y2 },
    { x: x1, y: y2 },
  ]);
}

function extremeCorners(contour: BoardPoint[]): BoardPoint[] {
  if (!contour.length) return [];

  // Use sums/diffs to approximate oriented box corners
  const tl = contour.reduce((min, p) => (p.x + p.y < min.x + min.y ? p : min), contour[0]);
  const br = contour.reduce((max, p) => (p.x + p.y > max.x + max.y ? p : max), contour[0]);
  const tr = contour.reduce((min, p) => (p.y - p.x < min.y - min.x ? p : min), contour[0]);
  const bl = contour.reduce((max, p) => (p.y - p.x > max.y - max.x ? p : max), contour[0]);

  return orderCornersClockwise([tl, tr, br, bl]);
}

function squareFromCenter(
  center: BoardPoint,
  size: number,
  angle: number
): BoardPoint[] {
  const half = size / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  const dx = cos * half;
  const dy = sin * half;
  const dxPerp = -sin * half;
  const dyPerp = Math.cos(angle) * half;

  return orderCornersClockwise([
    { x: center.x - dx - dxPerp, y: center.y - dy - dyPerp },
    { x: center.x + dx - dxPerp, y: center.y + dy - dyPerp },
    { x: center.x + dx + dxPerp, y: center.y + dy + dyPerp },
    { x: center.x - dx + dxPerp, y: center.y - dy + dyPerp },
  ]);
}

function rotatePoint(p: BoardPoint, angle: number, center: BoardPoint): BoardPoint {
  const dx = p.x - center.x;
  const dy = p.y - center.y;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    x: center.x + dx * cos - dy * sin,
    y: center.y + dx * sin + dy * cos,
  };
}

function computePrincipalAngle(points: BoardPoint[]): number | null {
  if (points.length < 2) return null;
  const mean = points.reduce(
    (acc, p) => ({ x: acc.x + p.x / points.length, y: acc.y + p.y / points.length }),
    { x: 0, y: 0 }
  );
  let covXX = 0;
  let covYY = 0;
  let covXY = 0;
  for (const p of points) {
    const dx = p.x - mean.x;
    const dy = p.y - mean.y;
    covXX += dx * dx;
    covYY += dy * dy;
    covXY += dx * dy;
  }
  if (!Number.isFinite(covXX) || !Number.isFinite(covYY) || !Number.isFinite(covXY)) {
    return null;
  }
  return 0.5 * Math.atan2(2 * covXY, covXX - covYY);
}

function buildPrincipalSquare(args: {
  points: BoardPoint[];
  defaultAngle: number;
  size: number;
  center: BoardPoint;
}): BoardPoint[] | null {
  const { points, defaultAngle, size, center } = args;
  const angle = computePrincipalAngle(points) ?? defaultAngle;
  if (!Number.isFinite(angle)) return null;

  // Estimate width/height along principal axes
  const rotated = points.map(p => rotatePoint(p, -angle, center));
  const xs = rotated.map(p => p.x);
  const ys = rotated.map(p => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = maxX - minX;
  const height = maxY - minY;
  const fittedSize = Math.max(size, width, height);

  return squareFromCenter(center, fittedSize, angle);
}

function scorePolygon(points: BoardPoint[], targetArea: number): number {
  if (points.length !== 4) return Number.POSITIVE_INFINITY;
  const area = polygonArea(points);
  const lengths = edgeLengths(points);
  if (lengths.length !== 4) return Number.POSITIVE_INFINITY;
  const minSide = Math.min(...lengths);
  const maxSide = Math.max(...lengths);
  if (!Number.isFinite(area) || area <= 0 || !Number.isFinite(minSide) || minSide <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  const aspectPenalty = Math.abs((maxSide / minSide) - 1); // 0 when perfectly square
  const areaPenalty = Math.abs(area / targetArea - 1);
  return aspectPenalty * 2 + areaPenalty; // weight squareness a bit more
}

function clampPolygon(points: BoardPoint[], maxX: number, maxY: number): BoardPoint[] {
  return points.map(p => ({
    x: clamp(p.x, 0, maxX),
    y: clamp(p.y, 0, maxY),
  }));
}

function refineBoardPolygon(args: {
  corners: BoardPoint[];
  contour: BoardPoint[];
  bbox: BoundingBox;
  maskWidth: number;
  maskHeight: number;
}): BoardPoint[] {
  const { corners, contour, bbox, maskWidth, maskHeight } = args;
  if (!corners.length) return cornersFromBox(bbox, maskWidth, maskHeight);

  const contourCorners = extremeCorners(contour);
  const contourAreaVal = polygonArea(contourCorners);

  let chosen = corners;
  const chosenArea = polygonArea(chosen);
  if (contourCorners.length === 4 && contourAreaVal > chosenArea * 1.2) {
    chosen = contourCorners;
  }

  let lengths = edgeLengths(chosen);
  const minSide = Math.min(...lengths);
  const boxArea = Math.max(1, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1));
  const area = polygonArea(chosen);

  // If the polygon collapses to a line or is far smaller than bbox, fall back to bbox
  if (!Number.isFinite(minSide) || minSide < 5 || area < boxArea * 0.4) {
    chosen = cornersFromBox(bbox, maskWidth, maskHeight);
    lengths = edgeLengths(chosen);
  }

  // Normalize to a near-square shape if aspect ratio is off
  const width = (lengths[0] + lengths[2]) / 2;
  const height = (lengths[1] + lengths[3]) / 2;
  const aspect = height === 0 ? 1 : width / height;

  const center = chosen.reduce(
    (acc, p) => ({ x: acc.x + p.x / 4, y: acc.y + p.y / 4 }),
    { x: 0, y: 0 }
  );
  const size = Math.max(width, height);
  const angle = Math.atan2(chosen[1].y - chosen[0].y, chosen[1].x - chosen[0].x);

  const principalSquare = buildPrincipalSquare({
    points: contour.length ? contour : chosen,
    defaultAngle: angle,
    size,
    center,
  });

  // Evaluate candidates and pick the most square / area-matching polygon
  const candidates: BoardPoint[][] = [];
  candidates.push(chosen);
  if (principalSquare) candidates.push(principalSquare);
  candidates.push(cornersFromBox(bbox, maskWidth, maskHeight));

  const best = candidates.reduce(
    (bestPoly, poly) => {
      const clamped = clampPolygon(poly, maskWidth, maskHeight);
      const score = scorePolygon(clamped, boxArea);
      return score < bestPoly.score ? { poly: clamped, score } : bestPoly;
    },
    { poly: clampPolygon(chosen, maskWidth, maskHeight), score: scorePolygon(chosen, boxArea) }
  );

  return best.poly;
}

/**
 * Find the 4 most extreme corners from a polygon
 */
function findFourCorners(points: BoardPoint[]): BoardPoint[] {
  if (points.length <= 4) return points;

  // Find centroid
  const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;

  // Find the 4 points furthest from centroid, one in each quadrant
  const quadrants: { point: BoardPoint; dist: number }[] = [
    { point: points[0], dist: 0 }, // top-left (x < cx, y < cy)
    { point: points[0], dist: 0 }, // top-right (x > cx, y < cy)
    { point: points[0], dist: 0 }, // bottom-right (x > cx, y > cy)
    { point: points[0], dist: 0 }, // bottom-left (x < cx, y > cy)
  ];

  for (const p of points) {
    const dist = Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2);
    let quadrant: number;

    if (p.x <= cx && p.y <= cy) quadrant = 0; // top-left
    else if (p.x > cx && p.y <= cy) quadrant = 1; // top-right
    else if (p.x > cx && p.y > cy) quadrant = 2; // bottom-right
    else quadrant = 3; // bottom-left

    if (dist > quadrants[quadrant].dist) {
      quadrants[quadrant] = { point: p, dist };
    }
  }

  return quadrants.map(q => q.point);
}

/**
 * Order 4 corners clockwise starting from top-left
 */
function orderCornersClockwise(points: BoardPoint[]): BoardPoint[] {
  if (points.length !== 4) {
    // If not exactly 4, try to get 4
    const corners = findFourCorners(points);
    return orderCornersClockwise(corners);
  }

  // Find centroid
  const cx = points.reduce((sum, p) => sum + p.x, 0) / 4;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / 4;

  // Separate into top and bottom points
  const topPoints = points.filter(p => p.y < cy).sort((a, b) => a.x - b.x);
  const bottomPoints = points.filter(p => p.y >= cy).sort((a, b) => a.x - b.x);

  // Handle edge cases
  if (topPoints.length < 2 || bottomPoints.length < 2) {
    // Fallback: sort by angle from centroid
    const sorted = [...points].sort((a, b) => {
      const angleA = Math.atan2(a.y - cy, a.x - cx);
      const angleB = Math.atan2(b.y - cy, b.x - cx);
      return angleA - angleB;
    });
    // Rotate to start with top-left
    const minSum = sorted.reduce((min, p, i) => 
      (p.x + p.y < sorted[min].x + sorted[min].y) ? i : min, 0);
    return [...sorted.slice(minSum), ...sorted.slice(0, minSum)];
  }

  // Return in order: top-left, top-right, bottom-right, bottom-left
  return [
    topPoints[0],                        // top-left
    topPoints[topPoints.length - 1],     // top-right
    bottomPoints[bottomPoints.length - 1], // bottom-right
    bottomPoints[0],                      // bottom-left
  ];
}
