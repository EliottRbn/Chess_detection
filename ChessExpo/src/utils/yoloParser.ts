import { PieceDetection, PIECE_CLASSES, MODEL_INPUT_SIZE } from '../types';

const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

// ... (parsePieceDetections, calculateIoU, nonMaxSuppression remain unchanged)
export function parsePieceDetections(outputData: Float32Array): PieceDetection[] {
  const numClasses = 12;
  const numDetections = 8400;
  
  const detections: PieceDetection[] = [];

  for (let i = 0; i < numDetections; i++) {
    const cx = outputData[0 * numDetections + i];
    const cy = outputData[1 * numDetections + i];
    const w = outputData[2 * numDetections + i];
    const h = outputData[3 * numDetections + i];

    let maxScore = 0;
    let maxClassId = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = outputData[(4 + c) * numDetections + i];
      if (score > maxScore) {
        maxScore = score;
        maxClassId = c;
      }
    }

    if (maxScore >= CONFIDENCE_THRESHOLD) {
      const pieceInfo = PIECE_CLASSES[maxClassId];
      detections.push({
        box: { x: cx, y: cy, width: w, height: h },
        classId: maxClassId,
        className: pieceInfo?.name ?? `class_${maxClassId}`,
        confidence: maxScore,
      });
    }
  }
  
  return nonMaxSuppression(detections);
}

function calculateIoU(box1: PieceDetection['box'], box2: PieceDetection['box']): number {
  const x1 = Math.max(box1.x - box1.width / 2, box2.x - box2.width / 2);
  const y1 = Math.max(box1.y - box1.height / 2, box2.y - box2.height / 2);
  const x2 = Math.min(box1.x + box1.width / 2, box2.x + box2.width / 2);
  const y2 = Math.min(box1.y + box1.height / 2, box2.y + box2.height / 2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area1 = box1.width * box1.height;
  const area2 = box2.width * box2.height;
  const union = area1 + area2 - intersection;

  return union > 0 ? intersection / union : 0;
}

function nonMaxSuppression(detections: PieceDetection[]): PieceDetection[] {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: PieceDetection[] = [];

  for (const detection of sorted) {
    let shouldKeep = true;
    for (const kept_detection of kept) {
      if (calculateIoU(detection.box, kept_detection.box) > IOU_THRESHOLD) {
        shouldKeep = false;
        break;
      }
    }
    if (shouldKeep) {
      kept.push(detection);
    }
  }

  return kept;
}

export interface BoardCorners {
  topLeft: { x: number; y: number };
  topRight: { x: number; y: number };
  bottomLeft: { x: number; y: number };
  bottomRight: { x: number; y: number };
  confidence: number;
}

export function parseSimpleSegmentation(
  output: Float32Array,
  size: number = 256
): BoardCorners | null {
  console.log(`[SimpleSegmentation] Parsing output size ${output.length} (${size}x${size})`);

  const mask = new Float32Array(size * size);
  let activePixels = 0;
  
  for (let i = 0; i < output.length; i++) {
    const val = 1 / (1 + Math.exp(-output[i]));
    mask[i] = val;
    if (val > 0.5) activePixels++;
  }
  
  console.log(`[SimpleSegmentation] Active pixels: ${activePixels} (${(activePixels/(size*size)*100).toFixed(1)}%)`);

  if (activePixels < 100) {
    console.log('[SimpleSegmentation] Not enough active pixels');
    return null;
  }

  const corners = findBoardCornersRobust(mask, size, 1.0);
  
  if (corners) {
    // Scale corners from mask size (256) to model size (640) for consistency with pieces
    const scale = MODEL_INPUT_SIZE / size;
    return {
      topLeft: { x: corners.topLeft.x * scale, y: corners.topLeft.y * scale },
      topRight: { x: corners.topRight.x * scale, y: corners.topRight.y * scale },
      bottomLeft: { x: corners.bottomLeft.x * scale, y: corners.bottomLeft.y * scale },
      bottomRight: { x: corners.bottomRight.x * scale, y: corners.bottomRight.y * scale },
      confidence: corners.confidence,
    };
  }

  return null;
}

/**
 * Point interface
 */
interface Point {
  x: number;
  y: number;
}

/**
 * Robust corner detection using Convex Hull and Polygon Simplification
 */
function findBoardCornersRobust(
  mask: Float32Array,
  size: number,
  confidence: number
): BoardCorners | null {
  const threshold = 0.5;
  const points: Point[] = [];
  
  // 1. Collect all points above threshold
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (mask[y * size + x] > threshold) {
        points.push({ x, y });
      }
    }
  }

  if (points.length < 100) return null;

  // 2. Compute Convex Hull (Monotone Chain)
  points.sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);

  const cross = (o: Point, a: Point, b: Point) => 
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point[] = [];
  for (const p of points) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point[] = [];
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  console.log(`[Segmentation] Convex hull size: ${hull.length}`);

  // 3. Simplify Polygon to get ~4 points
  // We want to find the 4 vertices that best approximate the hull
  // Since it's a board, the hull should be roughly a quadrilateral
  
  // Simple heuristic: Find the 4 points that are furthest from the center
  // But better: Find points with max distance from each other?
  
  // Let's use a simplified approach:
  // Find the top-most, bottom-most, left-most, right-most points
  // AND the points that are "corners" (max distance from the center in diagonal directions)
  
  // Calculate centroid
  let cx = 0, cy = 0;
  for (const p of hull) {
    cx += p.x;
    cy += p.y;
  }
  cx /= hull.length;
  cy /= hull.length;

  // Sort hull points by angle to ensure order
  hull.sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));

  // Find the 4 points that maximize the area of the quadrilateral
  // This is computationally expensive O(N^4), but N (hull size) is small (<50 usually)
  // Optimization: We can just pick points that are "peaks" in distance from center
  
  let bestQuad: Point[] = [];
  let maxArea = 0;

  // If hull is small, just take bounding box corners mapped to hull?
  // Let's stick to the quadrant method but refined:
  // Instead of just max distance, we look for the point that is most "extreme" in the direction
  // TL: min(x+y), TR: max(x-y), BR: max(x+y), BL: min(x-y)
  // This works for rotated rectangles too!
  
  let topLeft = hull[0];
  let topRight = hull[0];
  let bottomRight = hull[0];
  let bottomLeft = hull[0];

  let minSum = Infinity; // x + y (TL)
  let maxSum = -Infinity; // x + y (BR)
  let minDiff = Infinity; // x - y (BL)
  let maxDiff = -Infinity; // x - y (TR)

  for (const p of hull) {
    const sum = p.x + p.y;
    const diff = p.x - p.y;

    if (sum < minSum) { minSum = sum; topLeft = p; }
    if (sum > maxSum) { maxSum = sum; bottomRight = p; }
    if (diff < minDiff) { minDiff = diff; bottomLeft = p; }
    if (diff > maxDiff) { maxDiff = diff; topRight = p; }
  }

  // Ensure points are distinct (if board is perfectly aligned, some might overlap?)
  // Usually fine for real photos.

  console.log(`[Segmentation] Corners found (Sum/Diff):`);
  console.log(`  TL: (${topLeft.x}, ${topLeft.y})`);
  console.log(`  TR: (${topRight.x}, ${topRight.y})`);
  console.log(`  BL: (${bottomLeft.x}, ${bottomLeft.y})`);
  console.log(`  BR: (${bottomRight.x}, ${bottomRight.y})`);

  return {
    topLeft,
    topRight,
    bottomLeft,
    bottomRight,
    confidence,
  };
}
