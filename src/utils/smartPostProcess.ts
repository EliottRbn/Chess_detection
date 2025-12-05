/**
 * Smart post-processing for chess piece detections
 * Applies NMS and chess rules constraints
 */

// Detection type matching useChessDetection
export type Detection = {
  bbox: [number, number, number, number];
  confidence: number;
  classId?: number;
  label?: string;
  symbol?: string;
  color?: string;
  polygon?: Array<{ x: number; y: number }>;
  class?: string;
};

// Chess piece count constraints per color
const PIECE_LIMITS: Record<string, number> = {
  king: 1,
  queen: 1,  // Can be more with promotion, but typically 1
  rook: 2,
  bishop: 2,
  knight: 2,
  pawn: 8,
};

/**
 * Calculate IoU (Intersection over Union) between two bounding boxes
 */
function calculateIoU(
  box1: [number, number, number, number],
  box2: [number, number, number, number]
): number {
  const [x1_1, y1_1, x2_1, y2_1] = box1;
  const [x1_2, y1_2, x2_2, y2_2] = box2;

  // Calculate intersection
  const xLeft = Math.max(x1_1, x1_2);
  const yTop = Math.max(y1_1, y1_2);
  const xRight = Math.min(x2_1, x2_2);
  const yBottom = Math.min(y2_1, y2_2);

  if (xRight < xLeft || yBottom < yTop) {
    return 0;
  }

  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);

  // Calculate union
  const box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
  const box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);
  const unionArea = box1Area + box2Area - intersectionArea;

  return intersectionArea / unionArea;
}

/**
 * Apply Non-Maximum Suppression to remove overlapping detections
 */
function applyNMS(detections: Detection[], iouThreshold: number = 0.5): Detection[] {
  if (detections.length === 0) return [];

  // Sort by confidence (descending)
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    
    const current = sorted[i];
    if (!current.bbox) {
      kept.push(current);
      continue;
    }

    kept.push(current);

    // Suppress overlapping boxes with same class
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      
      const other = sorted[j];
      if (!other.bbox) continue;

      // Only suppress if same piece type
      if (current.label === other.label) {
        const iou = calculateIoU(current.bbox, other.bbox);
        if (iou > iouThreshold) {
          suppressed.add(j);
        }
      }
    }
  }

  return kept;
}

/**
 * Apply chess piece constraints
 * Keeps only the allowed number of each piece type per color
 */
function applyChessConstraints(detections: Detection[]): Detection[] {
  // Group detections by piece type (e.g., "white-king", "black-pawn")
  const grouped: Record<string, Detection[]> = {};
  const nonPieceDetections: Detection[] = [];

  for (const det of detections) {
    if (det.label) {
      if (!grouped[det.label]) {
        grouped[det.label] = [];
      }
      grouped[det.label].push(det);
    } else {
      nonPieceDetections.push(det);
    }
  }

  const result: Detection[] = [];

  // For each piece type, keep only the allowed number with highest confidence
  for (const [label, dets] of Object.entries(grouped)) {
    const pieceType = label.split("-")[1]; // e.g., "king" from "white-king"
    const limit = PIECE_LIMITS[pieceType] ?? 2;
    
    // Sort by confidence and take top N
    const sorted = dets.sort((a, b) => b.confidence - a.confidence);
    result.push(...sorted.slice(0, limit));
  }

  // Add non-piece detections (like chessboard)
  result.push(...nonPieceDetections);

  return result;
}

/**
 * Apply global NMS across all classes for heavily overlapping detections
 * This handles cases where different piece types are detected on the same location
 */
function applyGlobalNMS(detections: Detection[], iouThreshold: number = 0.7): Detection[] {
  if (detections.length === 0) return [];

  // Sort by confidence (descending)
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    
    const current = sorted[i];
    if (!current.bbox) {
      kept.push(current);
      continue;
    }

    kept.push(current);

    // Suppress ANY overlapping boxes (cross-class)
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      
      const other = sorted[j];
      if (!other.bbox) continue;

      const iou = calculateIoU(current.bbox, other.bbox);
      if (iou > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return kept;
}

/**
 * Main function: apply all post-processing steps
 */
export function smartPostProcess(
  detections: Detection[],
  options?: {
    nmsThreshold?: number;
    globalNmsThreshold?: number;
    applyConstraints?: boolean;
  }
): Detection[] {
  const {
    nmsThreshold = 0.5,
    globalNmsThreshold = 0.7,
    applyConstraints = true,
  } = options ?? {};

  // Step 1: Apply NMS per class
  let result = applyNMS(detections, nmsThreshold);

  // Step 2: Apply global NMS for cross-class overlaps
  result = applyGlobalNMS(result, globalNmsThreshold);

  // Step 3: Apply chess constraints
  if (applyConstraints) {
    result = applyChessConstraints(result);
  }

  // Sort by confidence for consistent output
  result.sort((a, b) => b.confidence - a.confidence);

  return result;
}
