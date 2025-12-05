/**
 * Smart post-processing for chess piece detections
 * Applies NMS and uses 2nd guess when piece limits are exceeded
 */

import { getPieceDisplay } from "./pieceLabels";
import { type ClassPrediction } from "./postprocessPieces";

// Detection type
export type Detection = {
  bbox: [number, number, number, number];
  confidence: number;
  classId?: number;
  label?: string;
  symbol?: string;
  color?: string;
  polygon?: Array<{ x: number; y: number }>;
  class?: string;
  predictions?: ClassPrediction[];
};

// Chess piece count constraints per color
const PIECE_LIMITS: Record<string, number> = {
  king: 1,
  queen: 1,
  rook: 2,
  bishop: 2,
  knight: 2,
  pawn: 8,
};

/**
 * Calculate IoU between two bounding boxes
 */
function calculateIoU(
  box1: [number, number, number, number],
  box2: [number, number, number, number]
): number {
  const [x1_1, y1_1, x2_1, y2_1] = box1;
  const [x1_2, y1_2, x2_2, y2_2] = box2;

  const xLeft = Math.max(x1_1, x1_2);
  const yTop = Math.max(y1_1, y1_2);
  const xRight = Math.min(x2_1, x2_2);
  const yBottom = Math.min(y2_1, y2_2);

  if (xRight < xLeft || yBottom < yTop) return 0;

  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  const box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
  const box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);
  const unionArea = box1Area + box2Area - intersectionArea;

  return intersectionArea / unionArea;
}

/**
 * Apply NMS - remove overlapping detections of the SAME class
 */
function applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
  if (detections.length === 0) return [];

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

    // Only suppress same-class overlapping boxes
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      const other = sorted[j];
      if (!other.bbox) continue;

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
 * Apply piece limits with 2nd guess fallback
 * When a piece type exceeds its limit, the lowest confidence one 
 * switches to its 2nd prediction from the model
 */
function applyLimitsWithFallback(detections: Detection[]): Detection[] {
  // Group by label
  const groups: Record<string, Detection[]> = {};
  const others: Detection[] = [];

  for (const det of detections) {
    if (det.label) {
      if (!groups[det.label]) groups[det.label] = [];
      groups[det.label].push(det);
    } else {
      others.push(det);
    }
  }

  const result: Detection[] = [];
  const usedLabels: Record<string, number> = {};

  // Process all groups, sorted by confidence
  const allDets = detections
    .filter(d => d.label)
    .sort((a, b) => b.confidence - a.confidence);

  for (const det of allDets) {
    if (!det.label) continue;

    const pieceType = det.label.split("-")[1];
    const limit = PIECE_LIMITS[pieceType] ?? 2;
    const currentCount = usedLabels[det.label] ?? 0;

    if (currentCount < limit) {
      // Can use this detection as-is
      result.push(det);
      usedLabels[det.label] = currentCount + 1;
    } else {
      // Limit exceeded - try 2nd prediction
      if (det.predictions && det.predictions.length >= 2) {
        const secondGuess = det.predictions[1];
        const secondDisplay = getPieceDisplay(secondGuess.classId);
        const secondLabel = secondDisplay.label;
        const secondPieceType = secondLabel.split("-")[1];
        const secondLimit = PIECE_LIMITS[secondPieceType] ?? 2;
        const secondCount = usedLabels[secondLabel] ?? 0;

        if (secondCount < secondLimit) {
          // Use 2nd prediction
          result.push({
            ...det,
            classId: secondGuess.classId,
            confidence: secondGuess.score,
            label: secondDisplay.label,
            symbol: secondDisplay.symbol,
            color: secondDisplay.color,
          });
          usedLabels[secondLabel] = secondCount + 1;
        }
        // If 2nd guess also exceeds limit, drop the detection
      }
      // If no 2nd prediction available, drop the detection
    }
  }

  // Add non-piece detections
  result.push(...others);

  return result;
}

/**
 * Main function: apply smart post-processing
 */
export function smartPostProcess(
  detections: Detection[],
  options?: { nmsThreshold?: number }
): Detection[] {
  const { nmsThreshold = 0.5 } = options ?? {};

  // Step 1: Apply NMS per class
  let result = applyNMS(detections, nmsThreshold);

  // Step 2: Apply piece limits with 2nd guess fallback
  result = applyLimitsWithFallback(result);

  // Sort by confidence
  result.sort((a, b) => b.confidence - a.confidence);

  return result;
}


