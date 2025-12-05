import type * as ort from "onnxruntime-react-native";

export type ClassPrediction = {
  classId: number;
  score: number;
};

export type PieceDetection = {
  bbox: [number, number, number, number];
  confidence: number;
  classId: number;
  // Top 2 class predictions for fallback
  predictions: ClassPrediction[];
};

/**
 * Postprocess adapté au modèle EXPORTÉ :
 * dims = [1, 16, 8400]
 *  - 4 premiers = [x, y, w, h]
 *  - PAS d'objectness
 *  - 12 classes directement
 * 
 * Returns top 2 class predictions per detection for fallback logic
 */
export function postprocessPieces(
  outputs: Record<string, ort.Tensor>,
  confThreshold = 0.25
): PieceDetection[] {
  const name = Object.keys(outputs)[0];
  const out = outputs[name];
  const data = out.data as Float32Array;
  const [batch, D, N] = out.dims;

  if (batch !== 1) {
    console.warn("Batch != 1, utilisation batch 0 uniquement");
  }
  if (D !== 16) {
    throw new Error(`Modèle pièces inattendu : D=${D}, attendu 16`);
  }

  const detections: PieceDetection[] = [];

  const at = (d: number, i: number) => data[d * N + i];

  for (let i = 0; i < N; i++) {
    // BBOX
    const x = at(0, i);
    const y = at(1, i);
    const w = at(2, i);
    const h = at(3, i);

    // Collect all class scores
    const classScores: ClassPrediction[] = [];
    for (let c = 4; c < 16; c++) {
      const score = at(c, i);
      classScores.push({ classId: c - 4, score });
    }

    // Sort by score descending and take top 2
    classScores.sort((a, b) => b.score - a.score);
    const top2 = classScores.slice(0, 2);

    const bestScore = top2[0].score;
    const bestCls = top2[0].classId;

    if (bestScore < confThreshold) continue;

    const x1 = x - w / 2;
    const y1 = y - h / 2;
    const x2 = x + w / 2;
    const y2 = y + h / 2;

    detections.push({
      bbox: [x1, y1, x2, y2],
      confidence: bestScore,
      classId: bestCls,
      predictions: top2,
    });
  }

  return detections;
}