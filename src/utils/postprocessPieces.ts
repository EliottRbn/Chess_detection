import type * as ort from "onnxruntime-react-native";

export type PieceDetection = {
  bbox: [number, number, number, number];
  confidence: number;
  classId: number;
};

/**
 * Postprocess adapté au modèle EXPORTÉ :
 * dims = [1, 16, 8400]
 *  - 4 premiers = [x, y, w, h]
 *  - PAS d'objectness
 *  - 12 classes directement
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

    // Classes (12)
    let bestCls = -1;
    let bestScore = 0;
    for (let c = 4; c < 16; c++) {
      const score = at(c, i);
      if (score > bestScore) {
        bestScore = score;
        bestCls = c - 4;
      }
    }

    if (bestScore < confThreshold) continue;

    const x1 = x - w / 2;
    const y1 = y - h / 2;
    const x2 = x + w / 2;
    const y2 = y + h / 2;

    detections.push({
      bbox: [x1, y1, x2, y2],
      confidence: bestScore,
      classId: bestCls,
    });
  }

  return detections;
}