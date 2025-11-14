import { sigmoid } from "./sigmoid";
import { findContours, approxPolyDP, contourArea } from "./polyUtils";

export type BoardPoint = { x: number; y: number };   // ← ajoute ça

export function postprocessBoard(outputs: Record<string, any>) {
  const out0 = outputs[Object.keys(outputs)[0]]; // [1,38,8400]
  const proto = outputs[Object.keys(outputs)[1]]; // [1,32,160,160]

  if (!out0 || !proto) {
    console.warn("postprocessBoard: sorties YOLO-seg invalides");
    return null;
  }

  const pred = out0.data as Float32Array;
  const [_, C, N] = out0.dims; // 38,8400

  const numMasks = C - 6; // 4 bbox + obj + class
  const maskProto = proto.data as Float32Array;
  const [__, M, H, W] = proto.dims; // 32,160,160

  let bestIndex = -1;
  let bestScore = 0;

  // on cherche le chessboard uniquement (classe=1)
  const classIdx = 5 + 1;

  for (let i = 0; i < N; i++) {
    const obj = pred[4 * N + i];
    const cls = pred[classIdx * N + i];
    const score = obj * cls;

    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  if (bestScore < 0.2 || bestIndex === -1) return null;

  const at = (d: number, i: number) => pred[d * N + i];

  const cx = at(0, bestIndex);
  const cy = at(1, bestIndex);
  const w = at(2, bestIndex);
  const h = at(3, bestIndex);

  const x1 = cx - w / 2;
  const y1 = cy - h / 2;
  const x2 = cx + w / 2;
  const y2 = cy + h / 2;

  // coefficients de masque
  const coeffs = [];
  for (let m = 6; m < C; m++) coeffs.push(at(m, bestIndex));

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

  const contours = findContours(binary);
  if (!contours.length) return null;

  const biggest = contours.reduce((a, b) =>
    contourArea(b) > contourArea(a) ? b : a
  );
  const polygonRaw: Array<[number, number] | { x: number; y: number }> =
  approxPolyDP(biggest, 0.02);

if (!polygonRaw || polygonRaw.length < 4) {
  console.warn("postprocessBoard: polygon trop petit");
  return null;
}

const polygon: BoardPoint[] = polygonRaw.map(
  (p: [number, number] | { x: number; y: number }) => {
    if (Array.isArray(p)) {
      return { x: p[0], y: p[1] };
    }
    return { x: p.x, y: p.y };
  }
);

  return {
    class: "chessboard",
    confidence: bestScore,
    bbox: [x1, y1, x2, y2],
    polygon,
  };
}