import { Skia, PaintStyle } from "@shopify/react-native-skia";
import * as FileSystem from "expo-file-system/legacy";
import { encode as b64encode } from "base64-arraybuffer";

export async function drawDetectionsOnImage(
  uri: string,
  detections: any[],
  inputSize: number
) {
  const width = inputSize;
  const height = inputSize;

  // 1) Lire image base64
  const base64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });

  const skData = Skia.Data.fromBase64(base64);
  const image = Skia.Image.MakeImageFromEncoded(skData);
  if (!image) throw new Error("Skia failed to load image");

  // 2) Surface
  const surface = Skia.Surface.Make(width, height);
  if (!surface) throw new Error("Cannot create surface");

  const canvas = surface.getCanvas();
  canvas.clear(Skia.Color("#FFFFFF"));

  // 3) Draw original image
  const imgPaint = Skia.Paint();
  canvas.drawImageRect(
    image,
    { x: 0, y: 0, width: image.width(), height: image.height() },
    { x: 0, y: 0, width, height },
    imgPaint
  );

  // 4) Paint outils
  const rectPaint = Skia.Paint();
  rectPaint.setColor(Skia.Color("#FF0000"));
  rectPaint.setStrokeWidth(3);
  rectPaint.setStyle(PaintStyle.Stroke);

  const polyPaint = Skia.Paint();
  polyPaint.setColor(Skia.Color("#00FFFF"));
  polyPaint.setStrokeWidth(4);
  polyPaint.setStyle(PaintStyle.Stroke);

  // 5) Draw detections
  for (const det of detections) {
    if (det.bbox) {
      const [x1, y1, x2, y2] = det.bbox;
      const rect = Skia.XYWHRect(x1, y1, x2 - x1, y2 - y1);
      canvas.drawRect(rect, rectPaint);
    }

    if (det.polygon) {
      const pts = det.polygon;
      const path = Skia.Path.Make();

      path.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) path.lineTo(pts[i].x, pts[i].y);
      path.close();

      canvas.drawPath(path, polyPaint);
    }
  }

  // 6) Snapshot → PNG → Uint8Array
  const snapshot = surface.makeImageSnapshot();
  const bytes = snapshot.encodeToBytes(); // Uint8Array

  // Convert to ArrayBuffer WITHOUT error TS
  const arrayBuf: ArrayBuffer = new Uint8Array(bytes).buffer;

  // Convert to base64 PNG
  const outBase64 = b64encode(arrayBuf);

  // Write file
  const outPath = FileSystem.cacheDirectory + "annotated.png";

  await FileSystem.writeAsStringAsync(outPath, outBase64, {
    encoding: FileSystem.EncodingType.Base64,
  });

  return outPath;
}