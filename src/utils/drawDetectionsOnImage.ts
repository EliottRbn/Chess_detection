import { Skia, PaintStyle } from "@shopify/react-native-skia";
import * as FileSystem from "expo-file-system/legacy";
import { encode as b64encode } from "base64-arraybuffer";

export type Detection = {
  bbox?: [number, number, number, number];
  confidence: number;
  classId?: number;
  label?: string;
  symbol?: string;
  color?: string;
  polygon?: Array<{ x: number; y: number }>;
  class?: string;
};

export async function drawDetectionsOnImage(
  uri: string,
  detections: Detection[],
  inputSize: number
) {
  const width = inputSize;
  const height = inputSize;

  // 1) Read image as base64
  const base64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });

  const skData = Skia.Data.fromBase64(base64);
  const image = Skia.Image.MakeImageFromEncoded(skData);
  if (!image) throw new Error("Skia failed to load image");

  // 2) Create surface
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

  // 4) Draw detections
  for (const det of detections) {
    try {
      if (det.bbox) {
        const [x1, y1, x2, y2] = det.bbox;
        const color = det.color ?? "#FF0000";
        
        // Draw bounding box
        const rectPaint = Skia.Paint();
        rectPaint.setColor(Skia.Color(color));
        rectPaint.setStrokeWidth(3);
        rectPaint.setStyle(PaintStyle.Stroke);
        
        canvas.drawRect(Skia.XYWHRect(x1, y1, x2 - x1, y2 - y1), rectPaint);
        
        // Draw label inside bbox (top-left corner)
        if (det.label) {
          const pieceType = det.label.split("-")[1]?.[0] ?? "?";
          const isWhite = det.label.startsWith("white");
          const labelText = isWhite ? pieceType.toUpperCase() : pieceType.toLowerCase();
          
          // Label background
          const bgPaint = Skia.Paint();
          bgPaint.setColor(Skia.Color(color));
          bgPaint.setStyle(PaintStyle.Fill);
          canvas.drawRect(Skia.XYWHRect(x1, y1, 20, 20), bgPaint);
          
          // Label text
          const font = Skia.Font(undefined, 14);
          const textPaint = Skia.Paint();
          textPaint.setColor(Skia.Color("#FFFFFF"));
          canvas.drawText(labelText, x1 + 5, y1 + 15, textPaint, font);
        }
      }

      // Draw polygon for board
      if (det.polygon && det.polygon.length > 0) {
        const pts = det.polygon;
        const path = Skia.Path.Make();
        const polyPaint = Skia.Paint();
        polyPaint.setColor(Skia.Color("#00FFFF"));
        polyPaint.setStrokeWidth(4);
        polyPaint.setStyle(PaintStyle.Stroke);

        path.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          path.lineTo(pts[i].x, pts[i].y);
        }
        path.close();
        canvas.drawPath(path, polyPaint);
      }
    } catch {
      // Skip detection if drawing fails
    }
  }

  // 5) Export to PNG
  const snapshot = surface.makeImageSnapshot();
  const bytes = snapshot.encodeToBytes();
  const arrayBuf: ArrayBuffer = new Uint8Array(bytes).buffer;
  const outBase64 = b64encode(arrayBuf);

  // 6) Save to file with unique name
  const outPath = FileSystem.cacheDirectory + "annotated_" + Date.now() + ".png";
  await FileSystem.writeAsStringAsync(outPath, outBase64, {
    encoding: FileSystem.EncodingType.Base64,
  });

  return outPath;
}