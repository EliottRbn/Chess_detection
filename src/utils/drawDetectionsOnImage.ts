import { Skia, PaintStyle, FontStyle } from "@shopify/react-native-skia";
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
  const paint = Skia.Paint();
  paint.setStyle(PaintStyle.Stroke);
  paint.setStrokeWidth(2);
  
  const textPaint = Skia.Paint();
  textPaint.setColor(Skia.Color("#FFFFFF"));
  
  // Check for board polygon to draw grid
  const boardDet = detections.find(d => d.class === "chessboard" && d.polygon);
  if (boardDet && boardDet.polygon && boardDet.polygon.length >= 4) {
    try {
      const poly = boardDet.polygon;
      const path = Skia.Path.Make();
      path.moveTo(poly[0].x, poly[0].y);
      for (let i = 1; i < poly.length; i++) {
        path.lineTo(poly[i].x, poly[i].y);
      }
      path.close();
      
      const boardPaint = Skia.Paint();
      boardPaint.setStyle(PaintStyle.Stroke);
      boardPaint.setStrokeWidth(3);
      boardPaint.setColor(Skia.Color("#00FFFF"));
      canvas.drawPath(path, boardPaint);
      
      // Draw grid lines
      const { computeBoardGrid } = require("./boardGridAnalysis");
      const grid = computeBoardGrid(poly);
      
      if (grid) {
        const gridPaint = Skia.Paint();
        gridPaint.setStyle(PaintStyle.Stroke);
        gridPaint.setStrokeWidth(1);
        gridPaint.setColor(Skia.Color("rgba(0, 255, 255, 0.5)"));
        
        // Draw all 64 squares
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const sq = grid.squares[r][c];
            const sqPath = Skia.Path.Make();
            sqPath.moveTo(sq.corners[0].x, sq.corners[0].y);
            sqPath.lineTo(sq.corners[1].x, sq.corners[1].y);
            sqPath.lineTo(sq.corners[2].x, sq.corners[2].y);
            sqPath.lineTo(sq.corners[3].x, sq.corners[3].y);
            sqPath.close();
            canvas.drawPath(sqPath, gridPaint);
          }
        }
      }
    } catch (e) {
      console.warn("Failed to draw grid:", e);
    }
  }

  for (const det of detections) {
    if (det.class === "chessboard") continue; // Already handled
    
    if (det.bbox) {
      const [x1, y1, x2, y2] = det.bbox;
      
      // Check for NaN
      if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) continue;

      paint.setColor(Skia.Color(det.color || "#00FF00"));
      canvas.drawRect({ x: x1, y: y1, width: x2 - x1, height: y2 - y1 }, paint);

      if (det.label) {
        // Draw label background
        const fontMgr = Skia.FontMgr.System();
        // matchFamilyStyle requires 2 args: familyName and style
        // We can use default style or create one
        const typeface = fontMgr.matchFamilyStyle("sans-serif", FontStyle.Normal);
        if (!typeface) continue;
        const font = Skia.Font(typeface, 14);
        
        const text = det.label;
        const textWidth = font.getTextWidth(text);
        const textHeight = 14;
        
        const bgPaint = Skia.Paint();
        bgPaint.setColor(Skia.Color(det.color || "#00FF00"));
        
        canvas.drawRect(
          { x: x1, y: y1 - textHeight - 4, width: textWidth + 8, height: textHeight + 4 },
          bgPaint
        );
        
        canvas.drawText(text, x1 + 4, y1 - 4, textPaint, font);
      }
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