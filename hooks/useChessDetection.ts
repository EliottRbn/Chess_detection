import { useState, useRef, useCallback } from "react";
import * as ort from "onnxruntime-react-native";
import { loadOnnxModel } from "../src/utils/loadOnnxModel";
import { preprocessImage } from "../src/utils/preprocessImage";
import { postprocessPieces, type PieceDetection, type ClassPrediction } from "../src/utils/postprocessPieces";
import { postprocessBoard } from "../src/utils/postprocessBoard";
import { getPieceDisplay } from "../src/utils/pieceLabels";
import { mapPiecesToBoard, boardToFEN, type BoardState } from "../src/utils/boardMapping";

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

export type AnalysisResult = {
  detections: Detection[];
  boardState: BoardState | null;
  boardPolygon: Array<{ x: number; y: number }> | null;
  fen: string;
};

type ModelState = "idle" | "loading" | "ready" | "error";

const INPUT_SIZE = 640;

// Singleton sessions
let piecesSession: ort.InferenceSession | null = null;
let boardSession: ort.InferenceSession | null = null;
let isLoadingModels = false;
let loadPromise: Promise<void> | null = null;

async function ensureModelsLoaded(): Promise<void> {
  if (piecesSession && boardSession) return;
  
  if (loadPromise) {
    await loadPromise;
    return;
  }
  
  if (isLoadingModels) return;
  
  isLoadingModels = true;
  loadPromise = (async () => {
    try {
      if (!piecesSession) {
        piecesSession = await loadOnnxModel(
          require("@/assets/models/yolo8-20epoch.onnx")
        );
      }
      if (!boardSession) {
        boardSession = await loadOnnxModel(
          require("@/assets/models/chessboard_detection.onnx")
        );
      }
    } finally {
      isLoadingModels = false;
    }
  })();
  
  await loadPromise;
}

export function useChessDetection() {
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const processingRef = useRef(false);

  const loadModels = useCallback(async () => {
    if (modelState === "ready" || modelState === "loading") return;
    
    setModelState("loading");
    setError(null);
    
    try {
      await ensureModelsLoaded();
      setModelState("ready");
    } catch (e: any) {
      setError(e?.message ?? "Failed to load models");
      setModelState("error");
    }
  }, [modelState]);

  /**
   * Run full analysis: board detection FIRST, then pieces
   * Returns both raw detections and mapped board state
   */
  const runAnalysis = useCallback(async (
    imageUri: string,
    options?: { confidenceThreshold?: number }
  ): Promise<AnalysisResult> => {
    const { confidenceThreshold = 0.25 } = options ?? {};
    
    if (processingRef.current) {
      console.log("Inference already in progress, skipping");
      return { detections: [], boardState: null, boardPolygon: null, fen: "8/8/8/8/8/8/8/8" };
    }
    
    if (modelState !== "ready") {
      await loadModels();
    }
    
    if (!piecesSession || !boardSession) {
      throw new Error("Models not loaded");
    }
    
    processingRef.current = true;
    setIsProcessing(true);
    
    try {
      // Preprocess image
      const inputTensor = await preprocessImage(imageUri, INPUT_SIZE);
      const tensor = new ort.Tensor("float32", inputTensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      
      // STEP 1: Detect BOARD first
      console.log("Step 1: Detecting board...");
      const outputsBoard = await boardSession.run({ images: tensor });
      const boardDetection = postprocessBoard(outputsBoard);
      
      let boardPolygon: Array<{ x: number; y: number }> | null = null;
      const detections: Detection[] = [];
      
      if (boardDetection) {
        boardPolygon = boardDetection.polygon;
        console.log("Board detected with", boardPolygon?.length, "corners");
        
        detections.push({
          bbox: boardDetection.bbox as [number, number, number, number],
          confidence: boardDetection.confidence,
          class: "chessboard",
          polygon: boardPolygon,
          color: "#00FFFF",
        });
      } else {
        console.warn("No board detected!");
      }
      
      // STEP 2: Detect pieces
      console.log("Step 2: Detecting pieces...");
      const outputsPieces = await piecesSession.run({ images: tensor });
      const pieceDetections = postprocessPieces(outputsPieces, confidenceThreshold);
      
      // Format piece detections
      const pieceResults: Detection[] = pieceDetections.map((det: PieceDetection) => {
        const display = getPieceDisplay(det.classId);
        return {
          bbox: det.bbox,
          confidence: det.confidence,
          classId: det.classId,
          label: display.label,
          symbol: display.symbol,
          color: display.color,
          predictions: det.predictions,
        };
      });
      
      // Apply smart post-processing (NMS + chess constraints)
      const { smartPostProcess } = await import("../src/utils/smartPostProcess");
      const processedPieces = smartPostProcess(pieceResults) as Detection[];
      
      detections.push(...processedPieces);
      console.log("Detected", processedPieces.length, "pieces");
      
      // STEP 3: Map pieces to board squares using perspective transform
      console.log("Step 3: Mapping pieces to board...");
      const boardState = mapPiecesToBoard(processedPieces, boardPolygon ?? undefined);
      const fen = boardToFEN(boardState);
      console.log("FEN:", fen);
      
      return {
        detections,
        boardState,
        boardPolygon,
        fen,
      };
    } finally {
      processingRef.current = false;
      setIsProcessing(false);
    }
  }, [modelState, loadModels]);

  // Legacy runInference for compatibility
  const runInference = useCallback(async (
    imageUri: string,
    options?: { detectBoard?: boolean; confidenceThreshold?: number }
  ): Promise<Detection[]> => {
    const result = await runAnalysis(imageUri, options);
    return result.detections;
  }, [runAnalysis]);

  return {
    modelState,
    isProcessing,
    error,
    loadModels,
    runInference,
    runAnalysis,  // New method that returns full result
    inputSize: INPUT_SIZE,
  };
}
