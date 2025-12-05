import { useState, useRef, useCallback } from "react";
import * as ort from "onnxruntime-react-native";
import { loadOnnxModel } from "../src/utils/loadOnnxModel";
import { preprocessImage } from "../src/utils/preprocessImage";
import { postprocessPieces, type PieceDetection } from "../src/utils/postprocessPieces";
import { postprocessBoard } from "../src/utils/postprocessBoard";
import { getPieceDisplay } from "../src/utils/pieceLabels";

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

type ModelState = "idle" | "loading" | "ready" | "error";

const INPUT_SIZE = 640;

// Singleton sessions to avoid reloading models
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

  const runInference = useCallback(async (
    imageUri: string,
    options?: { detectBoard?: boolean; confidenceThreshold?: number }
  ): Promise<Detection[]> => {
    const { detectBoard = true, confidenceThreshold = 0.25 } = options ?? {};
    
    // Prevent concurrent inference
    if (processingRef.current) {
      console.log("Inference already in progress, skipping");
      return [];
    }
    
    if (modelState !== "ready") {
      await loadModels();
    }
    
    if (!piecesSession) {
      throw new Error("Pieces model not loaded");
    }
    
    processingRef.current = true;
    setIsProcessing(true);
    
    try {
      // Preprocess image
      const inputTensor = await preprocessImage(imageUri, INPUT_SIZE);
      const tensor = new ort.Tensor("float32", inputTensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      
      // Run pieces detection
      const outputsPieces = await piecesSession.run({ images: tensor });
      const pieceDetections = postprocessPieces(outputsPieces, confidenceThreshold);
      
      // Format detections with labels
      let detections: Detection[] = pieceDetections.map((det: PieceDetection) => {
        const display = getPieceDisplay(det.classId);
        return {
          bbox: det.bbox,
          confidence: det.confidence,
          classId: det.classId,
          label: display.label,
          symbol: display.symbol,
          color: display.color,
        };
      });
      
      // Apply smart post-processing (NMS + chess constraints)
      const { smartPostProcess } = await import("../src/utils/smartPostProcess");
      detections = smartPostProcess(detections) as Detection[];
      
      // Optionally detect board
      if (detectBoard && boardSession) {
        const outputsBoard = await boardSession.run({ images: tensor });
        const boardDetection = postprocessBoard(outputsBoard);
        
        if (boardDetection) {
          detections.push({
            bbox: boardDetection.bbox as [number, number, number, number],
            confidence: boardDetection.confidence,
            class: "chessboard",
            polygon: boardDetection.polygon,
            color: "#00FFFF",
          });
        }
      }
      
      return detections;
    } finally {
      processingRef.current = false;
      setIsProcessing(false);
    }
  }, [modelState, loadModels]);

  return {
    modelState,
    isProcessing,
    error,
    loadModels,
    runInference,
    inputSize: INPUT_SIZE,
  };
}
