import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { PieceDetection } from '../types';
import { parsePieceDetections, parseSimpleSegmentation, BoardCorners } from './yoloParser';

let boardModel: InferenceSession | null = null;
let pieceModel: InferenceSession | null = null;

/**
 * Load the ONNX models
 */
export async function loadModels(): Promise<void> {
  try {
    console.log('[ONNX] Loading models...');

    // Load NEW board detection model (256x256)
    const boardAsset = Asset.fromModule(require('../../assets/models/ultimate_v2_breakthrough_accurate.onnx'));
    await boardAsset.downloadAsync();
    if (boardAsset.localUri) {
      boardModel = await InferenceSession.create(boardAsset.localUri);
      console.log('[ONNX] Board model loaded (Ultimate V2)');
      console.log('[ONNX] Board model inputs:', boardModel.inputNames);
    }

    // Load piece detection model (640x640)
    const pieceAsset = Asset.fromModule(require('../../assets/models/detect_pieces_1.onnx'));
    await pieceAsset.downloadAsync();
    if (pieceAsset.localUri) {
      pieceModel = await InferenceSession.create(pieceAsset.localUri);
      console.log('[ONNX] Piece model loaded');
    }

    console.log('[ONNX] All models loaded successfully');
  } catch (error) {
    console.error('[ONNX] Failed to load models:', error);
    throw error;
  }
}

/**
 * Check if models are loaded
 */
export function areModelsLoaded(): boolean {
  return boardModel !== null && pieceModel !== null;
}

/**
 * Detect the chessboard corners using segmentation (256x256 input)
 */
export async function detectBoardCorners(
  imageTensor: Float32Array
): Promise<BoardCorners | null> {
  if (!boardModel) {
    throw new Error('Board model not loaded');
  }

  try {
    console.log('[ONNX] Running board segmentation (256x256)...');
    // Input is [1, 3, 256, 256]
    const inputTensor = new Tensor('float32', imageTensor, [1, 3, 256, 256]);
    const results = await boardModel.run({ input: inputTensor }); // Note: input name is 'input'

    const output = results['output'].data as Float32Array;
    console.log(`[ONNX] Board output size: ${output.length}`);

    return parseSimpleSegmentation(output, 256);
  } catch (error) {
    console.error('[ONNX] Board detection failed:', error);
    return null;
  }
}

/**
 * Detect chess pieces in the image (640x640 input)
 */
export async function detectPieces(imageTensor: Float32Array): Promise<PieceDetection[]> {
  if (!pieceModel) {
    throw new Error('Piece model not loaded');
  }

  try {
    console.log('[ONNX] Running piece detection (640x640)...');
    const inputTensor = new Tensor('float32', imageTensor, [1, 3, 640, 640]);
    const results = await pieceModel.run({ images: inputTensor });

    const output = results['output0'].data as Float32Array;
    
    const detections = parsePieceDetections(output);
    console.log(`[ONNX] Detected ${detections.length} pieces`);

    return detections;
  } catch (error) {
    console.error('[ONNX] Piece detection failed:', error);
    return [];
  }
}
