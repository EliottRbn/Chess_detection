import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
  Alert,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

import { ChessBoard2D } from '../../src/components/ChessBoard2D';
import { DebugOverlay } from '../../src/components/DebugOverlay';
import { BoardState, createEmptyBoard, PieceDetection } from '../../src/types';
import { preprocessBoardImage, preprocessImage } from '../../src/utils/imagePreprocess';
import { loadModels, detectBoardCorners, detectPieces } from '../../src/utils/onnxService';
import { mapPiecesToBoard } from '../../src/utils/boardMapping';
import { BoardCorners } from '../../src/utils/yoloParser';

type AppState = 'camera' | 'preview' | 'analyzing' | 'result';

interface DetectionResult {
  pieces: PieceDetection[];
  boardCorners: BoardCorners | null;
}

export default function AnalyseScreen() {
  const [state, setState] = useState<AppState>('camera');
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [boardState, setBoardState] = useState<BoardState>(createEmptyBoard());
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  // Load ONNX models on mount
  useEffect(() => {
    async function initModels() {
      try {
        await loadModels();
        setModelsLoaded(true);
      } catch (error) {
        console.error('Failed to load models:', error);
        Alert.alert('Erreur', 'Impossible de charger les modèles de détection');
      }
    }
    initModels();
  }, []);

  const takePhoto = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
      });
      if (photo?.uri) {
        setPhotoUri(photo.uri);
        setState('preview');
      }
    } catch (error) {
      console.error('Failed to take photo:', error);
      Alert.alert('Erreur', 'Impossible de prendre la photo');
    }
  };

  const analyzePhoto = async () => {
    if (!photoUri || !modelsLoaded) return;

    setState('analyzing');

    try {
      console.log('[Analyse] Starting...');

      // 1. Preprocess for Board Detection (256x256)
      const boardTensor = await preprocessBoardImage(photoUri);
      console.log('[Analyse] Board image preprocessed (256x256)');

      // 2. Detect Board Corners
      // const boardCorners = await detectBoardCorners(boardTensor);
      const boardCorners = null; // FORCE FALLBACK
      console.log('[Analyse] Board corners:', boardCorners ? 'found' : 'not found (using fallback)');

      // 3. Preprocess for Piece Detection (640x640)
      const pieceImage = await preprocessImage(photoUri, 640);
      console.log('[Analyse] Piece image preprocessed (640x640)');

      // 4. Detect Pieces
      const pieces = await detectPieces(pieceImage.tensor);
      console.log('[Analyse] Piece detection done:', pieces.length, 'pieces');

      // Store detection result for debug display
      setDetectionResult({
        pieces,
        boardCorners,
      });

      // Map pieces to board using perspective transformation
      const mappedBoard = mapPiecesToBoard(pieces, boardCorners);
      setBoardState(mappedBoard);

      setState('result');
    } catch (error) {
      console.error('[Analyse] Failed:', error);
      Alert.alert('Erreur', 'Analyse échouée: ' + (error as Error).message);
      setState('preview');
    }
  };

  const reset = () => {
    setPhotoUri(null);
    setBoardState(createEmptyBoard());
    setDetectionResult(null);
    setState('camera');
  };

  // Permission handling
  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.center}>
          <Text style={styles.title}>Permission Caméra</Text>
          <Text style={styles.subtitle}>L'app a besoin d'accéder à la caméra</Text>
          <TouchableOpacity style={styles.button} onPress={requestPermission}>
            <Text style={styles.buttonText}>Autoriser</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Camera View */}
      {state === 'camera' && (
        <View style={styles.cameraContainer}>
          <CameraView 
            ref={cameraRef} 
            style={styles.camera}
            facing="back"
          >
            <View style={styles.overlay}>
              <View style={styles.guideFrame} />
            </View>
          </CameraView>
          
          <View style={styles.controls}>
            {!modelsLoaded && (
              <View style={styles.loadingModels}>
                <ActivityIndicator color="#fff" />
                <Text style={styles.loadingText}>Chargement des modèles...</Text>
              </View>
            )}
            <TouchableOpacity
              style={[styles.captureButton, !modelsLoaded && styles.disabled]}
              onPress={takePhoto}
              disabled={!modelsLoaded}
            >
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Photo Preview */}
      {state === 'preview' && photoUri && (
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Image source={{ uri: photoUri }} style={styles.previewImage} />
          <View style={styles.previewButtons}>
            <TouchableOpacity style={styles.secondaryButton} onPress={reset}>
              <Text style={styles.secondaryButtonText}>Reprendre</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={analyzePhoto}>
              <Text style={styles.buttonText}>Analyser</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      )}

      {/* Analyzing State */}
      {state === 'analyzing' && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#4A90D9" />
          <Text style={styles.analyzingText}>Analyse en cours...</Text>
        </View>
      )}

      {/* Result View */}
      {state === 'result' && (
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Text style={styles.title}>Échiquier Détecté</Text>
          
          {/* Chess Board 2D */}
          <View style={styles.boardContainer}>
            <ChessBoard2D boardState={boardState} size={300} />
          </View>

          {/* Debug Overlay - Image with detection boxes */}
          {photoUri && detectionResult && (
            <View style={styles.debugSection}>
              <Text style={styles.debugTitle}>Debug - Détections</Text>
              <DebugOverlay
                photoUri={photoUri}
                pieces={detectionResult.pieces}
                boardCorners={detectionResult.boardCorners}
                displaySize={300}
              />
            </View>
          )}

          <TouchableOpacity style={styles.button} onPress={reset}>
            <Text style={styles.buttonText}>Nouvelle Photo</Text>
          </TouchableOpacity>
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#aaa',
    marginBottom: 20,
    textAlign: 'center',
  },
  
  // Camera styles
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  guideFrame: {
    width: 280,
    height: 280,
    borderWidth: 2,
    borderColor: '#4A90D9',
    borderRadius: 8,
  },
  controls: {
    height: 120,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingModels: {
    flexDirection: 'row',
    alignItems: 'center',
    position: 'absolute',
    top: 10,
  },
  loadingText: {
    color: '#fff',
    marginLeft: 10,
    fontSize: 14,
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#4A90D9',
  },
  disabled: {
    opacity: 0.5,
  },

  // Preview styles
  previewImage: {
    width: 320,
    height: 320,
    borderRadius: 8,
    marginBottom: 20,
  },
  previewButtons: {
    flexDirection: 'row',
    gap: 15,
  },

  // Buttons
  button: {
    backgroundColor: '#4A90D9',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    marginTop: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    borderWidth: 2,
    borderColor: '#4A90D9',
  },
  secondaryButtonText: {
    color: '#4A90D9',
    fontSize: 16,
    fontWeight: '600',
  },

  // Analyzing
  analyzingText: {
    color: '#fff',
    fontSize: 18,
    marginTop: 20,
  },

  // Result
  boardContainer: {
    marginVertical: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },

  // Debug section
  debugSection: {
    marginTop: 30,
    width: '100%',
    alignItems: 'center',
  },
  debugTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4A90D9',
    marginBottom: 15,
  },
});