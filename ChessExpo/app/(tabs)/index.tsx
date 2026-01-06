import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  ScrollView,
  Alert,
  Switch,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ChessBoard2D } from '../../src/components/ChessBoard2D';
import { BoardState, createEmptyBoard } from '../../src/types';
import { parseFEN } from '../../src/utils/fenParser';
import { saveToHistory } from './history';

const SERVER_URL = 'http://192.168.1.63:7860';
const { width: SCREEN_WIDTH } = Dimensions.get('window');

type AppState = 'camera' | 'preview' | 'analyzing' | 'result';

export default function ChessAnalyzerScreen() {
  const [state, setState] = useState<AppState>('camera');
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [boardState, setBoardState] = useState<BoardState>(createEmptyBoard());
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  
  const [isWhiteTurn, setIsWhiteTurn] = useState(true);
  const [bestMove, setBestMove] = useState<string | null>(null);
  const [fen, setFen] = useState<string>('');

  const takePhoto = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      if (photo?.uri) {
        setPhotoUri(photo.uri);
        setBestMove(null);
        setState('preview');
      }
    } catch (error) {
      Alert.alert('Erreur', 'Impossible de prendre la photo');
    }
  };

  const pickFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.8,
      });
      if (!result.canceled && result.assets[0]) {
        setPhotoUri(result.assets[0].uri);
        setBestMove(null);
        setState('preview');
      }
    } catch (error) {
      Alert.alert('Erreur', 'Impossible de charger l\'image');
    }
  };

  const analyzePhoto = async () => {
    if (!photoUri) return;
    setState('analyzing');

    try {
      const formData = new FormData();
      formData.append('file', {
        uri: photoUri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      } as any);

      const turn = isWhiteTurn ? 'w' : 'b';
      const response = await fetch(`${SERVER_URL}/detect_fen?turn=${turn}`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.ok) throw new Error(`Erreur serveur: ${response.status}`);

      const data = await response.json();

      if (data.fen) {
        const newBoardState = parseFEN(data.fen);
        setBoardState(newBoardState);
        setBestMove(data.best_move || null);
        setFen(data.fen);
        setState('result');
        
        // Save to history
        saveToHistory({
          fen: data.fen,
          bestMove: data.best_move || null,
          turn: turn as 'w' | 'b',
        });
      } else {
        throw new Error('Aucune position détectée');
      }
    } catch (error) {
      Alert.alert('Erreur', (error as Error).message);
      setState('preview');
    }
  };

  const reset = () => {
    setPhotoUri(null);
    setBoardState(createEmptyBoard());
    setBestMove(null);
    setFen('');
    setState('camera');
  };

  const parseBestMove = (move: string | null) => {
    if (!move || move.length < 4) return null;
    return { from: move.substring(0, 2), to: move.substring(2, 4) };
  };

  if (!permission) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#6366f1" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.container}>
        <SafeAreaView style={styles.centerContent}>
          <Ionicons name="camera-outline" size={80} color="#6366f1" />
          <Text style={styles.permissionTitle}>Accès Caméra</Text>
          <Text style={styles.permissionText}>
            L'application a besoin d'accéder à votre caméra pour analyser les positions d'échecs
          </Text>
          <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
            <Text style={styles.primaryButtonText}>Autoriser l'accès</Text>
          </TouchableOpacity>
        </SafeAreaView>
      </LinearGradient>
    );
  }

  const moveData = parseBestMove(bestMove);

  return (
    <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.container}>
      <SafeAreaView style={styles.safeArea} edges={['top']}>
        
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>♟️ Chess Vision</Text>
          {state !== 'camera' && (
            <TouchableOpacity onPress={reset} style={styles.headerButton}>
              <Ionicons name="close" size={24} color="#fff" />
            </TouchableOpacity>
          )}
        </View>

        {/* Camera View */}
        {state === 'camera' && (
          <View style={styles.cameraWrapper}>
            <CameraView ref={cameraRef} style={styles.camera} facing="back">
              <View style={styles.cameraOverlay}>
                <View style={styles.scanFrame}>
                  <View style={[styles.corner, styles.cornerTL]} />
                  <View style={[styles.corner, styles.cornerTR]} />
                  <View style={[styles.corner, styles.cornerBL]} />
                  <View style={[styles.corner, styles.cornerBR]} />
                </View>
                <Text style={styles.scanHint}>Cadrez l'échiquier</Text>
              </View>
            </CameraView>
            
            <View style={styles.cameraControls}>
              <TouchableOpacity style={styles.iconButton} onPress={pickFromGallery}>
                <Ionicons name="images" size={26} color="#fff" />
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.captureButton} onPress={takePhoto}>
                <View style={styles.captureInner} />
              </TouchableOpacity>
              
              <View style={styles.iconButtonPlaceholder} />
            </View>
          </View>
        )}

        {/* Preview */}
        {state === 'preview' && photoUri && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <View style={styles.previewCard}>
              <Image source={{ uri: photoUri }} style={styles.previewImage} />
            </View>
            
            <View style={styles.turnCard}>
              <Text style={styles.turnTitle}>Qui joue ?</Text>
              <View style={styles.turnRow}>
                <Text style={[styles.turnLabel, !isWhiteTurn && styles.turnActive]}>
                  ♟ Noirs
                </Text>
                <Switch
                  value={isWhiteTurn}
                  onValueChange={setIsWhiteTurn}
                  trackColor={{ false: '#4b5563', true: '#6366f1' }}
                  thumbColor="#fff"
                  style={styles.turnSwitch}
                />
                <Text style={[styles.turnLabel, isWhiteTurn && styles.turnActive]}>
                  ♙ Blancs
                </Text>
              </View>
            </View>
            
            <View style={styles.buttonRow}>
              <TouchableOpacity style={styles.secondaryButton} onPress={reset}>
                <Text style={styles.secondaryButtonText}>Reprendre</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.primaryButton} onPress={analyzePhoto}>
                <Ionicons name="sparkles" size={18} color="#fff" style={{ marginRight: 8 }} />
                <Text style={styles.primaryButtonText}>Analyser</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        )}

        {/* Analyzing */}
        {state === 'analyzing' && (
          <View style={styles.centerContent}>
            <ActivityIndicator size="large" color="#6366f1" />
            <Text style={styles.analyzingText}>Analyse en cours...</Text>
            <Text style={styles.analyzingSubtext}>Détection des pièces et calcul du meilleur coup</Text>
          </View>
        )}

        {/* Result */}
        {state === 'result' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <View style={styles.boardCard}>
              <ChessBoard2D 
                boardState={boardState} 
                size={SCREEN_WIDTH - 60} 
                highlightFrom={moveData?.from}
                highlightTo={moveData?.to}
              />
            </View>

            {bestMove ? (
              <View style={styles.moveCard}>
                <View style={styles.moveHeader}>
                  <Ionicons name="bulb" size={24} color="#fbbf24" />
                  <Text style={styles.moveTitle}>Meilleur coup</Text>
                </View>
                <Text style={styles.moveText}>
                  {moveData?.from?.toUpperCase()} → {moveData?.to?.toUpperCase()}
                </Text>
              </View>
            ) : (
              <View style={styles.noMoveCard}>
                <Ionicons name="information-circle" size={20} color="#9ca3af" />
                <Text style={styles.noMoveText}>Aucun coup suggéré</Text>
              </View>
            )}

            <View style={styles.fenCard}>
              <Text style={styles.fenLabel}>Position FEN</Text>
              <Text style={styles.fenText}>{fen}</Text>
            </View>

            <TouchableOpacity style={styles.primaryButton} onPress={reset}>
              <Ionicons name="camera" size={18} color="#fff" style={{ marginRight: 8 }} />
              <Text style={styles.primaryButtonText}>Nouvelle analyse</Text>
            </TouchableOpacity>
          </ScrollView>
        )}
      </SafeAreaView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#1a1a2e' },
  centerContent: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 30 },
  scrollContent: { padding: 20, paddingBottom: 40 },
  
  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  headerTitle: { fontSize: 22, fontWeight: '700', color: '#fff' },
  headerButton: { padding: 8 },

  // Permission
  permissionTitle: { fontSize: 24, fontWeight: '700', color: '#fff', marginTop: 20 },
  permissionText: { fontSize: 16, color: '#9ca3af', textAlign: 'center', marginTop: 10, marginBottom: 30, paddingHorizontal: 20 },

  // Camera
  cameraWrapper: { flex: 1 },
  camera: { flex: 1 },
  cameraOverlay: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.4)' },
  scanFrame: { width: 280, height: 280, position: 'relative' },
  corner: { position: 'absolute', width: 40, height: 40, borderColor: '#6366f1', borderWidth: 4 },
  cornerTL: { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0, borderTopLeftRadius: 12 },
  cornerTR: { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0, borderTopRightRadius: 12 },
  cornerBL: { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0, borderBottomLeftRadius: 12 },
  cornerBR: { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0, borderBottomRightRadius: 12 },
  scanHint: { color: '#fff', fontSize: 16, marginTop: 20, opacity: 0.8 },
  
  cameraControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 50,
    paddingVertical: 30,
    backgroundColor: 'rgba(26,26,46,0.95)',
  },
  iconButton: {
    width: 50, height: 50, borderRadius: 25,
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center', alignItems: 'center',
  },
  iconButtonPlaceholder: { width: 50, height: 50 },
  captureButton: {
    width: 75, height: 75, borderRadius: 38,
    backgroundColor: '#fff',
    justifyContent: 'center', alignItems: 'center',
  },
  captureInner: {
    width: 62, height: 62, borderRadius: 31,
    backgroundColor: '#fff',
    borderWidth: 4, borderColor: '#1a1a2e',
  },

  // Preview
  previewCard: {
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 20,
    shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8, elevation: 8,
  },
  previewImage: { width: '100%', aspectRatio: 1, backgroundColor: '#2a2a3e' },
  
  turnCard: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    alignItems: 'center',
  },
  turnTitle: { color: '#fff', fontSize: 16, fontWeight: '600', marginBottom: 15 },
  turnRow: { flexDirection: 'row', alignItems: 'center' },
  turnLabel: { color: '#6b7280', fontSize: 16, width: 80, textAlign: 'center' },
  turnActive: { color: '#fff', fontWeight: '600' },
  turnSwitch: { marginHorizontal: 15 },

  // Buttons
  buttonRow: { flexDirection: 'row', gap: 12 },
  primaryButton: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#6366f1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  secondaryButton: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.1)',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButtonText: { color: '#fff', fontSize: 16 },

  // Analyzing
  analyzingText: { color: '#fff', fontSize: 18, fontWeight: '600', marginTop: 20 },
  analyzingSubtext: { color: '#9ca3af', fontSize: 14, marginTop: 8 },

  // Result
  boardCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 10,
    marginBottom: 20,
    alignItems: 'center',
  },
  moveCard: {
    backgroundColor: 'rgba(251,191,36,0.15)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 15,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(251,191,36,0.3)',
  },
  moveHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  moveTitle: { color: '#fbbf24', fontSize: 16, fontWeight: '600', marginLeft: 8 },
  moveText: { color: '#fff', fontSize: 32, fontWeight: '700', letterSpacing: 2 },
  noMoveCard: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 15,
    marginBottom: 15,
    alignItems: 'center',
    justifyContent: 'center',
  },
  noMoveText: { color: '#9ca3af', fontSize: 14, marginLeft: 8 },
  fenCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 15,
    marginBottom: 20,
  },
  fenLabel: { color: '#9ca3af', fontSize: 12, marginBottom: 5 },
  fenText: { color: '#fff', fontSize: 12, fontFamily: 'monospace' },
});
