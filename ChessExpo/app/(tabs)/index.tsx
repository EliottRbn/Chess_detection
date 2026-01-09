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
import { Chess, Move } from 'chess.js';

import { ChessBoard2D } from '../../src/components/ChessBoard2D';
import { BoardState, PieceType, createEmptyBoard } from '../../src/types';
import { parseFEN } from '../../src/utils/fenParser';
import { saveToHistory } from './history';

//const SERVER_URL = 'https://raphalp-chess-fen-detection.hf.space';
const SERVER_URL = 'http://localhost:7860';
const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Piece images for selector
const PIECE_IMAGES: Record<string, any> = {
  'K': require('../../assets/pieces/white_king.png'),
  'Q': require('../../assets/pieces/white_queen.png'),
  'R': require('../../assets/pieces/white_rook.png'),
  'B': require('../../assets/pieces/white_bishop.png'),
  'N': require('../../assets/pieces/white_knight.png'),
  'P': require('../../assets/pieces/white_pawn.png'),
  'k': require('../../assets/pieces/black_king.png'),
  'q': require('../../assets/pieces/black_queen.png'),
  'r': require('../../assets/pieces/black_rook.png'),
  'b': require('../../assets/pieces/black_bishop.png'),
  'n': require('../../assets/pieces/black_knight.png'),
  'p': require('../../assets/pieces/black_pawn.png'),
};

type AppState = 'camera' | 'preview' | 'analyzing' | 'result' | 'verification_needed' | 'recording' | 'playing';
type CaptureMode = 'photo' | 'video';

export default function ChessAnalyzerScreen() {
  const [state, setState] = useState<AppState>('camera');
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [boardState, setBoardState] = useState<BoardState>(createEmptyBoard());
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  
  const [isWhiteTurn, setIsWhiteTurn] = useState(true);
  const [bestMove, setBestMove] = useState<string | null>(null);
  const [fen, setFen] = useState<string>('');
  
  // Multi-photo verification state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [verificationReason, setVerificationReason] = useState<string>('');
  const [confidenceInfo, setConfidenceInfo] = useState<{avg: number, min: number, lowCount: number} | null>(null);

  // Video mode state
  const [captureMode, setCaptureMode] = useState<CaptureMode>('photo');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const VIDEO_MAX_DURATION = 5; // 5 seconds

  // Piece correction state
  const [showBestMove, setShowBestMove] = useState(false);  // Hide best move until user validates
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);  // e.g. "e4"
  const [hasChanges, setHasChanges] = useState(false);  // Track if user modified pieces
  const [showPieceSelector, setShowPieceSelector] = useState(false);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const [showMoveHighlight, setShowMoveHighlight] = useState(false);  // Toggle best move visibility on board

  // Game state
  const chessInstance = useRef<Chess | null>(null);
  const [legalMoves, setLegalMoves] = useState<string[]>([]);
  const [promotionSquare, setPromotionSquare] = useState<{from: string, to: string} | null>(null);
  const [showPromotionModal, setShowPromotionModal] = useState(false);
  const [isGameOver, setIsGameOver] = useState(false);

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

  // Video recording functions
  const startRecording = async () => {
    if (!cameraRef.current) return;
    try {
      setIsRecording(true);
      setState('recording');
      setRecordingTime(0);
      
      // Start timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= VIDEO_MAX_DURATION - 1) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
      
      const video = await cameraRef.current.recordAsync({
        maxDuration: VIDEO_MAX_DURATION,
      });
      
      if (video?.uri) {
        analyzeVideo(video.uri);
      }
    } catch (error) {
      console.error('Recording error:', error);
      Alert.alert('Erreur', 'Impossible d\'enregistrer la vidéo');
      setIsRecording(false);
      setState('camera');
    }
  };

  const stopRecording = async () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    setIsRecording(false);
    
    if (cameraRef.current) {
      cameraRef.current.stopRecording();
    }
  };

  const analyzeVideo = async (videoUri: string) => {
    setState('analyzing');
    
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: videoUri,
        name: 'video.mp4',
        type: 'video/mp4',
      } as any);

      const turn = isWhiteTurn ? 'w' : 'b';
      const response = await fetch(`${SERVER_URL}/detect_fen_video?turn=${turn}&max_frames=15`, {
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
      setState('camera');
    }
  };

  const pickVideoFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['videos'],
        quality: 0.8,
      });
      
      if (!result.canceled && result.assets[0]) {
        const video = result.assets[0];
        console.log('[Video] Selected video:', video.uri, 'duration:', video.duration);
        
        // Duration is in milliseconds in expo-image-picker
        const durationSeconds = video.duration ? video.duration / 1000 : 0;
        console.log('[Video] Duration in seconds:', durationSeconds);
        
        if (durationSeconds > VIDEO_MAX_DURATION) {
          Alert.alert(
            'Vidéo trop longue',
            `Veuillez sélectionner une vidéo de ${VIDEO_MAX_DURATION} secondes maximum. (Durée: ${Math.round(durationSeconds)}s)`,
            [{ text: 'OK' }]
          );
          return;
        }
        
        analyzeVideo(video.uri);
      }
    } catch (error) {
      console.error('[Video] Gallery error:', error);
      Alert.alert('Erreur', 'Impossible de charger la vidéo');
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
        // Always go directly to result (no verification needed)
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

  const takeVerificationPhoto = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      if (photo?.uri) {
        analyzeVerificationPhoto(photo.uri);
      }
    } catch (error) {
      Alert.alert('Erreur', 'Impossible de prendre la photo');
    }
  };

  const pickVerificationFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.8,
      });
      if (!result.canceled && result.assets[0]) {
        analyzeVerificationPhoto(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Erreur', 'Impossible de charger l\'image');
    }
  };

  const analyzeVerificationPhoto = async (verifyPhotoUri: string) => {
    if (!sessionId) {
      Alert.alert('Erreur', 'Session expirée, veuillez recommencer');
      reset();
      return;
    }
    
    setState('analyzing');

    try {
      const formData = new FormData();
      formData.append('file', {
        uri: verifyPhotoUri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      } as any);

      const response = await fetch(`${SERVER_URL}/detect_fen_verify?session_id=${sessionId}`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.ok) {
        if (response.status === 410) {
          throw new Error('Session expirée, veuillez recommencer');
        }
        throw new Error(`Erreur serveur: ${response.status}`);
      }

      const data = await response.json();

      if (data.fen) {
        const newBoardState = parseFEN(data.fen);
        setBoardState(newBoardState);
        setBestMove(data.best_move || null);
        setFen(data.fen);
        setSessionId(null);
        setState('result');
        
        // Save to history
        const turn = isWhiteTurn ? 'w' : 'b';
        saveToHistory({
          fen: data.fen,
          bestMove: data.best_move || null,
          turn: turn as 'w' | 'b',
        });
      } else {
        throw new Error('Erreur lors de la vérification');
      }
    } catch (error) {
      Alert.alert('Erreur', (error as Error).message);
      setState('verification_needed');
    }
  };

  const skipVerification = () => {
    // User accepts preliminary result without second photo
    const turn = isWhiteTurn ? 'w' : 'b';
    saveToHistory({
      fen: fen,
      bestMove: bestMove || null,
      turn: turn as 'w' | 'b',
    });
    setSessionId(null);
    setState('result');
  };

  const reset = () => {
    setPhotoUri(null);
    setBoardState(createEmptyBoard());
    setBestMove(null);
    setFen('');
    setSessionId(null);
    setVerificationReason('');
    setConfidenceInfo(null);
    setShowBestMove(false);
    setSelectedSquare(null);
    setHasChanges(false);
    setShowPieceSelector(false);
    setShowMoveHighlight(false);
    setShowMoveHighlight(false);
    chessInstance.current = null;
    setState('camera');
  };

  // Piece correction functions
  const handleSquarePress = (square: string) => {
    setSelectedSquare(square);
    setShowPieceSelector(true);
  };

  const handlePieceSelect = (piece: PieceType) => {
    if (!selectedSquare) return;
    
    // Convert square notation (e.g. "e4") to row/col
    const col = selectedSquare.charCodeAt(0) - 'a'.charCodeAt(0);
    const row = 8 - parseInt(selectedSquare[1]);
    
    // Update board state
    const newBoard: BoardState = boardState.map((r, ri) => 
      r.map((cell, ci) => {
        if (ri === row && ci === col) {
          return piece;
        }
        return cell;
      })
    );
    
    setBoardState(newBoard);
    setHasChanges(true);
    setShowPieceSelector(false);
    setSelectedSquare(null);
    
    // Update FEN string
    const newFen = boardToFEN(newBoard);
    setFen(newFen);
  };

  const startPlaying = () => {
    try {
      // Initialize chess instance with current FEN
      // Defaulting castling/enpassant to allowed/none as we can't detect them from static image
      // Standard: [placement] [turn] [castling] [enpassant] [halfmove] [fullmove]
      // We start with full castling rights if kings/rooks are in place to allow play, or '-' if unsafe.
      // For simplicity in "continue from image", we might default to no castling to avoid illegal move errors if rooks moved.
      // Let's rely on simple FEN for now:
      const safeFen = `${fen} ${isWhiteTurn ? 'w' : 'b'} - - 0 1`;
      chessInstance.current = new Chess(safeFen);
      
      setState('playing');
      setLegalMoves([]);
      setShowMoveHighlight(false); // Don't show hint initially
      setIsGameOver(false);
    } catch (e) {
      console.error('Error starting game:', e);
      Alert.alert('Erreur', 'Impossible de démarrer la partie avec cette position');
    }
  };

  const handleGameSquarePress = (square: string) => {
    if (!chessInstance.current) return;
    const chess = chessInstance.current;

    // If we have a selected square (source), try to move
    if (selectedSquare) {
      // If tapped on same square, deselect
      if (square === selectedSquare) {
        setSelectedSquare(null);
        setLegalMoves([]);
        return;
      }

      // Check if move is legal
      const moves = chess.moves({ verbose: true });
      const move = moves.find(m => m.from === selectedSquare && m.to === square);

      if (move) {
        // Valid move found
        if (move.flags.includes('p')) {
          // Promotion needed
          setPromotionSquare({ from: selectedSquare, to: square });
          setShowPromotionModal(true);
        } else {
          makeMove({ from: selectedSquare, to: square });
        }
      } else {
        // Not a valid move. 
        // If tapped on own piece, switch selection
        const piece = chess.get(square as any);
        if (piece && piece.color === (chess.turn())) {
          setSelectedSquare(square);
          const newMoves = chess.moves({ square: square as any, verbose: true });
          setLegalMoves(newMoves.map(m => m.to));
        } else {
          // Tapped empty or enemy square (invalid move) -> Deselect
          setSelectedSquare(null);
          setLegalMoves([]);
        }
      }
    } else {
      // No selection - try to select a piece
      const piece = chess.get(square as any);
      if (piece && piece.color === (chess.turn())) {
        setSelectedSquare(square);
        const moves = chess.moves({ square: square as any, verbose: true });
        setLegalMoves(moves.map(m => m.to));
      }
    }
  };

  const makeMove = (move: { from: string, to: string, promotion?: string }) => {
    if (!chessInstance.current) return;
    const chess = chessInstance.current;

    try {
      const result = chess.move(move);
      if (result) {
        // Move successful
        updateBoardFromChess();
        setSelectedSquare(null);
        setLegalMoves([]);
        
        // Check game over
        if (chess.isGameOver()) {
          setIsGameOver(true);
          let reason = '';
          if (chess.isCheckmate()) reason = 'Echec et mat !';
          else if (chess.isDraw()) reason = 'Match nul !';
          else if (chess.isStalemate()) reason = 'Pat !';
          
          Alert.alert('Fin de partie', reason);
        } else {
            // Trigger best move calculation for new position
            calculateBestMove(chess.fen(), chess.turn());
        }
      }
    } catch (e) {
      console.error('Move error:', e);
    }
  };

  const onPromotionSelect = (piece: string) => { // 'q', 'r', 'b', 'n'
    if (promotionSquare) {
      makeMove({ ...promotionSquare, promotion: piece });
      setShowPromotionModal(false);
      setPromotionSquare(null);
    }
  };

  const calculateBestMove = async (currentFen: string, turn: string) => {
    try {
        const response = await fetch(
          `${SERVER_URL}/calculate_move?fen=${encodeURIComponent(currentFen)}&turn=${turn}`,
          { method: 'POST' }
        );
        if (response.ok) {
            const data = await response.json();
            setBestMove(data.best_move || null);
        }
    } catch (e) {
        console.error("Best move calc error", e);
    }
  };

  const updateBoardFromChess = () => {
    if (!chessInstance.current) return;
    const chess = chessInstance.current;
    
    // Convert chess.js board to our BoardState
    // chess.board() returns 8x8 array of { type: 'p', color: 'w' } | null
    const rawBoard = chess.board();
    const newBoard: BoardState = rawBoard.map(row => 
      row.map(cell => {
        if (!cell) return null;
        // Convert { type: 'p', color: 'w' } -> 'P' or 'p'
        return cell.color === 'w' ? cell.type.toUpperCase() as PieceType : cell.type as PieceType;
      })
    );

    setBoardState(newBoard);
    setFen(chess.fen());
    setIsWhiteTurn(chess.turn() === 'w');
  };

  const boardToFEN = (board: BoardState): string => {
    const rows = board.map(row => {
      let fenRow = '';
      let emptyCount = 0;
      
      for (const cell of row) {
        if (cell === null) {
          emptyCount++;
        } else {
          if (emptyCount > 0) {
            fenRow += emptyCount;
            emptyCount = 0;
          }
          fenRow += cell;
        }
      }
      
      if (emptyCount > 0) {
        fenRow += emptyCount;
      }
      
      return fenRow;
    });
    
    return rows.join('/');
  };

  const handleValidate = async () => {
    if (hasChanges) {
      // Need to recalculate best move
      setIsRecalculating(true);
      try {
        const turn = isWhiteTurn ? 'w' : 'b';
        const response = await fetch(
          `${SERVER_URL}/calculate_move?fen=${encodeURIComponent(fen)}&turn=${turn}`,
          { method: 'POST' }
        );
        
        if (!response.ok) throw new Error('Erreur serveur');
        
        const data = await response.json();
        setBestMove(data.best_move || null);
      } catch (error) {
        console.error('Recalculate error:', error);
        // Still show board, just without move
      }
      setIsRecalculating(false);
    }
    
    // Show best move
    setShowBestMove(true);
    
    // Save to history
    const turn = isWhiteTurn ? 'w' : 'b';
    saveToHistory({
      fen: fen,
      bestMove: bestMove || null,
      turn: turn as 'w' | 'b',
    });
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
          <TouchableOpacity style={[styles.primaryButton, { flex: 0, width: '100%' }]} onPress={requestPermission}>
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
          <Image source={require('../../assets/logoapp.png')} style={styles.headerLogo} resizeMode="contain" />
          {state !== 'camera' && (
            <TouchableOpacity onPress={reset} style={styles.headerButton}>
              <Ionicons name="close" size={24} color="#fff" />
            </TouchableOpacity>
          )}
        </View>

        {/* Camera View */}
        {(state === 'camera' || state === 'recording') && (
          <View style={styles.cameraWrapper}>
            {/* Mode Selector */}
            <View style={styles.modeSelector}>
              <TouchableOpacity 
                style={[styles.modeButton, captureMode === 'photo' && styles.modeButtonActive]}
                onPress={() => setCaptureMode('photo')}
                disabled={isRecording}
              >
                <Ionicons name="camera" size={20} color={captureMode === 'photo' ? '#fff' : '#9ca3af'} />
                <Text style={[styles.modeButtonText, captureMode === 'photo' && styles.modeButtonTextActive]}>
                  Photo
                </Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={[styles.modeButton, captureMode === 'video' && styles.modeButtonActive]}
                onPress={() => setCaptureMode('video')}
                disabled={isRecording}
              >
                <Ionicons name="videocam" size={20} color={captureMode === 'video' ? '#fff' : '#9ca3af'} />
                <Text style={[styles.modeButtonText, captureMode === 'video' && styles.modeButtonTextActive]}>
                  Vidéo
                </Text>
              </TouchableOpacity>
            </View>

            <CameraView 
              ref={cameraRef} 
              style={styles.camera} 
              facing="back"
              mode={captureMode === 'video' ? 'video' : 'picture'}
            >
              <View style={styles.cameraOverlay}>
                <View style={styles.scanFrame}>
                  <View style={[styles.corner, styles.cornerTL]} />
                  <View style={[styles.corner, styles.cornerTR]} />
                  <View style={[styles.corner, styles.cornerBL]} />
                  <View style={[styles.corner, styles.cornerBR]} />
                </View>
                {isRecording ? (
                  <View style={styles.recordingIndicator}>
                    <View style={styles.recordingDot} />
                    <Text style={styles.recordingText}>
                      {recordingTime}s / {VIDEO_MAX_DURATION}s
                    </Text>
                  </View>
                ) : (
                  <Text style={styles.scanHint}>
                    {captureMode === 'photo' ? 'Cadrez l\'échiquier' : 'Filmez l\'échiquier (5s)'}
                  </Text>
                )}
              </View>
            </CameraView>
            
            <View style={styles.cameraControls}>
              {/* Gallery button - left side for both modes */}
              <TouchableOpacity 
                style={styles.iconButton} 
                onPress={captureMode === 'photo' ? pickFromGallery : pickVideoFromGallery}
                disabled={isRecording}
              >
                <Ionicons name="images" size={26} color={isRecording ? '#4b5563' : '#fff'} />
              </TouchableOpacity>
              
              {/* Capture button - center */}
              {captureMode === 'photo' ? (
                <TouchableOpacity style={styles.captureButton} onPress={takePhoto}>
                  <View style={styles.captureInner} />
                </TouchableOpacity>
              ) : (
                <TouchableOpacity 
                  style={[styles.captureButton, isRecording && styles.recordingButton]} 
                  onPress={isRecording ? stopRecording : startRecording}
                >
                  {isRecording ? (
                    <View style={styles.stopRecordingInner} />
                  ) : (
                    <View style={styles.videoInner} />
                  )}
                </TouchableOpacity>
              )}
              
              {/* Placeholder - right side */}
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

        {/* Verification Needed */}
        {state === 'verification_needed' && (
          <View style={styles.cameraWrapper}>
            <View style={styles.verificationHeader}>
              <View style={styles.warningBadge}>
                <Ionicons name="warning" size={20} color="#fbbf24" />
                <Text style={styles.warningText}>Vérification recommandée</Text>
              </View>
              <Text style={styles.verificationReason}>{verificationReason}</Text>
              {confidenceInfo && (
                <Text style={styles.confidenceText}>
                  Confiance: {Math.round(confidenceInfo.avg * 100)}% (min: {Math.round(confidenceInfo.min * 100)}%)
                </Text>
              )}
            </View>

            <CameraView ref={cameraRef} style={styles.verificationCamera} facing="back">
              <View style={styles.cameraOverlay}>
                <View style={styles.scanFrame}>
                  <View style={[styles.corner, styles.cornerTL]} />
                  <View style={[styles.corner, styles.cornerTR]} />
                  <View style={[styles.corner, styles.cornerBL]} />
                  <View style={[styles.corner, styles.cornerBR]} />
                </View>
                <Text style={styles.scanHint}>Prenez une 2ème photo sous un angle différent</Text>
              </View>
            </CameraView>
            
            <View style={styles.verificationControls}>
              <TouchableOpacity style={styles.iconButton} onPress={pickVerificationFromGallery}>
                <Ionicons name="images" size={26} color="#fff" />
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.captureButton} onPress={takeVerificationPhoto}>
                <View style={styles.captureInner} />
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.iconButton} onPress={skipVerification}>
                <Ionicons name="checkmark" size={26} color="#4ade80" />
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Result */}
        {state === 'result' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            {/* Lightbulb toggle - above board, right aligned */}
            {showBestMove && bestMove && (
              <View style={styles.boardHeader}>
                <TouchableOpacity 
                  style={[
                    styles.lightbulbButton,
                    showMoveHighlight && styles.lightbulbButtonActive
                  ]}
                  onPress={() => setShowMoveHighlight(!showMoveHighlight)}
                >
                  <Ionicons 
                    name={showMoveHighlight ? "bulb" : "bulb-outline"} 
                    size={24} 
                    color={showMoveHighlight ? "#fff" : "#fbbf24"} 
                  />
                </TouchableOpacity>
              </View>
            )}

            {/* Board */}
            <View style={styles.boardCard}>
              {!showBestMove ? (
                // Editable mode - render clickable squares
                <View style={{ position: 'relative' }}>
                  <ChessBoard2D 
                    boardState={boardState} 
                    size={SCREEN_WIDTH - 60}
                    onSquarePress={handleSquarePress}
                    selectedSquare={selectedSquare}
                  />
                  <Text style={styles.editHint}>
                    Touchez une case pour corriger une pièce
                  </Text>
                </View>
              ) : (
                // Validated - show board with best move highlight
                <ChessBoard2D 
                  boardState={boardState} 
                  size={SCREEN_WIDTH - 60} 
                  highlightFrom={showMoveHighlight ? moveData?.from : undefined}
                  highlightTo={showMoveHighlight ? moveData?.to : undefined}
                />
              )}
            </View>

            {/* Validate button - before validation */}
            {!showBestMove && (
              <TouchableOpacity 
                style={[styles.primaryButton, hasChanges && styles.warningButton]} 
                onPress={handleValidate}
                disabled={isRecalculating}
              >
                {isRecalculating ? (
                  <ActivityIndicator size="small" color="#fff" style={{ marginRight: 8 }} />
                ) : (
                  <Ionicons 
                    name={hasChanges ? "refresh" : "checkmark-circle"} 
                    size={18} 
                    color="#fff" 
                    style={{ marginRight: 8 }} 
                  />
                )}
                <Text style={styles.primaryButtonText}>
                  {isRecalculating ? 'Recalcul...' : hasChanges ? 'Recalculer et Valider' : 'Valider la position'}
                </Text>
              </TouchableOpacity>
            )}

            {/* Action buttons - after validation */}
            {showBestMove && (
              <View style={styles.buttonRow}>
                <TouchableOpacity style={styles.secondaryButton} onPress={reset}>
                  <Ionicons name="camera" size={18} color="#fff" style={{ marginRight: 6 }} />
                  <Text style={styles.secondaryButtonText}>Nouvelle analyse</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.primaryButton} onPress={startPlaying}>
                  <Ionicons name="play" size={18} color="#fff" style={{ marginRight: 6 }} />
                  <Text style={styles.primaryButtonText}>Jouer la partie</Text>
                </TouchableOpacity>
              </View>
            )}

            {/* Piece Selector Modal */}
            {showPieceSelector && selectedSquare && (
              <View style={styles.pieceSelectorOverlay}>
                <View style={styles.pieceSelector}>
                  <Text style={styles.pieceSelectorTitle}>
                    Pièce sur {selectedSquare.toUpperCase()}
                  </Text>
                  
                  {/* Empty */}
                  <TouchableOpacity 
                    style={styles.pieceOption} 
                    onPress={() => handlePieceSelect(null)}
                  >
                    <Text style={styles.pieceOptionText}>Vide</Text>
                  </TouchableOpacity>
                  
                  {/* White pieces */}
                  <Text style={styles.pieceSectionLabel}>Blancs</Text>
                  <View style={styles.pieceRow}>
                    {(['K', 'Q', 'R', 'B', 'N', 'P'] as PieceType[]).map(piece => (
                      <TouchableOpacity 
                        key={piece} 
                        style={styles.pieceButton}
                        onPress={() => handlePieceSelect(piece)}
                      >
                        <Image 
                          source={PIECE_IMAGES[piece!]} 
                          style={styles.pieceImage} 
                          resizeMode="contain"
                        />
                      </TouchableOpacity>
                    ))}
                  </View>
                  
                  {/* Black pieces */}
                  <Text style={styles.pieceSectionLabel}>Noirs</Text>
                  <View style={styles.pieceRow}>
                    {(['k', 'q', 'r', 'b', 'n', 'p'] as PieceType[]).map(piece => (
                      <TouchableOpacity 
                        key={piece} 
                        style={styles.pieceButton}
                        onPress={() => handlePieceSelect(piece)}
                      >
                        <Image 
                          source={PIECE_IMAGES[piece!]} 
                          style={styles.pieceImage} 
                          resizeMode="contain"
                        />
                      </TouchableOpacity>
                    ))}
                  </View>
                  
                  <TouchableOpacity 
                    style={styles.cancelButton} 
                    onPress={() => { setShowPieceSelector(false); setSelectedSquare(null); }}
                  >
                    <Text style={styles.cancelButtonText}>Annuler</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </ScrollView>
        )}

        {/* Playing Mode */}
        {state === 'playing' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            {/* Header with abandon button and lightbulb */}
            <View style={styles.playingHeader}>
              <TouchableOpacity 
                style={[styles.abandonButton, isGameOver && styles.finishButton]}
                onPress={() => setState('result')}
              >
                <Ionicons name={isGameOver ? "checkmark-circle" : "flag"} size={18} color={isGameOver ? "#10b981" : "#ef4444"} />
                <Text style={[styles.abandonButtonText, isGameOver && styles.finishButtonText]}>
                  {isGameOver ? 'Terminer' : 'Abandonner'}
                </Text>
              </TouchableOpacity>
              
              {bestMove && (
                <TouchableOpacity 
                  style={[
                    styles.lightbulbButton,
                    showMoveHighlight && styles.lightbulbButtonActive
                  ]}
                  onPress={() => setShowMoveHighlight(!showMoveHighlight)}
                >
                  <Ionicons 
                    name={showMoveHighlight ? "bulb" : "bulb-outline"} 
                    size={24} 
                    color={showMoveHighlight ? "#fff" : "#fbbf24"} 
                  />
                </TouchableOpacity>
              )}
            </View>

            {/* Board - interactive for playing */}
            <View style={styles.boardCard}>
              <ChessBoard2D 
                boardState={boardState} 
                size={SCREEN_WIDTH - 60} 
                highlightFrom={showMoveHighlight ? moveData?.from : undefined}
                highlightTo={showMoveHighlight ? moveData?.to : undefined}
                onSquarePress={handleGameSquarePress}
                selectedSquare={selectedSquare}
                flipBlackPieces={true}
                legalMoves={legalMoves}
              />
            </View>

            {/* Turn indicator */}
            <View style={styles.turnIndicator}>
              <Text style={styles.turnIndicatorText}>
                {isWhiteTurn ? '♙ Blancs jouent' : '♟ Noirs jouent'}
              </Text>
            </View>

            {/* Promotion Selection Modal */}
            {showPromotionModal && promotionSquare && (
              <View style={styles.pieceSelectorOverlay}>
                <View style={styles.pieceSelector}>
                  <Text style={styles.pieceSelectorTitle}>Promotion</Text>
                  <Text style={styles.pieceSelectorSubtitle}>Choisissez une pièce</Text>
                  
                  <View style={styles.pieceRow}>
                    {(['q', 'r', 'b', 'n'] as const).map(piece => {
                      const displayPiece = isWhiteTurn ? piece.toUpperCase() : piece;
                      return (
                        <TouchableOpacity 
                          key={piece} 
                          style={styles.pieceButton}
                          onPress={() => onPromotionSelect(piece)}
                        >
                          <Image 
                            source={PIECE_IMAGES[displayPiece]} 
                            style={styles.pieceImage} 
                            resizeMode="contain"
                          />
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                  
                  <TouchableOpacity 
                    style={styles.cancelButton} 
                    onPress={() => { setShowPromotionModal(false); setPromotionSquare(null); }}
                  >
                    <Text style={styles.cancelButtonText}>Annuler</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
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
    justifyContent: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
    position: 'relative',
    height: 60,
  },
  headerLogo: {
    width: 160,
    height: 400,
  },
  headerButton: { 
    padding: 8,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 20,
    position: 'absolute',
    right: 20,
  },

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
  boardHeader: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginBottom: 8,
  },
  boardCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 10,
    marginBottom: 20,
    alignItems: 'center',
  },
  lightbulbButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(251,191,36,0.2)',
    borderWidth: 2,
    borderColor: '#fbbf24',
    justifyContent: 'center',
    alignItems: 'center',
  },
  lightbulbButtonActive: {
    backgroundColor: '#f59e0b',
    borderColor: '#f59e0b',
  },
  // Playing mode
  playingHeader: {
    width: '100%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  abandonButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(239,68,68,0.15)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(239,68,68,0.3)',
  },
  abandonButtonText: {
    color: '#ef4444',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 6,
  },
  finishButton: {
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    borderColor: 'rgba(16, 185, 129, 0.3)',
  },
  finishButtonText: {
    color: '#10b981',
  },
  turnIndicator: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  turnIndicatorText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
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

  // Verification
  verificationHeader: {
    backgroundColor: 'rgba(251,191,36,0.1)',
    padding: 15,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(251,191,36,0.2)',
  },
  warningBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(251,191,36,0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginBottom: 8,
  },
  warningText: { color: '#fbbf24', fontSize: 14, fontWeight: '600', marginLeft: 6 },
  verificationReason: { color: '#fff', fontSize: 14, marginBottom: 4 },
  confidenceText: { color: '#9ca3af', fontSize: 12 },
  verificationCamera: { flex: 1 },
  verificationControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 30,
    paddingVertical: 25,
    backgroundColor: 'rgba(26,26,46,0.95)',
  },

  // Mode Selector
  modeSelector: {
    flexDirection: 'row',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(26,26,46,0.98)',
    gap: 12,
  },
  modeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 30,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    gap: 8,
  },
  modeButtonActive: {
    backgroundColor: '#6366f1',
    borderColor: '#6366f1',
    shadowColor: '#6366f1',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  modeButtonText: {
    color: '#9ca3af',
    fontSize: 15,
    fontWeight: '600',
  },
  modeButtonTextActive: {
    color: '#fff',
  },

  // Recording
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 20,
    gap: 8,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#ef4444',
  },
  recordingText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  recordingButton: {
    borderColor: '#ef4444',
    borderWidth: 3,
  },
  stopRecordingInner: {
    width: 30,
    height: 30,
    backgroundColor: '#ef4444',
    borderRadius: 4,
  },
  videoInner: {
    width: 62,
    height: 62,
    borderRadius: 31,
    backgroundColor: '#ef4444',
  },

  // Piece Correction
  editHint: {
    color: '#9ca3af',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 10,
  },
  warningButton: {
    backgroundColor: '#f59e0b',
  },
  pieceSelectorOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 100,
  },
  pieceSelector: {
    backgroundColor: '#1a1a2e',
    borderRadius: 20,
    padding: 24,
    width: '85%',
    maxWidth: 340,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  pieceSelectorTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    textAlign: 'center',
  },
  pieceSelectorSubtitle: {
    color: '#9ca3af',
    fontSize: 14,
    marginBottom: 15,
    textAlign: 'center',
  },
  pieceOption: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 10,
    marginBottom: 15,
    width: '100%',
    alignItems: 'center',
  },
  pieceOptionText: {
    color: '#9ca3af',
    fontSize: 16,
  },
  pieceSectionLabel: {
    color: '#9ca3af',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 8,
    marginTop: 5,
    alignSelf: 'flex-start',
  },
  pieceRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 20,
  },
  pieceButton: {
    width: 44,
    height: 44,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  pieceImage: {
    width: 36,
    height: 36,
  },
  pieceSymbol: {
    fontSize: 28,
  },
  cancelButton: {
    marginTop: 10,
    paddingVertical: 12,
  },
  cancelButtonText: {
    color: '#ef4444',
    fontSize: 16,
    fontWeight: '500',
  },
});
