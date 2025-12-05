
import React, { useState, useRef } from 'react';
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
  TextInput,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';

import { ChessBoard2D } from '../../src/components/ChessBoard2D';
import { BoardState, createEmptyBoard } from '../../src/types';
import { parseFEN } from '../../src/utils/fenParser';

// CHANGE THIS TO YOUR COMPUTER'S IP ADDRESS
const DEFAULT_SERVER_URL = 'http://192.168.1.28:8000';

type AppState = 'camera' | 'preview' | 'analyzing' | 'result';

export default function ServerAnalysisScreen() {
  const [state, setState] = useState<AppState>('camera');
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [boardState, setBoardState] = useState<BoardState>(createEmptyBoard());
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

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

  const pickFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setPhotoUri(result.assets[0].uri);
        setState('preview');
      }
    } catch (error) {
      console.error('Failed to pick image:', error);
      Alert.alert('Erreur', 'Impossible de charger l\'image');
    }
  };

  const analyzePhoto = async () => {
    if (!photoUri) return;

    setState('analyzing');

    try {
      console.log('[Server Analyse] Starting...');
      
      const formData = new FormData();
      formData.append('file', {
        uri: photoUri,
        name: 'photo.jpg',
        type: 'image/jpeg',
      } as any);

      const response = await fetch(`${serverUrl}/detect_fen`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log('[Server Analyse] Response:', data);

      if (data.fen) {
        const newBoardState = parseFEN(data.fen);
        setBoardState(newBoardState);
        setState('result');
      } else {
        throw new Error('No FEN in response');
      }

    } catch (error) {
      console.error('[Server Analyse] Failed:', error);
      Alert.alert('Erreur', 'Analyse serveur échouée: ' + (error as Error).message);
      setState('preview');
    }
  };

  const reset = () => {
    setPhotoUri(null);
    setBoardState(createEmptyBoard());
    setState('camera');
  };

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
            <View style={styles.urlContainer}>
              <Text style={styles.urlLabel}>Server URL:</Text>
              <TextInput 
                style={styles.urlInput}
                value={serverUrl}
                onChangeText={setServerUrl}
                placeholder="http://..."
                placeholderTextColor="#aaa"
              />
            </View>
            <TouchableOpacity
              style={styles.galleryButton}
              onPress={pickFromGallery}
            >
              <Text style={styles.galleryButtonText}>📁</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.captureButton}
              onPress={takePhoto}
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
              <Text style={styles.buttonText}>Envoyer au Serveur</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      )}

      {/* Analyzing State */}
      {state === 'analyzing' && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#4A90D9" />
          <Text style={styles.analyzingText}>Envoi au serveur...</Text>
        </View>
      )}

      {/* Result View */}
      {state === 'result' && (
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Text style={styles.title}>Résultat Serveur</Text>
          
          <View style={styles.boardContainer}>
            <ChessBoard2D boardState={boardState} size={300} />
          </View>

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
    height: 160,
    backgroundColor: '#1a1a2e',
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 20,
  },
  urlContainer: {
    width: '100%',
    paddingHorizontal: 20,
    marginBottom: 15,
    flexDirection: 'row',
    alignItems: 'center',
  },
  urlLabel: {
    color: '#fff',
    marginRight: 10,
  },
  urlInput: {
    flex: 1,
    backgroundColor: '#2a2a4e',
    color: '#fff',
    padding: 8,
    borderRadius: 5,
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

  // Gallery button
  galleryButton: {
    position: 'absolute',
    left: 30,
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#2a2a4e',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#4A90D9',
  },
  galleryButtonText: {
    fontSize: 24,
  },
});
