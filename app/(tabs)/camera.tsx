import { useState, useRef, useEffect, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Dimensions,
  ActivityIndicator,
  Platform,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { Skia, Canvas, Rect, Path } from "@shopify/react-native-skia";
import { useChessDetection, type Detection } from "@/hooks/useChessDetection";
import * as FileSystem from "expo-file-system/legacy";

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CAMERA_SIZE = SCREEN_WIDTH;
const INPUT_SIZE = 640;

// Throttle interval for inference (ms)
const INFERENCE_INTERVAL = 200; // ~5 FPS

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isLive, setIsLive] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  
  const cameraRef = useRef<CameraView>(null);
  const lastInferenceTime = useRef(0);
  const frameCount = useRef(0);
  const fpsTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  
  const { modelState, isProcessing, loadModels, runInference } = useChessDetection();

  // Preload models on mount
  useEffect(() => {
    loadModels();
    
    // FPS counter
    fpsTimer.current = setInterval(() => {
      setFps(frameCount.current);
      frameCount.current = 0;
    }, 1000);
    
    return () => {
      if (fpsTimer.current) clearInterval(fpsTimer.current);
    };
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (!cameraRef.current || !isLive || isProcessing) return;
    
    const now = Date.now();
    if (now - lastInferenceTime.current < INFERENCE_INTERVAL) return;
    lastInferenceTime.current = now;
    
    try {
      // Capture a frame
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5,
        skipProcessing: true,
      });
      
      if (!photo?.uri) return;
      
      // Run inference
      const results = await runInference(photo.uri, {
        detectBoard: true,
        confidenceThreshold: 0.3,
      });
      
      setDetections(results);
      frameCount.current++;
      
      // Clean up temp file
      await FileSystem.deleteAsync(photo.uri, { idempotent: true });
    } catch (e) {
      console.warn("Frame capture error:", e);
    }
  }, [isLive, isProcessing, runInference]);

  // Continuous capture loop
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (isLive && modelState === "ready") {
      intervalId = setInterval(captureAndAnalyze, 100);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isLive, modelState, captureAndAnalyze]);

  // Scale factor from input size to camera view
  const scale = CAMERA_SIZE / INPUT_SIZE;

  if (!permission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#2563eb" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionTitle}>📷 Accès Caméra</Text>
        <Text style={styles.permissionText}>
          L&apos;application a besoin d&apos;accéder à la caméra pour détecter les pièces d&apos;échecs en temps réel.
        </Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Autoriser la caméra</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>♟️ Détection Live</Text>
        <View style={styles.statsRow}>
          <View style={[styles.badge, modelState === "ready" ? styles.badgeSuccess : styles.badgeWarning]}>
            <Text style={styles.badgeText}>
              {modelState === "ready" ? "Modèle OK" : "Chargement..."}
            </Text>
          </View>
          {isLive && (
            <View style={styles.badge}>
              <Text style={styles.badgeText}>{fps} FPS</Text>
            </View>
          )}
          <View style={styles.badge}>
            <Text style={styles.badgeText}>{detections.filter(d => d.label).length} pièces</Text>
          </View>
        </View>
      </View>

      {/* Camera Preview */}
      <View style={styles.cameraContainer}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing="back"
        />
        
        {/* Detection Overlay */}
        <Canvas style={styles.overlay}>
          {detections.map((det, i) => {
            if (det.polygon) {
              // Draw board polygon
              const path = Skia.Path.Make();
              const pts = det.polygon;
              path.moveTo(pts[0].x * scale, pts[0].y * scale);
              for (let j = 1; j < pts.length; j++) {
                path.lineTo(pts[j].x * scale, pts[j].y * scale);
              }
              path.close();
              
              return (
                <Path
                  key={`board-${i}`}
                  path={path}
                  color="#00FFFF"
                  style="stroke"
                  strokeWidth={3}
                />
              );
            }
            
            if (det.bbox) {
              // Draw piece bounding box
              const [x1, y1, x2, y2] = det.bbox;
              return (
                <Rect
                  key={`piece-${i}`}
                  x={x1 * scale}
                  y={y1 * scale}
                  width={(x2 - x1) * scale}
                  height={(y2 - y1) * scale}
                  color={det.color ?? "#FF0000"}
                  style="stroke"
                  strokeWidth={2}
                />
              );
            }
            return null;
          })}
        </Canvas>
      </View>

      {/* Detection List */}
      <View style={styles.detectionList}>
        {detections.filter(d => d.label).slice(0, 6).map((det, i) => (
          <View key={i} style={[styles.detectionChip, { borderColor: det.color }]}>
            <Text style={styles.chipSymbol}>{det.symbol}</Text>
            <Text style={styles.chipConfidence}>{Math.round(det.confidence * 100)}%</Text>
          </View>
        ))}
        {detections.filter(d => d.label).length > 6 && (
          <View style={styles.detectionChip}>
            <Text style={styles.chipSymbol}>+{detections.filter(d => d.label).length - 6}</Text>
          </View>
        )}
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable
          style={[
            styles.liveButton,
            isLive && styles.liveButtonActive,
            modelState !== "ready" && styles.buttonDisabled,
          ]}
          onPress={() => setIsLive(!isLive)}
          disabled={modelState !== "ready"}
        >
          <View style={[styles.liveIndicator, isLive && styles.liveIndicatorActive]} />
          <Text style={[styles.liveButtonText, isLive && styles.liveButtonTextActive]}>
            {isLive ? "STOP" : "START"}
          </Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0f172a",
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: "#0f172a",
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
  permissionTitle: {
    fontSize: 28,
    fontWeight: "800",
    color: "#f8fafc",
    marginBottom: 16,
  },
  permissionText: {
    fontSize: 16,
    color: "#94a3b8",
    textAlign: "center",
    marginBottom: 32,
    lineHeight: 24,
  },
  header: {
    paddingTop: Platform.OS === "ios" ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: "#f8fafc",
    textAlign: "center",
  },
  statsRow: {
    flexDirection: "row",
    justifyContent: "center",
    gap: 8,
    marginTop: 12,
  },
  badge: {
    backgroundColor: "#1e293b",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  badgeSuccess: {
    backgroundColor: "#166534",
  },
  badgeWarning: {
    backgroundColor: "#854d0e",
  },
  badgeText: {
    color: "#f8fafc",
    fontSize: 12,
    fontWeight: "600",
  },
  cameraContainer: {
    width: CAMERA_SIZE,
    height: CAMERA_SIZE,
    alignSelf: "center",
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "#000",
  },
  camera: {
    width: "100%",
    height: "100%",
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
  },
  detectionList: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: 8,
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  detectionChip: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#1e293b",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "#334155",
    gap: 6,
  },
  chipSymbol: {
    fontSize: 18,
    color: "#f8fafc",
  },
  chipConfidence: {
    fontSize: 12,
    color: "#94a3b8",
    fontWeight: "600",
  },
  controls: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  liveButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#1e293b",
    paddingVertical: 16,
    paddingHorizontal: 40,
    borderRadius: 999,
    gap: 12,
  },
  liveButtonActive: {
    backgroundColor: "#dc2626",
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  liveIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#64748b",
  },
  liveIndicatorActive: {
    backgroundColor: "#fff",
  },
  liveButtonText: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "700",
  },
  liveButtonTextActive: {
    color: "#fff",
  },
  button: {
    backgroundColor: "#2563eb",
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 999,
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
});
