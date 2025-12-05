import { useState, useRef, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Image,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useChessDetection, type AnalysisResult } from "@/hooks/useChessDetection";
import { drawDetectionsOnImage } from "@/src/utils/drawDetectionsOnImage";
import { PIECE_LABELS, PIECE_COLORS, PIECE_SYMBOLS, type PieceLabel } from "@/src/utils/pieceLabels";
import { ChessBoard2D } from "@/src/components/ChessBoard2D";

// Minimum average confidence to accept results without 2nd photo
const MIN_CONFIDENCE_THRESHOLD = 0.8;

// Legend data
const LEGEND_DATA: { label: PieceLabel; symbol: string; color: string }[] = PIECE_LABELS.map(label => ({
  label,
  symbol: PIECE_SYMBOLS[label],
  color: PIECE_COLORS[label],
}));

type AnalysisStep = "camera" | "photo1_taken" | "analyzing1" | "result1" | "photo2_taken" | "analyzing2" | "final";

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [step, setStep] = useState<AnalysisStep>("camera");
  
  // Photo URIs
  const [photo1Uri, setPhoto1Uri] = useState<string | null>(null);
  const [photo2Uri, setPhoto2Uri] = useState<string | null>(null);
  
  // Analysis results (contains detections, boardState, fen)
  const [result1, setResult1] = useState<AnalysisResult | null>(null);
  const [finalResult, setFinalResult] = useState<AnalysisResult | null>(null);
  
  // Annotated images
  const [annotated1Uri, setAnnotated1Uri] = useState<string | null>(null);
  const [annotatedFinalUri, setAnnotatedFinalUri] = useState<string | null>(null);
  
  const cameraRef = useRef<CameraView>(null);
  const { modelState, loadModels, runAnalysis, inputSize } = useChessDetection();

  useEffect(() => {
    loadModels();
  }, []);

  // Calculate average confidence from analysis result
  function calcAvgConfidence(result: AnalysisResult): number {
    const pieces = result.detections.filter(d => d.label);
    if (pieces.length === 0) return 0;
    return pieces.reduce((sum, d) => sum + d.confidence, 0) / pieces.length;
  }

  // STEP 1: Take first photo
  async function takePhoto1() {
    if (!cameraRef.current) return;
    
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
    if (!photo?.uri) return;
    
    setPhoto1Uri(photo.uri);
    setStep("photo1_taken");
  }

  // STEP 2: Analyze first photo (board FIRST, then pieces)
  async function analyzePhoto1() {
    if (!photo1Uri) return;
    
    setStep("analyzing1");
    
    try {
      // Run full analysis: board detection first, then pieces, then mapping
      const result = await runAnalysis(photo1Uri, { confidenceThreshold: 0.2 });
      setResult1(result);
      
      console.log("Photo1 FEN:", result.fen);
      console.log("Detected", result.detections.filter(d => d.label).length, "pieces");
      
      // Generate annotated image
      const annotated = await drawDetectionsOnImage(photo1Uri, result.detections, inputSize);
      setAnnotated1Uri(annotated + "?t=" + Date.now());
      
      setStep("result1");
    } catch (error) {
      console.error("Analyse erreur:", error);
      setStep("photo1_taken");
    }
  }

  // STEP 3: Take second photo
  async function takePhoto2() {
    if (!cameraRef.current) return;
    
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
    if (!photo?.uri) return;
    
    setPhoto2Uri(photo.uri);
    setStep("photo2_taken");
  }

  // STEP 4: Analyze second photo and improve confidence
  async function analyzeAndImprove() {
    if (!photo2Uri || !photo1Uri || !result1) return;
    
    setStep("analyzing2");
    
    try {
      // Analyze second photo
      const result2 = await runAnalysis(photo2Uri, { confidenceThreshold: 0.2 });
      
      console.log("Photo2 FEN:", result2.fen);
      
      // Compare with result1 and boost confidence where they agree
      // Photo1 is the REFERENCE - we only boost confidence, never change positions
      const improvedDetections = result1.detections.map(d => {
        if (!d.label) return d; // Keep non-piece detections as-is
        
        // Find same piece at same square in result2
        // This uses the already-computed board positions from each analysis
        const matchingPiece = result2.detections.find(d2 => 
          d2.label === d.label && 
          d2.bbox && d.bbox &&
          Math.abs((d2.bbox[0] + d2.bbox[2])/2 - (d.bbox[0] + d.bbox[2])/2) < 50 &&
          Math.abs((d2.bbox[1] + d2.bbox[3])/2 - (d.bbox[1] + d.bbox[3])/2) < 50
        );
        
        if (matchingPiece) {
          // Boost confidence!
          return {
            ...d,
            confidence: Math.min(0.99, d.confidence + matchingPiece.confidence * 0.3)
          };
        }
        return d;
      });
      
      // Create improved result (use result1's boardState as reference)
      const improved: AnalysisResult = {
        ...result1,
        detections: improvedDetections,
      };
      
      setFinalResult(improved);
      
      // Generate annotated image using original photo1 with improved detections
      const annotated = await drawDetectionsOnImage(photo1Uri, improvedDetections, inputSize);
      setAnnotatedFinalUri(annotated + "?t=" + Date.now());
      
      setStep("final");
    } catch (error) {
      console.error("Improve erreur:", error);
      setStep("result1");
    }
  }

  // Accept result1 as final
  function acceptResult1() {
    if (!result1) return;
    setFinalResult(result1);
    setAnnotatedFinalUri(annotated1Uri);
    setStep("final");
  }

  // Reset everything
  function reset() {
    setStep("camera");
    setPhoto1Uri(null);
    setPhoto2Uri(null);
    setResult1(null);
    setFinalResult(null);
    setAnnotated1Uri(null);
    setAnnotatedFinalUri(null);
  }

  // Permission handling
  if (!permission) {
    return <View style={styles.container}><ActivityIndicator size="large" color="#2563eb" /></View>;
  }
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>📷 Accès caméra requis</Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Autoriser</Text>
        </Pressable>
      </View>
    );
  }

  // FINAL RESULT VIEW
  if (step === "final" && finalResult && annotatedFinalUri) {
    const pieceCount = finalResult.detections.filter(d => d.label).length;
    const avgConf = calcAvgConfidence(finalResult);
    
    return (
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={true}
      >
        <Text style={styles.title}>✅ Analyse terminée</Text>
        <Text style={styles.subtitle}>
          {pieceCount} pièce(s) • Confiance: {(avgConf * 100).toFixed(0)}%
          {photo2Uri ? " (2 photos)" : " (1 photo)"}
        </Text>
        
        {/* 2D Chess Board */}
        {finalResult.boardState && (
          <>
            <Text style={styles.sectionTitle}>🎯 Position détectée</Text>
            <ChessBoard2D boardState={finalResult.boardState} size={320} showNotation={true} />
            
            {/* FEN */}
            <Text style={styles.fenText}>FEN: {finalResult.fen}</Text>
          </>
        )}
        
        {/* Original annotated image */}
        <Text style={[styles.sectionTitle, { marginTop: 20 }]}>📷 Image annotée</Text>
        <Image source={{ uri: annotatedFinalUri }} style={styles.resultImageSmall} resizeMode="contain" />
        
        <Text style={styles.legendTitle}>Légende</Text>
        <View style={styles.legendContainer}>
          <View style={styles.legendColumn}>
            <Text style={styles.legendColumnTitle}>Noires</Text>
            {LEGEND_DATA.filter(d => d.label.startsWith("black")).map(item => (
              <View key={item.label} style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: item.color }]} />
                <Text style={styles.legendSymbol}>{item.symbol}</Text>
                <Text style={styles.legendLabel}>{item.label.split("-")[1]}</Text>
              </View>
            ))}
          </View>
          <View style={styles.legendColumn}>
            <Text style={styles.legendColumnTitle}>Blanches</Text>
            {LEGEND_DATA.filter(d => d.label.startsWith("white")).map(item => (
              <View key={item.label} style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: item.color }]} />
                <Text style={styles.legendSymbol}>{item.symbol}</Text>
                <Text style={styles.legendLabel}>{item.label.split("-")[1]}</Text>
              </View>
            ))}
          </View>
        </View>
        
        <Pressable style={styles.button} onPress={reset}>
          <Text style={styles.buttonText}>Nouvelle analyse</Text>
        </Pressable>
      </ScrollView>
    );
  }

  // RESULT 1 VIEW
  if (step === "result1" && result1 && annotated1Uri) {
    const avgConf = calcAvgConfidence(result1);
    const isLowConfidence = avgConf < MIN_CONFIDENCE_THRESHOLD;
    const pieceCount = result1.detections.filter(d => d.label).length;
    
    return (
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={true}
      >
        <Text style={styles.title}>📊 Première analyse</Text>
        <Text style={styles.subtitle}>
          {pieceCount} pièce(s) • Confiance: {(avgConf * 100).toFixed(0)}%
        </Text>
        
        {/* 2D Board Preview */}
        {result1.boardState && (
          <ChessBoard2D boardState={result1.boardState} size={280} showNotation={true} />
        )}
        
        <Image source={{ uri: annotated1Uri }} style={styles.resultImageSmall} resizeMode="contain" />
        
        {isLowConfidence ? (
          <View style={styles.warningBox}>
            <Text style={styles.warningTitle}>⚠️ Confiance faible</Text>
            <Text style={styles.warningText}>
              Prenez une 2ème photo pour améliorer la précision
            </Text>
          </View>
        ) : (
          <Text style={styles.successText}>✓ Bonne confiance</Text>
        )}
        
        {/* Legend */}
        <Text style={styles.legendTitle}>Légende</Text>
        <View style={styles.legendContainer}>
          <View style={styles.legendColumn}>
            <Text style={styles.legendColumnTitle}>Noires</Text>
            {LEGEND_DATA.filter(d => d.label.startsWith("black")).map(item => (
              <View key={item.label} style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: item.color }]} />
                <Text style={styles.legendSymbol}>{item.symbol}</Text>
                <Text style={styles.legendLabel}>{item.label.split("-")[1]}</Text>
              </View>
            ))}
          </View>
          <View style={styles.legendColumn}>
            <Text style={styles.legendColumnTitle}>Blanches</Text>
            {LEGEND_DATA.filter(d => d.label.startsWith("white")).map(item => (
              <View key={item.label} style={styles.legendItem}>
                <View style={[styles.legendColor, { backgroundColor: item.color }]} />
                <Text style={styles.legendSymbol}>{item.symbol}</Text>
                <Text style={styles.legendLabel}>{item.label.split("-")[1]}</Text>
              </View>
            ))}
          </View>
        </View>
        
        <View style={styles.buttonRow}>
          {isLowConfidence && (
            <Pressable style={styles.button} onPress={() => setStep("camera")}>
              <Text style={styles.buttonText}>📸 2ème photo</Text>
            </Pressable>
          )}
          <Pressable style={[styles.button, styles.buttonSecondary]} onPress={acceptResult1}>
            <Text style={styles.buttonText}>Accepter</Text>
          </Pressable>
        </View>
        
        <Pressable style={styles.linkButton} onPress={reset}>
          <Text style={styles.linkButtonText}>Recommencer</Text>
        </Pressable>
      </ScrollView>
    );
  }

  // PHOTO 2 TAKEN
  if (step === "photo2_taken" && photo1Uri && photo2Uri) {
    return (
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={true}
      >
        <Text style={styles.title}>📷 2 photos prises</Text>
        <Text style={styles.subtitle}>Prêt à améliorer l'analyse</Text>
        
        <View style={styles.photoRow}>
          <View style={styles.photoBox}>
            <Text style={styles.photoLabel}>Photo 1 (référence)</Text>
            <Image source={{ uri: photo1Uri }} style={styles.smallPhoto} />
          </View>
          <View style={styles.photoBox}>
            <Text style={styles.photoLabel}>Photo 2</Text>
            <Image source={{ uri: photo2Uri }} style={styles.smallPhoto} />
          </View>
        </View>
        
        <Pressable style={styles.button} onPress={analyzeAndImprove}>
          <Text style={styles.buttonText}>🔄 Analyser et améliorer</Text>
        </Pressable>
      </ScrollView>
    );
  }

  // PHOTO 1 TAKEN
  if (step === "photo1_taken" && photo1Uri) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>📷 Photo prise</Text>
        
        <Image source={{ uri: photo1Uri }} style={styles.resultImage} resizeMode="contain" />
        
        <Pressable style={styles.button} onPress={analyzePhoto1}>
          <Text style={styles.buttonText}>🔍 Analyser</Text>
        </Pressable>
        
        <Pressable style={styles.linkButton} onPress={reset}>
          <Text style={styles.linkButtonText}>Reprendre</Text>
        </Pressable>
      </View>
    );
  }

  // ANALYZING states
  if (step === "analyzing1" || step === "analyzing2") {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.subtitle}>
          {step === "analyzing1" ? "Détection board + pièces..." : "Amélioration de l'analyse..."}
        </Text>
      </View>
    );
  }

  // CAMERA VIEW
  const isSecondPhoto = photo1Uri !== null && step === "camera";
  
  return (
    <View style={styles.container}>
      <Text style={styles.title}>
        {isSecondPhoto ? "📷 2ème photo" : "📷 Analyse Live"}
      </Text>
      
      {modelState === "loading" && (
        <View style={styles.statusRow}>
          <ActivityIndicator size="small" color="#2563eb" />
          <Text style={styles.statusText}>Chargement modèles…</Text>
        </View>
      )}
      {modelState === "ready" && (
        <Text style={[styles.statusText, { color: "#16a34a" }]}>✓ Prêt</Text>
      )}
      
      <View style={styles.cameraContainer}>
        <CameraView ref={cameraRef} style={styles.camera} facing="back" />
      </View>
      
      {modelState === "ready" && (
        <Pressable style={styles.captureButton} onPress={isSecondPhoto ? takePhoto2 : takePhoto1}>
          <Text style={styles.captureButtonText}>📸</Text>
        </Pressable>
      )}
      
      <Text style={styles.hint}>
        {isSecondPhoto 
          ? "Prenez la photo sous un autre angle" 
          : "Positionnez l'échiquier et prenez une photo"
        }
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: "#1e293b", 
    alignItems: "center", 
    paddingTop: 60, 
    paddingBottom: 40 
  },
  scrollView: {
    flex: 1,
    backgroundColor: "#1e293b",
  },
  scrollContent: {
    alignItems: "center",
    paddingTop: 60,
    paddingBottom: 100, // Extra padding pour le tab bar
    paddingHorizontal: 16,
  },
  title: { fontSize: 24, fontWeight: "800", color: "#fff", marginBottom: 4 },
  subtitle: { fontSize: 14, color: "#94a3b8", marginBottom: 16, textAlign: "center" },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 12 },
  statusText: { fontSize: 14, color: "#94a3b8", marginBottom: 12 },
  cameraContainer: { width: "90%", aspectRatio: 1, borderRadius: 16, overflow: "hidden", backgroundColor: "#000" },
  camera: { flex: 1 },
  captureButton: { marginTop: 24, width: 80, height: 80, borderRadius: 40, backgroundColor: "#2563eb", justifyContent: "center", alignItems: "center", borderWidth: 4, borderColor: "#fff" },
  captureButtonText: { fontSize: 32 },
  hint: { marginTop: 16, fontSize: 14, color: "#64748b", textAlign: "center", paddingHorizontal: 32 },
  button: { marginTop: 16, backgroundColor: "#2563eb", paddingVertical: 14, paddingHorizontal: 32, borderRadius: 999 },
  buttonSecondary: { backgroundColor: "#16a34a" },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  buttonRow: { flexDirection: "row", gap: 12, marginTop: 16 },
  linkButton: { marginTop: 16, padding: 10, marginBottom: 20 },
  linkButtonText: { color: "#64748b", fontSize: 14 },
  resultImage: { width: "90%", aspectRatio: 1, borderRadius: 16, backgroundColor: "#fff" },
  resultImageSmall: { width: "70%", aspectRatio: 1, borderRadius: 12, backgroundColor: "#fff", marginTop: 12 },
  sectionTitle: { fontSize: 16, fontWeight: "600", color: "#fff", marginTop: 16, marginBottom: 8 },
  fenText: { fontSize: 12, color: "#64748b", marginTop: 8, fontFamily: "monospace" },
  warningBox: { backgroundColor: "#fef3c7", padding: 16, borderRadius: 12, marginTop: 16, width: "90%", alignItems: "center" },
  warningTitle: { fontSize: 18, fontWeight: "700", color: "#92400e", marginBottom: 4 },
  warningText: { fontSize: 14, color: "#78350f", textAlign: "center" },
  successText: { fontSize: 16, color: "#16a34a", marginTop: 16, fontWeight: "600" },
  photoRow: { flexDirection: "row", gap: 12, marginVertical: 16 },
  photoBox: { alignItems: "center" },
  photoLabel: { color: "#94a3b8", fontSize: 12, marginBottom: 4 },
  smallPhoto: { width: 150, height: 150, borderRadius: 8 },
  legendTitle: { fontSize: 16, fontWeight: "600", color: "#fff", marginTop: 20, marginBottom: 12 },
  legendContainer: { flexDirection: "row", gap: 24, backgroundColor: "#334155", padding: 16, borderRadius: 12, marginBottom: 16 },
  legendColumn: { gap: 8 },
  legendColumnTitle: { fontSize: 14, fontWeight: "600", color: "#94a3b8", marginBottom: 4 },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 8 },
  legendColor: { width: 16, height: 16, borderRadius: 4 },
  legendSymbol: { fontSize: 18, width: 24, textAlign: "center", color: "#fff" },
  legendLabel: { fontSize: 13, color: "#e2e8f0", textTransform: "capitalize" },
});