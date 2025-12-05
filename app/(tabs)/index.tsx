import {
  View,
  Text,
  Image,
  StyleSheet,
  Pressable,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import { useState, useEffect } from "react";
import { Asset } from "expo-asset";
import { useChessDetection } from "@/hooks/useChessDetection";
import { drawDetectionsOnImage } from "@/src/utils/drawDetectionsOnImage";
import { PIECE_LABELS, PIECE_COLORS, PIECE_SYMBOLS, type PieceLabel } from "@/src/utils/pieceLabels";

// Color legend data
const LEGEND_DATA: { label: PieceLabel; symbol: string; color: string }[] = PIECE_LABELS.map(label => ({
  label,
  symbol: PIECE_SYMBOLS[label],
  color: PIECE_COLORS[label],
}));

export default function AnalyseScreen() {
  const [outputUri, setOutputUri] = useState<string | null>(null);
  const [detectionCount, setDetectionCount] = useState(0);
  const { modelState, isProcessing, error, loadModels, runInference, inputSize } = useChessDetection();

  useEffect(() => {
    loadModels();
  }, []);

  async function handleAnalyze() {
    try {
      setOutputUri(null);
      setDetectionCount(0);

      const asset = Asset.fromModule(require("@/assets/test.jpg"));
      await asset.downloadAsync();
      const uri = asset.localUri ?? asset.uri;

      const detections = await runInference(uri);
      setDetectionCount(detections.filter(d => d.label).length);

      const annotated = await drawDetectionsOnImage(uri, detections, inputSize);
      setOutputUri(annotated + "?t=" + Date.now());
    } catch (e: any) {
      console.error("Analyse error:", e);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>♟️ Chess Vision</Text>
      <Text style={styles.subtitle}>Analyse d&apos;image statique</Text>

      <Image
        source={require("@/assets/test.jpg")}
        style={styles.image}
        resizeMode="contain"
      />

      <View style={styles.statusRow}>
        {modelState === "loading" && (
          <>
            <ActivityIndicator size="small" color="#2563eb" />
            <Text style={styles.statusText}>Chargement des modèles…</Text>
          </>
        )}
        {modelState === "ready" && (
          <Text style={[styles.statusText, { color: "#16a34a" }]}>✓ Modèles prêts</Text>
        )}
        {modelState === "error" && (
          <Text style={[styles.statusText, { color: "#dc2626" }]}>❌ {error}</Text>
        )}
      </View>

      <Pressable
        style={[styles.button, (isProcessing || modelState !== "ready") && styles.buttonDisabled]}
        onPress={handleAnalyze}
        disabled={isProcessing || modelState !== "ready"}
      >
        {isProcessing ? (
          <ActivityIndicator size="small" color="white" />
        ) : (
          <Text style={styles.buttonText}>Analyser l&apos;image</Text>
        )}
      </Pressable>

      {outputUri && (
        <>
          <Text style={styles.sectionTitle}>Résultat ({detectionCount} pièces)</Text>
          <Image
            source={{ uri: outputUri }}
            style={styles.image}
            resizeMode="contain"
          />

          {/* Color Legend */}
          <Text style={styles.legendTitle}>Légende des couleurs</Text>
          <View style={styles.legendContainer}>
            {/* Black pieces */}
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
            
            {/* White pieces */}
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
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingTop: 60,
    paddingBottom: 40,
    alignItems: "center",
    backgroundColor: "#f8fafc",
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: "#1e293b",
  },
  subtitle: {
    fontSize: 14,
    color: "#64748b",
    marginBottom: 20,
  },
  image: {
    width: "90%",
    height: 280,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#e2e8f0",
    backgroundColor: "#fff",
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 16,
    gap: 8,
    minHeight: 24,
  },
  statusText: {
    fontSize: 14,
    color: "#64748b",
  },
  button: {
    marginTop: 16,
    backgroundColor: "#2563eb",
    paddingVertical: 14,
    paddingHorizontal: 40,
    borderRadius: 999,
    minWidth: 180,
    alignItems: "center",
  },
  buttonDisabled: {
    backgroundColor: "#94a3b8",
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#1e293b",
    marginTop: 24,
    marginBottom: 12,
  },
  legendTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1e293b",
    marginTop: 20,
    marginBottom: 12,
  },
  legendContainer: {
    flexDirection: "row",
    gap: 24,
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  legendColumn: {
    gap: 8,
  },
  legendColumnTitle: {
    fontSize: 14,
    fontWeight: "600",
    color: "#64748b",
    marginBottom: 4,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  legendColor: {
    width: 16,
    height: 16,
    borderRadius: 4,
  },
  legendSymbol: {
    fontSize: 18,
    width: 24,
    textAlign: "center",
  },
  legendLabel: {
    fontSize: 13,
    color: "#334155",
    textTransform: "capitalize",
  },
});