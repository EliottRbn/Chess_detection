import {
  View,
  Text,
  Image,
  StyleSheet,
  Pressable,
  ScrollView,
} from "react-native";
import { useState } from "react";
import { Asset } from "expo-asset";
import * as ort from "onnxruntime-react-native";

import { loadOnnxModel } from "../src/utils/loadOnnxModel";
import { preprocessImage } from "../src/utils/preprocessImage";
import { postprocessPieces } from "../src/utils/postprocessPieces";
import { postprocessBoard } from "../src/utils/postprocessBoard";
import {
  drawDetectionsOnImage,
  type Detection,
} from "../src/utils/drawDetectionsOnImage";

/* ============================================================
   UTILITAIRE — LOGS LÉGERS POUR HERMES
   ============================================================ */
function debugOutputs(title: string, outputs: Record<string, any>) {
  console.log(`\n===== ${title} =====`);

  for (const key of Object.keys(outputs)) {
    const t = outputs[key];

    if (!t || !t.data) {
      console.log(`• ${key}: tensor vide ou invalide`);
      continue;
    }

    const dims = JSON.stringify(t.dims);
    const dtype = t.type;

    // 5 premières valeurs
    const first = Array.from(t.data.slice(0, 5));

    // min/max (échantillonnage pour éviter surcharge Hermes)
    let min = Infinity;
    let max = -Infinity;
    const step = Math.floor(t.data.length / 5000) + 1;

    for (let i = 0; i < t.data.length; i += step) {
      const v = t.data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }

    console.log(
      `• ${key}: dims=${dims}, dtype=${dtype}\n` +
      `  first5=${JSON.stringify(first)}\n` +
      `  approx min=${min.toFixed(4)}, max=${max.toFixed(4)}`
    );
  }
}

/* ============================================================
   COMPONENT
   ============================================================ */
export default function HomeScreen() {
  const [logText, setLogText] = useState("");
  const [outputUri, setOutputUri] = useState<string | null>(null);

  async function handleAnalyze() {
    try {
      setLogText("Chargement modèles ONNX…");
      setOutputUri(null);

      const piecesSession = await loadOnnxModel(
        require("@/assets/models/yolo8-20epoch.onnx")
      );

      const boardSession = await loadOnnxModel(
        require("@/assets/models/chessboard_detection.onnx")
      );

      setLogText((t) => t + "\nModèles chargés\nPréprocess…");

      const asset = Asset.fromModule(require("../assets/test.jpg"));
      await asset.downloadAsync();

      const uri = asset.localUri ?? asset.uri;
      const inputSize = 640;
      const inputTensor = await preprocessImage(uri, inputSize);

      const tensor = new ort.Tensor("float32", inputTensor, [
        1,
        3,
        inputSize,
        inputSize,
      ]);

      setLogText((t) => t + "\nInference…");

      /* =============================
         1) PREDICTIONS ONNX
         ============================= */
      const outputsPieces = await piecesSession.run({ images: tensor });
      debugOutputs("PIECES OUTPUTS", outputsPieces);

      const outputsBoard = await boardSession.run({ images: tensor });
      debugOutputs("BOARD OUTPUTS", outputsBoard);

      setLogText((t) => t + "\nPostprocess…");

      /* =============================
         2) POSTPROCESS
         ============================= */
      const pieceDetections = postprocessPieces(outputsPieces, 0.25);
      const boardDetection = postprocessBoard(outputsBoard);

      const detections: Detection[] = [
        ...pieceDetections,
        ...(boardDetection ? [boardDetection] : []),
      ];

      setLogText((t) => t + `\nDétections totales: ${detections.length}`);
        console.log("=== RAW DETECTIONS BEFORE DRAW ===");
        console.log(JSON.stringify(detections, null, 2));
      /* =============================
         3) DRAW
         ============================= */
      const annotated = await drawDetectionsOnImage(uri, detections, inputSize);
      setOutputUri(annotated);

      setLogText((t) => t + "\nImage annotée générée.");
    } catch (e: any) {
      setLogText((t) => t + "\nERREUR: " + String(e?.message ?? e));
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Chess Vision — ONNX</Text>

      <Image
        source={require("../assets/test.jpg")}
        style={styles.image}
        resizeMode="contain"
      />

      <Pressable style={styles.button} onPress={handleAnalyze}>
        <Text style={styles.buttonText}>Analyser l'image</Text>
      </Pressable>

      {outputUri && (
        <Image
          source={{ uri: outputUri }}
          style={[styles.image, { marginTop: 20 }]}
          resizeMode="contain"
        />
      )}

      <ScrollView style={styles.logBox}>
        <Text style={styles.logText}>{logText}</Text>
      </ScrollView>
    </ScrollView>
  );
}

/* ============================================================
   STYLES
   ============================================================ */
const styles = StyleSheet.create({
  container: {
    paddingTop: 80,
    paddingBottom: 40,
    alignItems: "center",
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 20,
  },
  image: {
    width: "90%",
    height: 320,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  button: {
    marginTop: 20,
    backgroundColor: "#2563eb",
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 999,
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
  logBox: {
    marginTop: 20,
    width: "90%",
    maxHeight: 220,
    borderWidth: 1,
    borderColor: "#eee",
    borderRadius: 8,
    padding: 8,
  },
  logText: {
    fontFamily: "Courier",
    fontSize: 12,
  },
});