import * as ort from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system/legacy";

export async function loadOnnxModel(moduleRef: any) {
  console.log("=== loadOnnxModel START ===");

  const asset = Asset.fromModule(moduleRef);
  await asset.downloadAsync();

  const localUri = asset.localUri ?? asset.uri;
  if (!localUri) {
    throw new Error("URI locale introuvable pour le modèle ONNX");
  }

  const filename = localUri.split("/").pop();
  if (!filename) {
    throw new Error("Impossible d'extraire le nom de fichier du modèle");
  }

  const cacheDir = FileSystem.cacheDirectory;
  if (!cacheDir) {
    throw new Error("FileSystem.cacheDirectory est null");
  }

  const destPath = cacheDir + filename;
  console.log("destPath:", destPath);

  const exists = await FileSystem.getInfoAsync(destPath);
  if (!exists.exists) {
    console.log("Copying model to cache…");
    await FileSystem.copyAsync({ from: localUri, to: destPath });
  }

  console.log("Creating ONNX session…");
  const session = await ort.InferenceSession.create(destPath);

  console.log("=== loadOnnxModel END ===");
  return session;
}