import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import * as FileSystem from "expo-file-system/legacy";
import jpeg from "jpeg-js";

/** Convertit base64 → Uint8Array (sans Buffer, compatible RN) */
function base64ToUint8Array(base64: string): Uint8Array {
  const binary = global.atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);

  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * Charge une image (URI local) et la convertit en tenseur Float32Array
 * shape [1, 3, inputSize, inputSize] normalisé [0,1].
 */
export async function preprocessImage(
  uri: string,
  inputSize = 640
): Promise<Float32Array> {
  // 1) Resize l’image
  const resized = await manipulateAsync(
    uri,
    [{ resize: { width: inputSize, height: inputSize } }],
    { format: SaveFormat.JPEG }
  );

  // 2) Lire en base64
  const base64 = await FileSystem.readAsStringAsync(resized.uri, {
    encoding: "base64",
  });

  // 3) Convertir base64 → Uint8Array
  const byteArray = base64ToUint8Array(base64);

  // 4) JPEG decode
  const decoded = jpeg.decode(byteArray, { useTArray: true });
  if (!decoded || !decoded.data) {
    throw new Error("Erreur de décodage JPEG");
  }

  const { width, height, data } = decoded;

  // 5) CHW float32
  const size = inputSize * inputSize;
  const tensor = new Float32Array(3 * size);

  for (let i = 0; i < size; i++) {
    tensor[i] = data[i * 4] / 255;         // R
    tensor[i + size] = data[i * 4 + 1] / 255; // G
    tensor[i + 2 * size] = data[i * 4 + 2] / 255; // B
  }

  return tensor;
}