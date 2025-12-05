import * as ImageManipulator from 'expo-image-manipulator';
import jpeg from 'jpeg-js';

/**
 * Preprocess an image for ONNX model input
 * - Resize to target size (default 640)
 * - Convert to RGB float32 tensor [1, 3, size, size]
 * - Normalize to 0-1
 */
export async function preprocessImage(
  uri: string, 
  targetSize: number = 640
): Promise<{
  tensor: Float32Array;
  originalWidth: number;
  originalHeight: number;
}> {
  console.log(`[Preprocess] Starting with URI for size ${targetSize}...`);
  
  // First, get original dimensions
  const originalInfo = await ImageManipulator.manipulateAsync(uri, []);
  const originalWidth = originalInfo.width;
  const originalHeight = originalInfo.height;

  // Resize to target size and convert to JPEG
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: targetSize, height: targetSize } }],
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );

  if (!resized.base64) {
    throw new Error('Failed to get base64 image');
  }

  // Decode JPEG to raw pixels
  const jpegData = Uint8Array.from(atob(resized.base64), c => c.charCodeAt(0));
  const rawImage = jpeg.decode(jpegData, { useTArray: true });
  const pixels = rawImage.data; // RGBA format

  // Convert to RGB float32 tensor [1, 3, H, W] normalized to 0-1
  const tensor = new Float32Array(1 * 3 * targetSize * targetSize);
  
  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const pixelIndex = (y * targetSize + x) * 4; // RGBA
      const tensorIndex = y * targetSize + x;
      
      // RGB channels (normalized 0-1)
      tensor[0 * targetSize * targetSize + tensorIndex] = pixels[pixelIndex] / 255;     // R
      tensor[1 * targetSize * targetSize + tensorIndex] = pixels[pixelIndex + 1] / 255; // G
      tensor[2 * targetSize * targetSize + tensorIndex] = pixels[pixelIndex + 2] / 255; // B
    }
  }

  return { tensor, originalWidth, originalHeight };
}

export async function preprocessBoardImage(uri: string): Promise<Float32Array> {
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: 256, height: 256 } }],
    { format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );

  const jpegData = Uint8Array.from(atob(resized.base64!), c => c.charCodeAt(0));
  const raw = jpeg.decode(jpegData, { useTArray: true });
  const pixels = raw.data;

  // Convert to CHW format (Planar: RRR...GGG...BBB...)
  const tensor = new Float32Array(3 * 256 * 256);
  const size = 256;
  
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const pixelIndex = (y * size + x) * 4; // RGBA index
      const tensorIndex = y * size + x;      // Spatial index
      
      // R channel
      tensor[tensorIndex] = pixels[pixelIndex] / 255.0;
      // G channel
      tensor[size * size + tensorIndex] = pixels[pixelIndex + 1] / 255.0;
      // B channel
      tensor[2 * size * size + tensorIndex] = pixels[pixelIndex + 2] / 255.0;
    }
  }
  return tensor;
}