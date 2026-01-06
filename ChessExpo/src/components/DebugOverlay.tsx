import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { Canvas, Rect, Line, Circle, vec, Points } from '@shopify/react-native-skia';
import { PieceDetection, MODEL_INPUT_SIZE } from '../types';
import { BoardCorners } from '../utils/yoloParser';

// ... (colors remain unchanged)
const PIECE_COLORS: Record<number, string> = {
  0: '#8B4513', 1: '#000000', 2: '#4B0082', 3: '#696969', 4: '#800080', 5: '#2F4F4F',
  6: '#FFD700', 7: '#FF0000', 8: '#00CED1', 9: '#98FB98', 10: '#FF69B4', 11: '#1E90FF',
};

const PIECE_NAMES: Record<number, string> = {
  0: 'B-Bishop', 1: 'B-King', 2: 'B-Knight', 3: 'B-Pawn', 4: 'B-Queen', 5: 'B-Rook',
  6: 'W-Bishop', 7: 'W-King', 8: 'W-Knight', 9: 'W-Pawn', 10: 'W-Queen', 11: 'W-Rook',
};

interface DebugOverlayProps {
  photoUri: string;
  pieces: PieceDetection[];
  boardCorners: BoardCorners | null;
  displaySize: number;
}

export function DebugOverlay({ photoUri, pieces, boardCorners, displaySize }: DebugOverlayProps) {
  // Scale factor from model coordinates (640) to display size
  const scale = displaySize / MODEL_INPUT_SIZE;

  return (
    <View style={styles.container}>
      {/* Original image */}
      <View style={[styles.imageContainer, { width: displaySize, height: displaySize }]}>
        <Image
          source={{ uri: photoUri }}
          style={{ width: displaySize, height: displaySize }}
          resizeMode="stretch"
        />
        
        {/* Canvas overlay for drawing boxes */}
        <Canvas style={[styles.canvas, { width: displaySize, height: displaySize }]}>
          {/* Draw board quadrilateral from corners */}
          {boardCorners && (
            <>
              {/* Board outline - 4 lines connecting corners (TL -> TR -> BR -> BL -> TL) */}
              <Line
                p1={vec(boardCorners.topLeft.x * scale, boardCorners.topLeft.y * scale)}
                p2={vec(boardCorners.topRight.x * scale, boardCorners.topRight.y * scale)}
                color="cyan"
                strokeWidth={3}
              />
              <Line
                p1={vec(boardCorners.topRight.x * scale, boardCorners.topRight.y * scale)}
                p2={vec(boardCorners.bottomRight.x * scale, boardCorners.bottomRight.y * scale)}
                color="cyan"
                strokeWidth={3}
              />
              <Line
                p1={vec(boardCorners.bottomRight.x * scale, boardCorners.bottomRight.y * scale)}
                p2={vec(boardCorners.bottomLeft.x * scale, boardCorners.bottomLeft.y * scale)}
                color="cyan"
                strokeWidth={3}
              />
              <Line
                p1={vec(boardCorners.bottomLeft.x * scale, boardCorners.bottomLeft.y * scale)}
                p2={vec(boardCorners.topLeft.x * scale, boardCorners.topLeft.y * scale)}
                color="cyan"
                strokeWidth={3}
              />
              
              {/* Corner circles with specific colors */}
              {/* TL = Green */}
              <Circle cx={boardCorners.topLeft.x * scale} cy={boardCorners.topLeft.y * scale} r={6} color="#00FF00" />
              {/* TR = Yellow */}
              <Circle cx={boardCorners.topRight.x * scale} cy={boardCorners.topRight.y * scale} r={6} color="#FFFF00" />
              {/* BR = Red */}
              <Circle cx={boardCorners.bottomRight.x * scale} cy={boardCorners.bottomRight.y * scale} r={6} color="#FF0000" />
              {/* BL = Orange */}
              <Circle cx={boardCorners.bottomLeft.x * scale} cy={boardCorners.bottomLeft.y * scale} r={6} color="#FFA500" />
            </>
          )}
          
          {/* Draw piece bounding boxes */}
          {pieces.map((piece, index) => (
            <Rect
              key={index}
              x={(piece.box.x - piece.box.width / 2) * scale}
              y={(piece.box.y - piece.box.height / 2) * scale}
              width={piece.box.width * scale}
              height={piece.box.height * scale}
              color={PIECE_COLORS[piece.classId] ?? '#FFFFFF'}
              style="stroke"
              strokeWidth={2}
            />
          ))}
        </Canvas>
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        <Text style={styles.legendTitle}>Détections: {pieces.length} pièces</Text>
        
        {boardCorners ? (
          <View style={styles.cornersInfo}>
            <Text style={styles.cornerLabel}><Text style={{ color: '#00FF00' }}>●</Text> TL (Haut-Gauche)</Text>
            <Text style={styles.cornerLabel}><Text style={{ color: '#FFFF00' }}>●</Text> TR (Haut-Droit)</Text>
            <Text style={styles.cornerLabel}><Text style={{ color: '#FFA500' }}>●</Text> BL (Bas-Gauche)</Text>
            <Text style={styles.cornerLabel}><Text style={{ color: '#FF0000' }}>●</Text> BR (Bas-Droit)</Text>
          </View>
        ) : (
          <Text style={styles.noBoard}>Board non détecté</Text>
        )}

        <View style={styles.colorGrid}>
          {Object.entries(PIECE_COLORS).map(([classId, color]) => {
            const count = pieces.filter(p => p.classId === parseInt(classId)).length;
            if (count === 0) return null;
            return (
              <View key={classId} style={styles.colorItem}>
                <View style={[styles.colorBox, { backgroundColor: color }]} />
                <Text style={styles.colorLabel}>
                  {PIECE_NAMES[parseInt(classId)]}: {count}
                </Text>
              </View>
            );
          })}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  imageContainer: {
    position: 'relative',
    borderWidth: 1,
    borderColor: '#333',
  },
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  legend: {
    marginTop: 15,
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 8,
    width: '100%',
  },
  legendTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  cornersInfo: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#555',
  },
  cornerLabel: {
    color: '#fff',
    fontSize: 12,
  },
  noBoard: {
    color: '#ff6666',
    fontSize: 12,
    marginBottom: 10,
  },
  colorGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  colorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 10,
  },
  colorBox: {
    width: 16,
    height: 16,
    borderRadius: 3,
    marginRight: 5,
    borderWidth: 1,
    borderColor: '#fff',
  },
  colorLabel: {
    color: '#fff',
    fontSize: 11,
  },
});
