import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { BoardState, PIECE_UNICODE, PieceType } from '../types';

interface ChessBoard2DProps {
  boardState: BoardState;
  size?: number;
}

const LIGHT_SQUARE = '#F0D9B5';
const DARK_SQUARE = '#B58863';

export function ChessBoard2D({ boardState, size = 320 }: ChessBoard2DProps) {
  const squareSize = size / 8;

  const renderPiece = (piece: PieceType) => {
    if (!piece) return null;
    return (
      <Text style={[styles.piece, { fontSize: squareSize * 0.7 }]}>
        {PIECE_UNICODE[piece]}
      </Text>
    );
  };

  const renderSquare = (row: number, col: number) => {
    const isLight = (row + col) % 2 === 0;
    const piece = boardState[row][col];
    const file = String.fromCharCode(97 + col);
    const rank = 8 - row;

    return (
      <View
        key={`${row}-${col}`}
        style={[
          styles.square,
          {
            width: squareSize,
            height: squareSize,
            backgroundColor: isLight ? LIGHT_SQUARE : DARK_SQUARE,
          },
        ]}
      >
        {renderPiece(piece)}
        {/* File labels on bottom row */}
        {row === 7 && (
          <Text style={[styles.fileLabel, { color: isLight ? DARK_SQUARE : LIGHT_SQUARE }]}>
            {file}
          </Text>
        )}
        {/* Rank labels on left column */}
        {col === 0 && (
          <Text style={[styles.rankLabel, { color: isLight ? DARK_SQUARE : LIGHT_SQUARE }]}>
            {rank}
          </Text>
        )}
      </View>
    );
  };

  return (
    <View style={[styles.board, { width: size, height: size }]}>
      {boardState.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.row}>
          {row.map((_, colIndex) => renderSquare(rowIndex, colIndex))}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  board: {
    borderWidth: 2,
    borderColor: '#333',
    borderRadius: 4,
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
  },
  square: {
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  piece: {
    textAlign: 'center',
  },
  fileLabel: {
    position: 'absolute',
    bottom: 1,
    right: 3,
    fontSize: 10,
    fontWeight: 'bold',
  },
  rankLabel: {
    position: 'absolute',
    top: 1,
    left: 3,
    fontSize: 10,
    fontWeight: 'bold',
  },
});
