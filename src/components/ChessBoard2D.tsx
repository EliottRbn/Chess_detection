/**
 * 2D Chessboard Component - chess.com style
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { type BoardState, getPieceChar } from '../utils/boardMapping';

// Chess.com style colors
const LIGHT_SQUARE = '#EEEED2';
const DARK_SQUARE = '#769656';
const PIECE_WHITE_COLOR = '#FFFFFF';
const PIECE_BLACK_COLOR = '#000000';

// Unicode chess pieces
const PIECE_SYMBOLS: Record<string, string> = {
  K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟',
};

type Props = {
  boardState: BoardState;
  size?: number;
  showNotation?: boolean;
};

export function ChessBoard2D({ boardState, size = 320, showNotation = true }: Props) {
  const squareSize = size / 8;
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const ranks = [8, 7, 6, 5, 4, 3, 2, 1];

  // Count pieces for debugging
  let pieceCount = 0;
  boardState.forEach(row => {
    row.forEach(square => {
      if (square.piece) pieceCount++;
    });
  });
  console.log(`[ChessBoard2D] Rendering board with ${pieceCount} pieces`);

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      {boardState.map((row, rankIdx) => (
        <View key={rankIdx} style={styles.row}>
          {row.map((square, fileIdx) => {
            const isLight = (rankIdx + fileIdx) % 2 === 0;
            const piece = square.piece;
            const pieceChar = piece ? getPieceChar(piece) : null;
            const pieceSymbol = pieceChar ? PIECE_SYMBOLS[pieceChar] : null;
            const isWhitePiece = pieceChar ? pieceChar === pieceChar.toUpperCase() : false;

            // Debug logging for pieces
            if (piece) {
              if (pieceSymbol) {
                console.log(`[ChessBoard2D] [${rankIdx},${fileIdx}] ${square.notation}: ${piece.label} → '${pieceChar}' → ${pieceSymbol}`);
              } else {
                console.warn(`[ChessBoard2D] ❌ [${rankIdx},${fileIdx}] ${square.notation}: ${piece.label} → '${pieceChar}' → NO SYMBOL!`);
              }
            }

            return (
              <View
                key={`${rankIdx}-${fileIdx}`}
                style={[
                  styles.square,
                  { 
                    width: squareSize, 
                    height: squareSize,
                    backgroundColor: isLight ? LIGHT_SQUARE : DARK_SQUARE,
                  },
                ]}
              >
                {/* Piece */}
                {pieceSymbol && (
                  <Text
                    style={[
                      styles.piece,
                      { 
                        fontSize: squareSize * 0.75,
                        color: isWhitePiece ? PIECE_WHITE_COLOR : PIECE_BLACK_COLOR,
                        textShadowColor: isWhitePiece ? '#000' : '#fff',
                        textShadowOffset: { width: 1, height: 1 },
                        textShadowRadius: 2,
                      },
                    ]}
                  >
                    {pieceSymbol}
                  </Text>
                )}
                
                {/* Confidence indicator */}
                {piece && (
                  <View 
                    style={[
                      styles.confidenceBadge,
                      { 
                        backgroundColor: piece.confidence > 0.7 ? '#16a34a' : 
                                        piece.confidence > 0.4 ? '#f59e0b' : '#dc2626'
                      }
                    ]}
                  >
                    <Text style={styles.confidenceText}>
                      {Math.round(piece.confidence * 100)}
                    </Text>
                  </View>
                )}

                {/* File letter (bottom row) */}
                {showNotation && rankIdx === 7 && (
                  <Text 
                    style={[
                      styles.fileLabel, 
                      { color: isLight ? DARK_SQUARE : LIGHT_SQUARE }
                    ]}
                  >
                    {files[fileIdx]}
                  </Text>
                )}

                {/* Rank number (left column) */}
                {showNotation && fileIdx === 0 && (
                  <Text 
                    style={[
                      styles.rankLabel, 
                      { color: isLight ? DARK_SQUARE : LIGHT_SQUARE }
                    ]}
                  >
                    {ranks[rankIdx]}
                  </Text>
                )}
              </View>
            );
          })}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#5D4E37',
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
    fontWeight: '400',
  },
  confidenceBadge: {
    position: 'absolute',
    bottom: 2,
    right: 2,
    paddingHorizontal: 3,
    paddingVertical: 1,
    borderRadius: 4,
    minWidth: 20,
    alignItems: 'center',
  },
  confidenceText: {
    fontSize: 8,
    fontWeight: '700',
    color: '#fff',
  },
  fileLabel: {
    position: 'absolute',
    bottom: 2,
    right: 4,
    fontSize: 10,
    fontWeight: '600',
  },
  rankLabel: {
    position: 'absolute',
    top: 2,
    left: 4,
    fontSize: 10,
    fontWeight: '600',
  },
});
