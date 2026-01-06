import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import Svg, { Line, Polygon } from 'react-native-svg';
import { BoardState, PieceType } from '../types';

interface ChessBoard2DProps {
  boardState: BoardState;
  size?: number;
  highlightFrom?: string;
  highlightTo?: string;
}

const LIGHT_SQUARE = '#F0D9B5';
const DARK_SQUARE = '#B58863';
const HIGHLIGHT_FROM = 'rgba(255, 235, 59, 0.6)';
const HIGHLIGHT_TO = 'rgba(76, 175, 80, 0.6)';
const ARROW_COLOR = 'rgba(255, 80, 80, 0.9)';

// Piece images mapping
const PIECE_IMAGES: Record<string, any> = {
  'K': require('../../assets/pieces/white_king.png'),
  'Q': require('../../assets/pieces/white_queen.png'),
  'R': require('../../assets/pieces/white_rook.png'),
  'B': require('../../assets/pieces/white_bishop.png'),
  'N': require('../../assets/pieces/white_knight.png'),
  'P': require('../../assets/pieces/white_pawn.png'),
  'k': require('../../assets/pieces/black_king.png'),
  'q': require('../../assets/pieces/black_queen.png'),
  'r': require('../../assets/pieces/black_rook.png'),
  'b': require('../../assets/pieces/black_bishop.png'),
  'n': require('../../assets/pieces/black_knight.png'),
  'p': require('../../assets/pieces/black_pawn.png'),
};

function parseSquare(square: string): { row: number; col: number } | null {
  if (!square || square.length !== 2) return null;
  const col = square.charCodeAt(0) - 97;
  const row = 8 - parseInt(square[1], 10);
  if (col < 0 || col > 7 || row < 0 || row > 7) return null;
  return { row, col };
}

export function ChessBoard2D({ 
  boardState, 
  size = 320,
  highlightFrom,
  highlightTo,
}: ChessBoard2DProps) {
  const squareSize = size / 8;
  
  const fromSquare = parseSquare(highlightFrom || '');
  const toSquare = parseSquare(highlightTo || '');

  const renderPiece = (piece: PieceType, squareSize: number) => {
    if (!piece) return null;
    const imageSource = PIECE_IMAGES[piece];
    if (!imageSource) return null;
    
    return (
      <Image 
        source={imageSource} 
        style={{ 
          width: squareSize * 0.85, 
          height: squareSize * 0.85,
        }} 
        resizeMode="contain"
      />
    );
  };

  const renderSquare = (row: number, col: number) => {
    const isLight = (row + col) % 2 === 0;
    const piece = boardState[row][col];
    const file = String.fromCharCode(97 + col);
    const rank = 8 - row;
    
    const isFromSquare = fromSquare && fromSquare.row === row && fromSquare.col === col;
    const isToSquare = toSquare && toSquare.row === row && toSquare.col === col;
    
    let backgroundColor = isLight ? LIGHT_SQUARE : DARK_SQUARE;
    if (isFromSquare) backgroundColor = HIGHLIGHT_FROM;
    if (isToSquare) backgroundColor = HIGHLIGHT_TO;

    return (
      <View
        key={`${row}-${col}`}
        style={[
          styles.square,
          {
            width: squareSize,
            height: squareSize,
            backgroundColor,
          },
        ]}
      >
        {renderPiece(piece, squareSize)}
        
        {row === 7 && (
          <Text style={[styles.fileLabel, { color: isLight ? DARK_SQUARE : LIGHT_SQUARE }]}>
            {file}
          </Text>
        )}
        {col === 0 && (
          <Text style={[styles.rankLabel, { color: isLight ? DARK_SQUARE : LIGHT_SQUARE }]}>
            {rank}
          </Text>
        )}
      </View>
    );
  };

  const renderArrow = () => {
    if (!fromSquare || !toSquare) return null;

    const fromX = (fromSquare.col + 0.5) * squareSize;
    const fromY = (fromSquare.row + 0.5) * squareSize;
    const toX = (toSquare.col + 0.5) * squareSize;
    const toY = (toSquare.row + 0.5) * squareSize;

    const angle = Math.atan2(toY - fromY, toX - fromX);
    const arrowHeadLength = 15;
    const arrowHeadWidth = 10;

    const shortenBy = 18;
    const endX = toX - Math.cos(angle) * shortenBy;
    const endY = toY - Math.sin(angle) * shortenBy;

    const tipX = toX - Math.cos(angle) * 5;
    const tipY = toY - Math.sin(angle) * 5;
    const baseX1 = tipX - arrowHeadLength * Math.cos(angle) - arrowHeadWidth * Math.sin(angle);
    const baseY1 = tipY - arrowHeadLength * Math.sin(angle) + arrowHeadWidth * Math.cos(angle);
    const baseX2 = tipX - arrowHeadLength * Math.cos(angle) + arrowHeadWidth * Math.sin(angle);
    const baseY2 = tipY - arrowHeadLength * Math.sin(angle) - arrowHeadWidth * Math.cos(angle);

    return (
      <Svg 
        style={[styles.arrowOverlay, { width: size, height: size }]} 
        pointerEvents="none"
      >
        <Line
          x1={fromX}
          y1={fromY}
          x2={endX}
          y2={endY}
          stroke={ARROW_COLOR}
          strokeWidth={8}
          strokeLinecap="round"
        />
        <Polygon
          points={`${tipX},${tipY} ${baseX1},${baseY1} ${baseX2},${baseY2}`}
          fill={ARROW_COLOR}
        />
      </Svg>
    );
  };

  return (
    <View style={[styles.board, { width: size, height: size }]}>
      {boardState.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.row}>
          {row.map((_, colIndex) => renderSquare(rowIndex, colIndex))}
        </View>
      ))}
      {renderArrow()}
    </View>
  );
}

const styles = StyleSheet.create({
  board: {
    borderWidth: 2,
    borderColor: '#333',
    borderRadius: 8,
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
  fileLabel: {
    position: 'absolute',
    bottom: 2,
    right: 4,
    fontSize: 10,
    fontWeight: 'bold',
  },
  rankLabel: {
    position: 'absolute',
    top: 2,
    left: 4,
    fontSize: 10,
    fontWeight: 'bold',
  },
  arrowOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
});
