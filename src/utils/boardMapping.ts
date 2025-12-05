/**
 * Board Mapping - Map piece detections to chess squares
 * Uses boardGridAnalysis for precise grid detection
 */

import { type Detection } from "../../hooks/useChessDetection";
import { 
  computeBoardGrid, 
  getSquareAtPoint, 
  squareToNotation,
  validateOrientation,
  rotateBoard,
  type BoardGrid,
  type GridSquare,
} from "./boardGridAnalysis";
import { type Point } from "./perspectiveTransform";

export type ChessSquare = {
  file: string; // a-h
  rank: number; // 1-8
  notation: string; // e.g., "e4"
  piece: Detection | null;
  gridSquare?: GridSquare; // Reference to grid square
};

export type BoardState = ChessSquare[][];

// Chess files
const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

/**
 * Main function: map piece detections to board squares
 * Uses precise grid calculation from board corners
 */
export function mapPiecesToBoard(
  detections: Detection[],
  boardPolygon: Array<{ x: number; y: number }> | undefined
): BoardState {
  // Initialize empty board
  const board: BoardState = [];
  for (let rankIdx = 0; rankIdx < 8; rankIdx++) {
    const row: ChessSquare[] = [];
    for (let fileIdx = 0; fileIdx < 8; fileIdx++) {
      // rankIdx 0 = top of display = rank 8
      const rank = 8 - rankIdx;
      row.push({
        file: FILES[fileIdx],
        rank,
        notation: `${FILES[fileIdx]}${rank}`,
        piece: null,
      });
    }
    board.push(row);
  }
  
  const pieces = detections.filter(d => d.label && d.bbox);
  console.log("[boardMapping] Pieces to map:", pieces.length);
  console.log("[boardMapping] Board polygon:", boardPolygon?.length, "corners");
  
  if (pieces.length === 0) {
    console.log("[boardMapping] No pieces to map, returning empty board");
    return board;
  }
  
  // If we have a valid board polygon, use precise grid analysis
  if (boardPolygon && boardPolygon.length >= 4) {
    console.log("[boardMapping] Using grid analysis with polygon");
    console.log("[boardMapping] Polygon corners:", JSON.stringify(boardPolygon.slice(0, 4)));
    
    // Compute grid from corners
    let grid = computeBoardGrid(boardPolygon);
    
    if (grid) {
      console.log("[boardMapping] Grid computed successfully");
      
      // Validate orientation using piece colors
      const pieceData = pieces.map(p => ({
        center: {
          x: (p.bbox![0] + p.bbox![2]) / 2,
          y: p.bbox![3] - 10, // Bottom center
        },
        isWhite: p.label?.startsWith('white') ?? false,
      }));
      
      const orientation = validateOrientation(grid, pieceData);
      console.log("[boardMapping] Orientation:", orientation);
      if (orientation === "black-bottom") {
        grid = rotateBoard(grid);
      }
      
      // Map each piece to its square
      let mappedCount = 0;
      let outsideGrid = 0;
      let outOfBounds = 0;
      
      console.log("[boardMapping] Starting piece mapping...");
      for (const p of pieces) {
        if (!p.bbox) continue;
        
        // Use bottom-center of piece (where it sits on the board)
        const pieceBottom: Point = {
          x: (p.bbox[0] + p.bbox[2]) / 2,
          y: p.bbox[3] - 10,
        };
        
        console.log(`[boardMapping] Mapping piece ${p.label} at (${pieceBottom.x.toFixed(1)}, ${pieceBottom.y.toFixed(1)})`);
        
        const gridSquare = getSquareAtPoint(pieceBottom, grid);
        if (!gridSquare) {
          console.log(`  ❌ Outside grid`);
          outsideGrid++;
          continue;
        }
        
        console.log(`  ✓ Found in grid square: file=${gridSquare.file}, rank=${gridSquare.rank} (${FILES[gridSquare.file]}${gridSquare.rank + 1})`);
        
        // Convert gridSquare to board indices
        // gridSquare.rank is 0-7 where 0 = rank 1 (bottom)
        // board[0] = rank 8 (top), board[7] = rank 1 (bottom)
        const boardRankIdx = 7 - gridSquare.rank;
        const boardFileIdx = gridSquare.file;
        
        console.log(`  Board indices: rankIdx=${boardRankIdx}, fileIdx=${boardFileIdx}`);
        
        if (boardRankIdx >= 0 && boardRankIdx < 8 && boardFileIdx >= 0 && boardFileIdx < 8) {
          const existing = board[boardRankIdx][boardFileIdx].piece;
          if (!existing || existing.confidence < p.confidence) {
            board[boardRankIdx][boardFileIdx].piece = p;
            board[boardRankIdx][boardFileIdx].gridSquare = gridSquare;
            console.log(`  ✅ Mapped to board[${boardRankIdx}][${boardFileIdx}]`);
            mappedCount++;
          } else {
            console.log(`  ⚠️  Square occupied by higher confidence piece`);
          }
        } else {
          console.log(`  ❌ Indices out of bounds!`);
          outOfBounds++;
        }
      }
      
      console.log(`[boardMapping] Summary: ${mappedCount} mapped, ${outsideGrid} outside grid, ${outOfBounds} out of bounds`);
      
      console.log("[boardMapping] Successfully mapped", mappedCount, "pieces");
      return board;
    } else {
      console.log("[boardMapping] Grid computation failed!");
    }
  }
  
  // Fallback: estimate grid from piece positions
  return mapPiecesFallback(pieces, board);
}

/**
 * Fallback mapping when no board polygon is available
 */
function mapPiecesFallback(pieces: Detection[], board: BoardState): BoardState {
  console.log("[mapPiecesFallback] Using fallback mapping for", pieces.length, "pieces");
  if (pieces.length === 0) return board;
  
  // Find bounds of all pieces
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  
  for (const p of pieces) {
    if (!p.bbox) continue;
    const cx = (p.bbox[0] + p.bbox[2]) / 2;
    const cy = p.bbox[3] - 10;
    minX = Math.min(minX, cx);
    maxX = Math.max(maxX, cx);
    minY = Math.min(minY, cy);
    maxY = Math.max(maxY, cy);
  }
  
  console.log(`[mapPiecesFallback] Bounds: X[${minX.toFixed(1)}, ${maxX.toFixed(1)}], Y[${minY.toFixed(1)}, ${maxY.toFixed(1)}]`);
  const width = maxX - minX || 1;
  const height = maxY - minY || 1;
  console.log(`[mapPiecesFallback] Grid size: ${width.toFixed(1)} x ${height.toFixed(1)}`);
  
  let mappedCount = 0;
  for (const p of pieces) {
    if (!p.bbox) continue;
    const cx = (p.bbox[0] + p.bbox[2]) / 2;
    const cy = p.bbox[3] - 10;
    
    const bx = (cx - minX) / width;
    const by = (cy - minY) / height;
    
    const fileIdx = Math.min(7, Math.max(0, Math.floor(bx * 8)));
    const rankIdx = Math.min(7, Math.max(0, Math.floor(by * 8)));
    
    console.log(`[mapPiecesFallback] ${p.label}: pos(${cx.toFixed(1)}, ${cy.toFixed(1)}) → normalized(${bx.toFixed(2)}, ${by.toFixed(2)}) → [${rankIdx}, ${fileIdx}]`);
    
    if (!board[rankIdx][fileIdx].piece) {
      board[rankIdx][fileIdx].piece = p;
      mappedCount++;
    } else {
      console.log(`  ⚠️ Square [${rankIdx}, ${fileIdx}] already occupied`);
    }
  }
  
  console.log(`[mapPiecesFallback] Mapped ${mappedCount}/${pieces.length} pieces`);
  return board;
}

/**
 * Get FEN-style piece character
 */
export function getPieceChar(piece: Detection): string {
  if (!piece.label) return '';
  
  const type = piece.label.split('-')[1];
  const isWhite = piece.label.startsWith('white');
  
  const pieceMap: Record<string, string> = {
    king: 'K',
    queen: 'Q',
    rook: 'R',
    bishop: 'B',
    knight: 'N',
    pawn: 'P',
  };
  
  const char = pieceMap[type] || '?';
  return isWhite ? char : char.toLowerCase();
}

/**
 * Generate FEN string from board state
 */
export function boardToFEN(board: BoardState): string {
  const rows: string[] = [];
  
  // board[0] = rank 8, board[7] = rank 1
  // FEN starts with rank 8
  for (let rankIdx = 0; rankIdx < 8; rankIdx++) {
    let row = '';
    let emptyCount = 0;
    
    for (let fileIdx = 0; fileIdx < 8; fileIdx++) {
      const piece = board[rankIdx][fileIdx].piece;
      if (piece) {
        if (emptyCount > 0) {
          row += emptyCount.toString();
          emptyCount = 0;
        }
        row += getPieceChar(piece);
      } else {
        emptyCount++;
      }
    }
    
    if (emptyCount > 0) {
      row += emptyCount.toString();
    }
    
    rows.push(row);
  }
  
  return rows.join('/');
}
