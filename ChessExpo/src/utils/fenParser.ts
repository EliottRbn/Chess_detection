
import { BoardState, createEmptyBoard, PieceType } from '../types';

export function parseFEN(fen: string): BoardState {
  const board = createEmptyBoard();
  const [position] = fen.split(' ');
  const rows = position.split('/');

  rows.forEach((rowStr, rowIndex) => {
    let colIndex = 0;
    for (let i = 0; i < rowStr.length; i++) {
      const char = rowStr[i];
      if (/\d/.test(char)) {
        colIndex += parseInt(char, 10);
      } else {
        if (colIndex < 8) {
          board[rowIndex][colIndex] = char as PieceType;
          colIndex++;
        }
      }
    }
  });

  return board;
}
