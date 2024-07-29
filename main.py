import sys
import time
from typing import List, Tuple, Optional

# Constants
EMPTY = 0
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = range(1, 7)
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = range(7, 13)

# Bitboard constants
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
RANK_1 = 0x00000000000000FF                                                                                                                                                                                                                                        
RANK_8 = 0xFF00000000000000

# Piece values and position tables (simplified)
PIECE_VALUES = [0, 100, 300, 300, 500, 900, 20000, -100, -300, -300, -500, -900, -20000]
POSITION_TABLES = [
    # Pawn
    [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    # Knight
    [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    # Bishop
    [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    # Rook
    [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    # Queen
    [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    # King
    [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ],
    # Empty (placeholder for index 0)
    [0] * 64
]

class ChessBoard:
    def __init__(self):
        self.pieces = [0] * 13
        self.color = 1 # 0 for white, 1 for black
        self.castling = 15  # 1111 in binary, representing KQkq castling rights
        self.ep = 0  # En passant target square
        self.halfmove = 0  # Halfmove clock for 50-move rule
        self.fullmove = 1  # Fullmove number
        self.history = []  # Move history for threefold repetition check
        self.init_board()

    def print_board(self):
        piece_symbols = {
            0: '.',
            1: '♙', 2: '♘', 3: '♗', 4: '♖', 5: '♕', 6: '♔',
            7: '♟', 8: '♞', 9: '♝', 10: '♜', 11: '♛', 12: '♚'
        }

        print('  a b c d e f g h')
        print(' +-----------------+')
        for rank in range(7, -1, -1):
            print(f'{rank+1}|', end=' ')
            for file in range(8):
                square = rank * 8 + file
                piece = self.get_piece(square)
                print(piece_symbols[piece], end=' ')
            print(f'|{rank+1}')
        print(' +-----------------+')
        print('  a b c d e f g h')

        print(f"\nSide to move: {'White' if self.color == 0 else 'Black'}")
        print(f"Fullmove number: {self.fullmove}")
        print(f"Halfmove clock: {self.halfmove}")
        print(f"En passant square: {'-' if self.ep == 0 else chr(self.ep % 8 + 97) + str(self.ep // 8 + 1)}")
        print(f"Castling rights: {'K' if self.castling & 1 else ''}{'Q' if self.castling & 2 else ''}{'k' if self.castling & 4 else ''}{'q' if self.castling & 8 else ''}")
    def init_board(self):
        self.pieces[1] = 0x000000000000FF00  # White pawns
        self.pieces[2] = 0x0000000000000042  # White knights
        self.pieces[3] = 0x0000000000000024  # White bishops
        self.pieces[4] = 0x0000000000000081  # White rooks
        self.pieces[5] = 0x0000000000000008  # White queen
        self.pieces[6] = 0x0000000000000010  # White king
        self.pieces[7] = 0x00FF000000000000  # Black pawns
        self.pieces[8] = 0x4200000000000000  # Black knights
        self.pieces[9] = 0x2400000000000000  # Black bishops
        self.pieces[10] = 0x8100000000000000  # Black rooks
        self.pieces[11] = 0x0800000000000000  # Black queen
        self.pieces[12] = 0x1000000000000000  # Black king
    def get_piece(self, square: int) -> int:
        for i, bb in enumerate(self.pieces):
            if bb & (1 << square):
                return i
        return EMPTY

    def make_move(self, move: Tuple[int, int, int]):
        from_sq, to_sq, promotion = move
        piece = self.get_piece(from_sq)
        captured = self.get_piece(to_sq)

        # Store state for undo_move
        state = (self.pieces[:], self.color, self.castling, self.ep, self.halfmove, self.fullmove)
        self.history.append(state)

        # Move piece
        self.pieces[piece] ^= (1 << from_sq) | (1 << to_sq)

        # Handle capture
        if captured:
            self.pieces[captured] &= ~(1 << to_sq)

        # Handle promotion
        if promotion:
            self.pieces[piece] &= ~(1 << to_sq)
            self.pieces[promotion] |= 1 << to_sq

        # Handle castling
        if piece in [WHITE_KING, BLACK_KING] and abs(from_sq - to_sq) == 2:
            if to_sq > from_sq:  # Kingside
                rook_from = from_sq + 3
                rook_to = from_sq + 1
            else:  # Queenside
                rook_from = from_sq - 4
                rook_to = from_sq - 1
            rook = WHITE_ROOK if piece == WHITE_KING else BLACK_ROOK
            self.pieces[rook] ^= (1 << rook_from) | (1 << rook_to)

        # Handle en passant
        if piece in [WHITE_PAWN, BLACK_PAWN] and to_sq == self.ep:
            cap_sq = to_sq + (-8 if self.color else 8)
            captured = self.get_piece(cap_sq)
            self.pieces[captured] &= ~(1 << cap_sq)

        # Update castling rights
        self.update_castling_rights(from_sq, to_sq)

        # Update en passant square
        self.ep = 0
        if piece in [WHITE_PAWN, BLACK_PAWN] and abs(from_sq - to_sq) == 16:
            self.ep = (from_sq + to_sq) // 2

        # Update move counters
        self.halfmove = 0 if piece in [WHITE_PAWN, BLACK_PAWN] or captured else self.halfmove + 1
        self.fullmove += self.color

        # Switch side to move
        self.color ^= 1

    def undo_move(self):
        if not self.history:
            return
        self.pieces, self.color, self.castling, self.ep, self.halfmove, self.fullmove = self.history.pop()

    def update_castling_rights(self, from_sq: int, to_sq: int):
        # Remove castling rights if king or rook moves
        if from_sq == 4 or to_sq == 4:  # White king
            self.castling &= 0b1100
        elif from_sq == 60 or to_sq == 60:  # Black king
            self.castling &= 0b0011
        elif from_sq == 0 or to_sq == 0:  # White queenside rook
            self.castling &= 0b1110
        elif from_sq == 7 or to_sq == 7:  # White kingside rook
            self.castling &= 0b1101
        elif from_sq == 56 or to_sq == 56:  # Black queenside rook
            self.castling &= 0b1011
        elif from_sq == 63 or to_sq == 63:  # Black kingside rook
            self.castling &= 0b0111

    def generate_moves(self) -> List[Tuple[int, int, int]]:
        moves = []
        for i in range(64):
            piece = self.get_piece(i)
            if piece and (piece < 7) == (self.color == 0):
                piece_moves = self.generate_piece_moves(i, piece)
                for move in piece_moves:
                    from_sq, to_sq, promotion = move
                    print(f"Generated move: {chr(from_sq%8+97)}{from_sq//8+1}{chr(to_sq%8+97)}{to_sq//8+1}")
                moves.extend(piece_moves)
        legal_moves = [move for move in moves if self.is_legal_move(move)]
        print(f"Generated {len(moves)} moves, {len(legal_moves)} are legal")
        return legal_moves

    def generate_piece_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        if piece in [WHITE_PAWN, BLACK_PAWN]:
            moves.extend(self.generate_pawn_moves(square, piece))
        elif piece in [WHITE_KNIGHT, BLACK_KNIGHT]:
            moves.extend(self.generate_knight_moves(square, piece))
        elif piece in [WHITE_BISHOP, BLACK_BISHOP, WHITE_QUEEN, BLACK_QUEEN]:
            moves.extend(self.generate_diagonal_moves(square, piece))
        if piece in [WHITE_ROOK, BLACK_ROOK, WHITE_QUEEN, BLACK_QUEEN]:
            moves.extend(self.generate_straight_moves(square, piece))
        elif piece in [WHITE_KING, BLACK_KING]:
            moves.extend(self.generate_king_moves(square, piece))
        return moves

    def generate_pawn_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        direction = 8 if piece == WHITE_PAWN else -8
        start_rank = 1 if piece == WHITE_PAWN else 6

        # Single push
        target = square + direction
        if 0 <= target < 64 and not self.get_piece(target):
            if target // 8 in [0, 7]:  # Promotion
                for promotion in [2, 3, 4, 5]:  # Knight, Bishop, Rook, Queen
                    moves.append((square, target, piece + promotion - 1))
            else:
                moves.append((square, target, 0))

            # Double push
            if square // 8 == start_rank:
                target = square + 2 * direction
                if 0 <= target < 64 and not self.get_piece(target):
                    moves.append((square, target, 0))

        # Captures
        for offset in [-1, 1]:
            target = square + direction + offset
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) == 1:
                captured = self.get_piece(target)
                if captured and (captured < 7) != (piece < 7):
                    if target // 8 in [0, 7]:  # Promotion
                        for promotion in [2, 3, 4, 5]:  # Knight, Bishop, Rook, Queen
                            moves.append((square, target, piece + promotion - 1))
                    else:
                        moves.append((square, target, 0))
                elif target == self.ep:  # En passant
                    moves.append((square, target, 0))

        return moves

    def generate_knight_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for offset in [-17, -15, -10, -6, 6, 10, 15, 17]:
            target = square + offset
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 2:
                captured = self.get_piece(target)
                if not captured or (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))
        return moves

    def generate_diagonal_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for direction in [-9, -7, 7, 9]:
            target = square + direction
            while 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
                captured = self.get_piece(target)
                if not captured:
                    moves.append((square, target, 0))
                elif (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))
                    break
                else:
                    break
                target += direction
        return moves

    def generate_straight_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for direction in [-8, -1, 1, 8]:
            target = square + direction
            while 0 <= target < 64 and (direction in [-8, 8] or abs((square % 8) - (target % 8)) <= 1):
                captured = self.get_piece(target)
                if not captured:
                    moves.append((square, target, 0))
                elif (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))
                    break
                else:
                    break
                target += direction
        return moves

    def generate_king_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for offset in [-9, -8, -7, -1, 1, 7, 8, 9]:
            target = square + offset
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
                captured = self.get_piece(target)
                if not captured or (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))

        # Castling
        if piece == WHITE_KING and square == 4:
            if self.castling & 1 and not self.get_piece(5) and not self.get_piece(6):
                moves.append((square, 6, 0))
            if self.castling & 2 and not self.get_piece(3) and not self.get_piece(2) and not self.get_piece(1):
                moves.append((square, 2, 0))
        elif piece == BLACK_KING and square == 60:
            if self.castling & 4 and not self.get_piece(61) and not self.get_piece(62):
                moves.append((square, 62, 0))
            if self.castling & 8 and not self.get_piece(59) and not self.get_piece(58) and not self.get_piece(57):
                moves.append((square, 58, 0))

        return moves

    def is_legal_move(self, move: Tuple[int, int, int]) -> bool:
        from_sq, to_sq, promotion = move
        piece = self.get_piece(from_sq)
        print(f"Checking move: {chr(from_sq%8+97)}{from_sq//8+1}{chr(to_sq%8+97)}{to_sq//8+1}")
        print(f"Piece: {piece}")
        self.make_move(move)
        is_legal = not self.is_check()
        self.undo_move()
        print(f"Is legal: {is_legal}")
        return is_legal

    def is_check(self) -> bool:
        king = WHITE_KING if self.color == 0 else BLACK_KING
        king_square = self.pieces[king].bit_length() - 1
        is_in_check = self.is_square_attacked(king_square)
        print(f"King on square {king_square}, is in check: {is_in_check}")
        return is_in_check

    def is_square_attacked(self, square: int) -> bool:
        print(f"Checking if square {square} is attacked")
        pawn = BLACK_PAWN if self.color == 0 else WHITE_PAWN
        pawn_attacks = [-7, -9] if self.color == 0 else [7, 9]
        for attack in pawn_attacks:
            target = square + attack
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) == 1:
                if self.pieces[pawn] & (1 << target):
                    return True

        # Check for knight attacks
        knight = BLACK_KNIGHT if self.color == 0 else WHITE_KNIGHT
        knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
        for move in knight_moves:
            target = square + move
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 2:
                if self.pieces[knight] & (1 << target):
                    return True
        # Check for diagonal attacks (bishop and queen)
        bishop = BLACK_BISHOP if self.color == 0 else WHITE_BISHOP
        queen = BLACK_QUEEN if self.color == 0 else WHITE_QUEEN
        for direction in [-9, -7, 7, 9]:
            target = square + direction
            while 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
                if self.pieces[bishop] & (1 << target) or self.pieces[queen] & (1 << target):
                    return True
                if self.get_piece(target) != EMPTY:
                    break
                target += direction

        # Check for straight attacks (rook and queen)
        rook = BLACK_ROOK if self.color == 0 else WHITE_ROOK
        for direction in [-8, -1, 1, 8]:
            target = square + direction
            while 0 <= target < 64 and (direction in [-8, 8] or abs((square % 8) - (target % 8)) <= 1):
                if self.pieces[rook] & (1 << target) or self.pieces[queen] & (1 << target):
                    return True
                if self.get_piece(target) != EMPTY:
                    break
                target += direction

        # Check for king attacks
        king = BLACK_KING if self.color == 0 else WHITE_KING
        king_moves = [-9, -8, -7, -1, 1, 7, 8, 9]
        for move in king_moves:
            if 0 <= square + move < 64 and abs((square % 8) - ((square + move) % 8)) <= 1:
                if self.pieces[king] & (1 << (square + move)):
                    return True

        return False

    def is_checkmate(self) -> bool:
        return self.is_check() and not self.generate_moves()

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.generate_moves()

    def is_insufficient_material(self) -> bool:
        # King vs. King
        if sum(bin(x).count('1') for x in self.pieces) == 2:
            return True

        # King and Bishop vs. King or King and Knight vs. King
        if sum(bin(x).count('1') for x in self.pieces) == 3:
            if self.pieces[WHITE_BISHOP] or self.pieces[BLACK_BISHOP] or \
               self.pieces[WHITE_KNIGHT] or self.pieces[BLACK_KNIGHT]:
                return True

        # King and Bishop vs. King and Bishop with the bishops on the same color
        if sum(bin(x).count('1') for x in self.pieces) == 4:
            if self.pieces[WHITE_BISHOP] and self.pieces[BLACK_BISHOP]:
                white_bishop_square = self.pieces[WHITE_BISHOP].bit_length() - 1
                black_bishop_square = self.pieces[BLACK_BISHOP].bit_length() - 1
                if (white_bishop_square + black_bishop_square) % 2 == 0:
                    return True

        return False

    def is_threefold_repetition(self) -> bool:
        current_position = self.get_position_key()
        repetition_count = 1
        for past_position in reversed(self.history):
            if past_position[0] == current_position:
                repetition_count += 1
                if repetition_count == 3:
                    return True
        return False

    def is_fifty_move_rule(self) -> bool:
        return self.halfmove >= 100

    def is_draw(self) -> bool:
        return self.is_stalemate() or self.is_insufficient_material() or \
               self.is_threefold_repetition() or self.is_fifty_move_rule()

    def get_position_key(self) -> Tuple:
        return tuple(self.pieces + [self.castling, self.ep])

    def get_fen(self) -> str:
        fen = []
        for rank in range(7, -1, -1):
            empty = 0
            rank_fen = []
            for file in range(8):
                piece = self.get_piece(rank * 8 + file)
                if piece == EMPTY:
                    empty += 1
                else:
                    if empty > 0:
                        rank_fen.append(str(empty))
                        empty = 0
                    rank_fen.append("PNBRQKpnbrqk"[piece - 1])
            if empty > 0:
                rank_fen.append(str(empty))
            fen.append("".join(rank_fen))

        fen = "/".join(fen)
        fen += " w " if self.color == 0 else " b "

        castling = ""
        if self.castling & 1: castling += "K"
        if self.castling & 2: castling += "Q"
        if self.castling & 4: castling += "k"
        if self.castling & 8: castling += "q"
        fen += castling if castling else "-"

        fen += " " + ("-" if self.ep == 0 else chr(self.ep % 8 + 97) + str(self.ep // 8 + 1))
        fen += f" {self.halfmove} {self.fullmove}"

        return fen

    def set_from_fen(self, fen: str):
        parts = fen.split()
        self.pieces = [0] * 13

        # Board position
        for rank, row in enumerate(parts[0].split('/')[::-1]):
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)
                else:
                    piece = "PNBRQKpnbrqk".index(char) + 1
                    self.pieces[piece] |= 1 << (rank * 8 + file)
                    file += 1

        # Active color
        self.color = 0 if parts[1] == 'w' else 1

        # Castling availability
        self.castling = 0
        if 'K' in parts[2]: self.castling |= 1
        if 'Q' in parts[2]: self.castling |= 2
        if 'k' in parts[2]: self.castling |= 4
        if 'q' in parts[2]: self.castling |= 8

        # En passant target square
        self.ep = 0 if parts[3] == '-' else (ord(parts[3][0]) - 97) + (int(parts[3][1]) - 1) * 8

        # Halfmove clock and fullmove number
        self.halfmove = int(parts[4])
        self.fullmove = int(parts[5])

        self.history = []

def evaluate_board(board: ChessBoard) -> int:
    if board.is_checkmate():
        return -1000000 if board.color == 0 else 1000000
    if board.is_draw():
        return 0

    score = 0
    for i, bb in enumerate(board.pieces):
        while bb:
            square = bb.bit_length() - 1
            score += PIECE_VALUES[i]
            score += POSITION_TABLES[i % 7][63 - square if i >= 7 else square]
            bb &= bb - 1

    return score if board.color == 0 else -score

def minimax(board: ChessBoard, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:
    if depth == 0 or board.is_checkmate() or board.is_draw():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.generate_moves():
            board.make_move(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.undo_move()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.generate_moves():
            board.make_move(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.undo_move()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def order_moves(board: ChessBoard, moves: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    def move_score(move):
        from_sq, to_sq, promotion = move
        piece = board.get_piece(from_sq)
        captured = board.get_piece(to_sq)
        score = 0
        if captured:
            score += 10 * abs(PIECE_VALUES[captured]) - abs(PIECE_VALUES[piece])
        if promotion:
            score += 1000
        if to_sq in [27, 28, 35, 36]:  # Central squares
            score += 10
        return score

    return sorted(moves, key=move_score, reverse=True)

def iterative_deepening(board: ChessBoard, time_limit: float) -> Optional[Tuple[int, int, int]]:
    start_time = time.time()
    best_move = None
    depth = 1
    max_depth = 20  # Increase this for stronger play, but be careful of time constraints

    legal_moves = order_moves(board, board.generate_moves())
    print(f"Number of legal moves: {len(legal_moves)}")
    if not legal_moves:
        return None

    while time.time() - start_time < time_limit and depth <= max_depth:
        print(f"Searching at depth {depth}")
        best_eval = float('-inf') if board.color == 0 else float('inf')
        for move in legal_moves:
            board.make_move(move)
            eval = minimax(board, depth - 1, float('-inf'), float('inf'), board.color == 1)
            board.undo_move()

            if board.color == 0 and eval > best_eval:
                best_eval = eval
                best_move = move
            elif board.color == 1 and eval < best_eval:
                best_eval = eval
                best_move = move

        print(f"Best move at depth {depth}: {best_move}")
        depth += 1
        if time.time() - start_time >= time_limit * 0.8:  # Use 80% of time for search, 20% for move execution
            print(f"Time limit approaching, stopping at depth {depth-1}")
            break

    return best_move

class UCIEngine:
    def __init__(self):
        self.board = ChessBoard()

    def uci(self):
        print("id name ComprehensiveChessEngine")
        print("id author Your Name")
        print("uciok")

    def isready(self):
        print("readyok")

    def ucinewgame(self):
        self.board = ChessBoard()

    def position(self, command):
        parts = command.split()
        if parts[1] == "startpos":
            self.board = ChessBoard()
            move_index = 3
        elif parts[1] == "fen":
            fen = " ".join(parts[2:8])
            self.board.set_from_fen(fen)
            move_index = 9
        else:
            return

        if len(parts) > move_index and parts[move_index] == "moves":
            for move in parts[move_index + 1:]:
                from_sq = (ord(move[0]) - ord('a')) + (8 * (ord(move[1]) - ord('1')))
                to_sq = (ord(move[2]) - ord('a')) + (8 * (ord(move[3]) - ord('1')))
                promotion = 0
                if len(move) == 5:
                    promotion = {'q': 5, 'r': 4, 'b': 3, 'n': 2}[move[4]]
                self.board.make_move((from_sq, to_sq, promotion))

    def go(self, command):
        parts = command.split()
        time_limit = 5.0  # Default time limit
        for i in range(0, len(parts), 2):
            if parts[i] == "wtime" and self.board.color == 0:
                time_limit = int(parts[i + 1]) / 1000.0 / 30  # Use 1/30 of remaining time
            elif parts[i] == "btime" and self.board.color == 1:
                time_limit = int(parts[i + 1]) / 1000.0 / 30  # Use 1/30 of remaining time
            elif parts[i] == "movetime":
                time_limit = int(parts[i + 1]) / 1000.0

        best_move = iterative_deepening(self.board, time_limit)
        if best_move:
            from_sq, to_sq, promotion = best_move
            move_str = f"{chr(from_sq % 8 + ord('a'))}{from_sq // 8 + 1}{chr(to_sq % 8 + ord('a'))}{to_sq // 8 + 1}"
            if promotion:
                move_str += "qrnb"[promotion - 2]
            print(f"bestmove {move_str}")
        else:
            print("No legal moves available or search failed")

        print("Board after move:")
        self.board.print_board()

    def quit(self):
        sys.exit()

def main():
    engine = UCIEngine()
    engine.uci()
    engine.isready()
    engine.ucinewgame()
    engine.position("position startpos")

    while True:
        command = input("Enter your move (e.g., 'e2e4') or 'go' to let the engine move, or 'quit' to exit: ")
        if command == "quit":
            break
        elif command == "go":
            engine.go("go movetime 15000")  # 10 seconds per move
        else:
            # Assume the input is a move
            from_sq = (ord(command[0]) - ord('a')) + (8 * (ord(command[1]) - ord('1')))
            to_sq = (ord(command[2]) - ord('a')) + (8 * (ord(command[3]) - ord('1')))
            promotion = 0
            if len(command) == 5:
                promotion = {'q': 5, 'r': 4, 'b': 3, 'n': 2}[command[4]]
            move = (from_sq, to_sq, promotion)
            engine.board.make_move(move)
            print("Your move:")
            engine.board.print_board()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()