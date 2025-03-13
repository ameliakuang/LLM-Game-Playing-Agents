import os
import logging
import datetime
import sys
from pathlib import Path
import numpy as np
import chess
from stockfish import Stockfish
from dotenv import load_dotenv
import pandas as pd
import io
import contextlib
import chess.pgn
import chess.svg
from IPython.display import SVG, display
import time
import json
import matplotlib.pyplot as plt

# Try to import IPython for interactive display, but provide fallbacks
try:
    from IPython.display import SVG, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    # Define dummy display function for non-IPython environments
    def display(obj):
        if isinstance(obj, str):
            print(obj)
        else:
            print("Display object (IPython not available)")

os.environ['TRACE_CUSTOMLLM_MODEL'] = "anthropic.claude-3-5-haiku-20241022-v1:0"
os.environ['TRACE_CUSTOMLLM_URL'] = "http://3.213.219.83:4000/"
os.environ['TRACE_CUSTOMLLM_API_KEY'] = "sk-Xhglzhzo3JZ5oHCHacozzijW6Vs3mLpZ3YaoZMM6HbjT2wgCUlZizvvamJdmhtvs"
os.environ['TRACE_DEFAULT_LLM_BACKEND'] = 'CustomLLM'

import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.trace.bundle import ExceptionNode
from opto.optimizers import OptoPrime
from opto.trace.errors import ExecutionError

load_dotenv()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
base_trace_ckpt_dir = Path("trace_ckpt")
base_trace_ckpt_dir.mkdir(exist_ok=True)

# Create a separate debug helper class that doesn't interfere with Trace
class DebugHelper:
    """Helper class for debugging that doesn't interfere with Trace's model copying"""
    enabled = False
    info = {}
    
    @staticmethod
    def enable(enable=True):
        """Enable or disable debug mode"""
        DebugHelper.enabled = enable
    
    @staticmethod
    def set_info(info):
        """Set debug information"""
        DebugHelper.info = info
    
    @staticmethod
    def get_info():
        """Get debug information"""
        return DebugHelper.info

class ChessTracedEnv:
    def __init__(self, stockfish_path="/opt/homebrew/bin/stockfish", stockfish_depth=10):
        """Initialize chess environment with Stockfish as opponent"""
        self.board = chess.Board()
        try:
            self.stockfish = Stockfish(path=stockfish_path, depth=stockfish_depth)
            # Set Stockfish to a very low ELO rating (around beginner level)
            self.stockfish.set_skill_level(0)  # Lowest skill level
            self.stockfish.set_elo_rating(1000)  # Beginner ELO rating
        except Exception as e:
            logging.error(f"Failed to initialize Stockfish: {e}")
            raise
        self.game_over = False
        self.result = None
        self.obs = None
        self.init()
    
    def init(self):
        """Reset the environment to initial state"""
        self.board = chess.Board()
        self.game_over = False
        self.result = None
        self.update_stockfish()
        self.obs = self.get_observation()
    
    def close(self):
        """Close the environment"""
        # Nothing to close for chess, but keeping the method for consistency
        pass
    
    def __del__(self):
        """Destructor"""
        self.close()
    
    def update_stockfish(self):
        """Update Stockfish with the current board position"""
        self.stockfish.set_position([move.uci() for move in self.board.move_stack])
    
    def get_observation(self):
        """Get the current observation of the chess board"""
        # Create a dictionary with the current board state
        obs = {
            'board_fen': self.board.fen(),
            'legal_moves': [move.uci() for move in self.board.legal_moves],
            'turn': 'white' if self.board.turn else 'black',
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_insufficient_material': self.board.is_insufficient_material(),
            'is_game_over': self.board.is_game_over(),
            'halfmove_clock': self.board.halfmove_clock,
            'fullmove_number': self.board.fullmove_number,
            'piece_map': {chess.square_name(square): piece.symbol() for square, piece in self.board.piece_map().items()},
            'white_pieces': [chess.square_name(square) for square, piece in self.board.piece_map().items() if piece.color == chess.WHITE],
            'black_pieces': [chess.square_name(square) for square, piece in self.board.piece_map().items() if piece.color == chess.BLACK],
            'last_move': self.board.move_stack[-1].uci() if self.board.move_stack else None,
            'reward': 0.0  # Will be updated after moves
        }
        return obs
    
    @bundle()
    def reset(self):
        """Reset the environment and return the initial observation"""
        self.init()
        info = {}
        return self.obs, info
    
    def step(self, action):
        """Take a step in the environment with the given action (chess move)"""
        try:
            # Convert action to a chess move
            move_uci = action.data if isinstance(action, trace.Node) else action
            
            # Check if the move is legal
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                raise ValueError(f"Illegal move: {move_uci}")
            
            # Make the player's move
            self.board.push(move)
            self.update_stockfish()
            
            # Check if the game is over after player's move
            if self.board.is_game_over():
                self.game_over = True
                self.result = self.board.outcome()
                reward = self.calculate_reward()
                self.obs = self.get_observation()
                self.obs['reward'] = reward
                return self.obs, reward, True, False, {"result": self.result}
            
            # Make Stockfish's move
            stockfish_move = self.stockfish.get_best_move()
            if stockfish_move:
                self.board.push(chess.Move.from_uci(stockfish_move))
                self.update_stockfish()
            
            # Check if the game is over after Stockfish's move
            if self.board.is_game_over():
                self.game_over = True
                self.result = self.board.outcome()
            
            # Calculate reward
            reward = self.calculate_reward()
            
            # Update observation
            self.obs = self.get_observation()
            self.obs['reward'] = reward
            
        except Exception as e:
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The chess step operation raises an exception.",
                name="exception_step",
            )
            raise ExecutionError(e_node)
        
        @bundle()
        def step(action):
            """Take a step in the chess environment and return the next observation"""
            return self.obs
        
        next_obs = step(action)
        return next_obs, reward, self.game_over, False, {"result": self.result}
    
    def calculate_reward(self):
        """Calculate the reward for the current state"""
        if not self.game_over:
            # Use material advantage as reward during the game
            white_material = sum(len(self.board.pieces(piece_type, chess.WHITE)) * value 
                               for piece_type, value in [(chess.PAWN, 1), (chess.KNIGHT, 3), 
                                                        (chess.BISHOP, 3), (chess.ROOK, 5), 
                                                        (chess.QUEEN, 9)])
            black_material = sum(len(self.board.pieces(piece_type, chess.BLACK)) * value 
                               for piece_type, value in [(chess.PAWN, 1), (chess.KNIGHT, 3), 
                                                        (chess.BISHOP, 3), (chess.ROOK, 5), 
                                                        (chess.QUEEN, 9)])
            # Normalize the material advantage
            return (white_material - black_material) / 39.0  # 39 is the max possible material (9+9+5+5+3+3+3+3+1*8)
        else:
            # Game is over, calculate final reward
            if self.result and self.result.winner == chess.WHITE:
                return 1.0  # Player wins
            elif self.result and self.result.winner == chess.BLACK:
                return -1.0  # Stockfish wins
            else:
                return 0.0  # Draw
    
    def render(self):
        """Render the current board state"""
        print(self.board)
        print(f"Turn: {'White' if self.board.turn else 'Black'}")
        if self.game_over:
            print(f"Game over: {self.result}")
    
    def save_game_pgn(self, filename):
        """Save the current game as a PGN file"""
        game = chess.pgn.Game()
        game.headers["Event"] = "Trace Chess Agent vs Stockfish"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Trace Chess Agent"
        game.headers["Black"] = "Stockfish"
        game.headers["Result"] = "1-0" if self.result and self.result.winner == chess.WHITE else "0-1" if self.result and self.result.winner == chess.BLACK else "1/2-1/2"
        
        # Add the moves
        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)
        
        # Save to file
        with open(filename, "w") as f:
            f.write(str(game))
        
        return game
    
    def create_board_svg(self, last_move=None):
        """Create an SVG representation of the current board state"""
        return chess.svg.board(
            board=self.board,
            lastmove=last_move,
            size=400
        )

# Helper function to extract data from MessageNode objects
def extract_node_data(obj):
    """Extract data from a MessageNode object if needed"""
    if hasattr(obj, 'data'):
        return obj.data
    return obj

@trace.model
class ChessPolicy(Module):
    def init(self):
        # Initialize any parameters needed for the policy
        pass
    
    def __call__(self, obs):
        # Handle the case where obs might be a MessageNode
        if hasattr(obs, 'data'):
            obs_data = obs.data
        else:
            obs_data = obs
            
        # First evaluate the position
        position_evaluation = self.evaluate_position(obs)
        # Then select the best move based on the evaluation
        move = self.select_move(position_evaluation, obs)
        
        # Store debug information if debug mode is enabled
        if DebugHelper.enabled:
            # Safely extract values from obs_data
            board_fen = obs_data.get('board_fen') if isinstance(obs_data, dict) else None
            legal_moves = obs_data.get('legal_moves') if isinstance(obs_data, dict) else []
            
            # Extract data from any node objects before storing
            clean_position_evaluation = {}
            if position_evaluation:
                for k, v in position_evaluation.items():
                    clean_k = extract_node_data(k)
                    clean_v = extract_node_data(v)
                    clean_position_evaluation[clean_k] = clean_v
            
            move_str = extract_node_data(move)
            
            DebugHelper.set_info({
                'position_evaluation': clean_position_evaluation,
                'selected_move': move_str,
                'board_fen': board_fen,
                'legal_moves': legal_moves,
                'timestamp': time.time()
            })
        
        return move
    
    # Keep the helper functions as private methods for reference, but let the LLM discover them
    def _evaluate_material(self, board):
        """Evaluate material balance on the board"""
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.25,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0  # King's value isn't counted in material
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def _evaluate_piece_position(self, board):
        """Evaluate piece positions using piece-square tables"""
        # Implementation details hidden for LLM to discover
        return 0
    
    def _is_endgame(self, board):
        """Determine if the position is in the endgame phase"""
        # Implementation details hidden for LLM to discover
        return False
    
    def _evaluate_mobility(self, board):
        """Evaluate piece mobility (number of legal moves)"""
        # Implementation details hidden for LLM to discover
        return 0
    
    def _evaluate_king_safety(self, board):
        """Evaluate king safety"""
        # Implementation details hidden for LLM to discover
        return 0
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure (doubled, isolated, passed pawns)"""
        # Implementation details hidden for LLM to discover
        return 0
    
    @bundle(trainable=True)
    def evaluate_position(self, obs):
        '''
        Evaluate the current chess position and return a dictionary of move evaluations.
        
        This function should analyze the current chess position using principles such as:
        1. Material value (piece count and value)
           - Pawns = 1, Knights = 3, Bishops = 3.25, Rooks = 5, Queens = 9
           - Consider the total material balance between white and black
        
        2. Piece development and mobility
           - Pieces should control central squares
           - Knights and bishops should be developed early
           - Pieces should have many available moves
           - Consider using piece-square tables to evaluate piece positioning
        
        3. King safety
           - The king should be castled in the opening and middlegame
           - Pawns in front of the castled king provide protection
           - Exposed kings are vulnerable to attacks
        
        4. Pawn structure
           - Doubled pawns (two pawns on the same file) are generally weak
           - Isolated pawns (no friendly pawns on adjacent files) are vulnerable
           - Passed pawns (no enemy pawns can stop them from promoting) are strong
           - Connected pawns support each other
        
        5. Center control
           - The central squares (e4, d4, e5, d5) are strategically important
           - Controlling the center provides mobility and attacking opportunities
        
        You might want to create helper functions to evaluate each of these aspects separately,
        then combine them with appropriate weights.
        
        Args:
            obs (dict): A dictionary containing the current chess board state with keys:
                - 'board_fen': FEN string representation of the board
                - 'legal_moves': List of legal moves in UCI format
                - 'turn': Current player's turn ('white' or 'black')
                - 'is_check': Boolean indicating if the current player is in check
                - 'piece_map': Dictionary mapping square names to piece symbols
                - 'white_pieces': List of squares with white pieces
                - 'black_pieces': List of squares with black pieces
                - 'last_move': The last move made in UCI format
        
        Returns:
            dict: A dictionary mapping moves (in UCI format) to their evaluation scores
        '''
        # Handle the case where obs might be a MessageNode
        if hasattr(obs, 'data'):
            obs_data = obs.data
        else:
            obs_data = obs
            
        # Create a chess board from FEN
        board_fen = obs_data.get('board_fen') if isinstance(obs_data, dict) else None
        if board_fen:
            board = chess.Board(board_fen)
        else:
            # Default to starting position if no FEN is provided
            board = chess.Board()
            
        legal_moves = obs_data.get('legal_moves', []) if isinstance(obs_data, dict) else []
        if not legal_moves and board:
            # If legal_moves not provided but we have a board, get them from the board
            legal_moves = [move.uci() for move in board.legal_moves]
            
        move_scores = {}
        
        # For each legal move, assign a basic score
        # This is a starting point - the LLM will improve this evaluation function
        for move in legal_moves:
            # You should implement a more sophisticated evaluation here
            # Consider material, piece position, king safety, pawn structure, etc.
            move_scores[move] = 0  # Default neutral score
        
        return move_scores
    
    @bundle(trainable=True)
    def search_position(self, board, depth, alpha, beta, maximizing_player):
        '''
        Perform a search of the chess position to find the best move.
        
        This function should implement a minimax search algorithm with alpha-beta pruning.
        The minimax algorithm works by:
        1. Exploring the game tree to a certain depth
        2. Evaluating the leaf nodes using a position evaluation function
        3. Backing up the values to determine the best move
        
        Alpha-beta pruning is an optimization that reduces the number of nodes explored:
        - Alpha is the best value that the maximizing player can guarantee
        - Beta is the best value that the minimizing player can guarantee
        - If alpha >= beta, we can prune (stop exploring) the current branch
        
        The search should consider:
        - Material exchanges
        - Tactical opportunities (captures, checks, threats)
        - Position evaluation at leaf nodes
        
        Args:
            board (chess.Board): The current chess board
            depth (int): How many moves ahead to search
            alpha (float): Alpha value for alpha-beta pruning
            beta (float): Beta value for alpha-beta pruning
            maximizing_player (bool): Whether the current player is maximizing (True) or minimizing (False)
        
        Returns:
            float: The evaluation score of the position after the search
        '''
        # Base case: if we've reached the maximum depth or the game is over
        if depth == 0 or board.is_game_over():
            # Evaluate the leaf node
            # You should implement a position evaluation function here
            # Consider material, piece position, king safety, etc.
            return 0
        
        # Recursive case: explore the game tree
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                # Make the move
                board.push(move)
                # Recursively evaluate the position
                eval = self.search_position(board, depth - 1, alpha, beta, False)
                # Undo the move
                board.pop()
                # Update the maximum evaluation
                max_eval = max(max_eval, eval)
                # Update alpha
                alpha = max(alpha, eval)
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                # Make the move
                board.push(move)
                # Recursively evaluate the position
                eval = self.search_position(board, depth - 1, alpha, beta, True)
                # Undo the move
                board.pop()
                # Update the minimum evaluation
                min_eval = min(min_eval, eval)
                # Update beta
                beta = min(beta, eval)
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            return min_eval
    
    @bundle(trainable=True)
    def select_move(self, position_evaluation, obs):
        '''
        Select the best chess move based on position evaluation.
        
        This function should take the evaluation of the current position and select
        the best move according to strategic considerations. It can incorporate factors
        beyond raw evaluation scores, such as:
        
        1. Opening principles (if in the opening phase):
           - Develop knights and bishops early
           - Control the center with pawns or pieces
           - Castle early to protect the king
           - Avoid moving the same piece multiple times
           - Connect the rooks
        
        2. Middlegame strategy:
           - Look for tactical opportunities (captures, checks, threats)
           - Improve piece positioning
           - Create and exploit weaknesses in the opponent's position
           - Coordinate pieces for an attack
        
        3. Endgame techniques:
           - Activate the king
           - Push passed pawns
           - Create passed pawns
           - Cut off the opponent's king
        
        4. Special considerations:
           - Avoid moving into checks or captures
           - Consider piece exchanges when ahead in material
           - Avoid piece exchanges when behind in material
        
        Args:
            position_evaluation (dict): Dictionary mapping moves to their evaluation scores
            obs (dict): A dictionary containing the current chess board state
        
        Returns:
            str: The selected move in UCI format (e.g., 'e2e4', 'g1f3')
        '''
        # Handle the case where obs might be a MessageNode
        if hasattr(obs, 'data'):
            obs_data = obs.data
        else:
            obs_data = obs
            
        legal_moves = obs_data.get('legal_moves', []) if isinstance(obs_data, dict) else []
        if not legal_moves:
            return None
        
        # Get the board FEN to check if we're in the opening
        board_fen = obs_data.get('board_fen') if isinstance(obs_data, dict) else None
        is_opening = False
        if board_fen:
            board = chess.Board(board_fen)
            # Consider it opening if fewer than 10 moves have been made
            is_opening = board.fullmove_number < 10
        
        # Special handling for opening moves
        if is_opening and board_fen and board_fen.split()[0] == chess.STARTING_FEN.split()[0]:
            # We're in the starting position
            # Prioritize good opening moves
            good_openings = ['e2e4', 'd2d4', 'g1f3', 'c2c4']
            for move in good_openings:
                if move in legal_moves:
                    return move
        
        # Clean the position evaluation to handle MessageNode objects
        clean_evaluation = {}
        if position_evaluation:
            for move_key, score in position_evaluation.items():
                clean_key = extract_node_data(move_key)
                clean_score = extract_node_data(score)
                clean_evaluation[clean_key] = clean_score
        
        # If we have evaluations, use them to select the best move
        if clean_evaluation and len(clean_evaluation) > 0:
            # Find the move with the highest evaluation score
            best_move = max(clean_evaluation.items(), key=lambda x: x[1])[0]
            return best_move
        
        # Fallback: return the first legal move
        return legal_moves[0]

def rollout(env, horizon, policy):
    """Rollout a policy in the chess environment for horizon steps or until game over."""
    try:
        obs, _ = env.reset()
        trajectory = dict(observations=[], actions=[], rewards=[], terminations=[], truncations=[], infos=[], steps=0)
        trajectory["observations"].append(obs)
        
        for _ in range(horizon):
            error = None
            try:
                action = policy(obs)
                next_obs, reward, termination, truncation, info = env.step(action)
            except trace.ExecutionError as e:
                error = e
                reward = np.nan
                termination = True
                truncation = False
                info = {}
            
            if error is None:
                trajectory["observations"].append(next_obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminations"].append(termination)
                trajectory["truncations"].append(truncation)
                trajectory["infos"].append(info)
                trajectory["steps"] += 1
                if termination or truncation:
                    break
                obs = next_obs
    finally:
        env.close()
    
    return trajectory, error

def test_policy(policy, num_games=5, max_moves=100):
    """Test the policy by playing multiple games against Stockfish."""
    logger.info("Evaluating chess policy")
    env = ChessTracedEnv()
    results = []
    
    for game in range(num_games):
        obs, _ = env.reset()
        game_reward = 0
        moves_made = 0
        
        for _ in range(max_moves):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            game_reward = reward  # Final reward is what matters in chess
            moves_made += 1
            
            if terminated or truncated:
                break
        
        results.append({
            "game": game + 1,
            "result": "Win" if game_reward > 0.9 else "Loss" if game_reward < -0.9 else "Draw",
            "reward": game_reward,
            "moves": moves_made
        })
    
    # Calculate statistics
    wins = sum(1 for r in results if r["result"] == "Win")
    draws = sum(1 for r in results if r["result"] == "Draw")
    losses = sum(1 for r in results if r["result"] == "Loss")
    avg_reward = np.mean([r["reward"] for r in results])
    avg_moves = np.mean([r["moves"] for r in results])
    
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / num_games,
        "avg_reward": avg_reward,
        "avg_moves": avg_moves,
        "results": results
    }

def visualize_game(trajectory, output_dir=None):
    """
    Visualize a chess game from a trajectory.
    
    Args:
        trajectory (dict): The trajectory from a rollout
        output_dir (str, optional): Directory to save visualizations. If None, just display.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a chess board
    board = chess.Board()
    
    # Get the moves from the trajectory
    moves = [action for action in trajectory["actions"] if action is not None]
    
    # Create a game object
    game = chess.pgn.Game()
    game.headers["Event"] = "Trace Chess Agent vs Stockfish"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "Trace Chess Agent"
    game.headers["Black"] = "Stockfish"
    
    # Add the moves to the game
    node = game
    for i, move_uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                node = node.add_variation(move)
                
                # Save or display the board after each move
                svg_str = chess.svg.board(board=board, lastmove=move, size=400)
                
                if output_dir:
                    svg_path = output_dir / f"move_{i+1:03d}.svg"
                    with open(svg_path, "w") as f:
                        f.write(svg_str)
                else:
                    display(SVG(svg_str))
                    print(f"Move {i+1}: {move_uci}")
                    time.sleep(0.5)  # Pause to see the move
        except Exception as e:
            print(f"Error on move {i+1} ({move_uci}): {e}")
    
    # Save the PGN file if output_dir is provided
    if output_dir:
        pgn_path = output_dir / "game.pgn"
        with open(pgn_path, "w") as f:
            f.write(str(game))
    
    return game

def debug_policy_decision(policy, obs, output_dir=None):
    """
    Debug a single policy decision by showing the evaluation and selected move.
    
    Args:
        policy (ChessPolicy): The policy to debug
        obs (dict): The observation to evaluate
        output_dir (str, optional): Directory to save debug info. If None, just display.
    """
    # Handle the case where obs might be a MessageNode
    if hasattr(obs, 'data'):
        obs_data = obs.data
    else:
        obs_data = obs
        
    # Enable debug mode
    DebugHelper.enable(True)
    
    # Make a decision
    move = policy(obs)
    move_str = extract_node_data(move)  # Extract data from node if needed
    
    # Get debug info
    debug_info = DebugHelper.get_info()
    
    # Create a board from the FEN
    board_fen = obs_data.get('board_fen') if isinstance(obs_data, dict) else None
    if board_fen:
        board = chess.Board(board_fen)
    else:
        # Default to starting position if no FEN is provided
        board = chess.Board()
    
    # Print the board
    print(board)
    print(f"Selected move: {move_str}")
    
    # Print the evaluation scores
    print("\nMove evaluations:")
    if debug_info.get('position_evaluation'):
        # Extract data from position_evaluation if it contains nodes
        position_evaluation = debug_info['position_evaluation']
        clean_evaluation = {}
        
        # Process the evaluation dictionary to extract data from any nodes
        for move_key, score in position_evaluation.items():
            clean_key = extract_node_data(move_key)
            clean_score = extract_node_data(score)
            clean_evaluation[clean_key] = clean_score
        
        # Sort moves by evaluation score (descending)
        sorted_moves = sorted(
            clean_evaluation.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for move_uci, score in sorted_moves:
            print(f"{move_uci}: {score:.4f}")
    
    # Visualize the board with the selected move
    try:
        if move_str:
            move_obj = chess.Move.from_uci(move_str)
            svg_str = chess.svg.board(
                board=board,
                arrows=[(move_obj.from_square, move_obj.to_square)],
                size=400
            )
        else:
            svg_str = chess.svg.board(board=board, size=400)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Save the SVG
            svg_path = output_dir / "decision.svg"
            with open(svg_path, "w") as f:
                f.write(svg_str)
            
            # Save a text representation for command-line viewing
            txt_path = output_dir / "decision.txt"
            with open(txt_path, "w") as f:
                f.write(str(board) + f"\n\nSelected move: {move_str}")
                f.write("\n\nMove evaluations:\n")
                for move_uci, score in sorted_moves:
                    f.write(f"{move_uci}: {score:.4f}\n")
            
            # Save the debug info as JSON
            # Convert any node objects to their data before serializing
            clean_debug_info = {}
            for key, value in debug_info.items():
                if key == 'position_evaluation':
                    clean_debug_info[key] = clean_evaluation
                else:
                    clean_debug_info[key] = extract_node_data(value)
            
            json_path = output_dir / "debug_info.json"
            with open(json_path, "w") as f:
                json.dump(clean_debug_info, f, indent=2)
        elif HAS_IPYTHON:
            display(SVG(svg_str))
        else:
            print("\nBoard visualization saved to output directory (IPython not available)")
    except Exception as e:
        print(f"Error visualizing move: {e}")
    
    # Disable debug mode
    DebugHelper.enable(False)
    
    return debug_info

def optimize_policy(
    horizon=100,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    logger=None,
    visualize=False,
    debug_interval=5
):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    policy = ChessPolicy()
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    env = ChessTracedEnv()
    
    perf_csv_filename = log_dir / f"chess_perf_{timestamp}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}.csv"
    trace_ckpt_dir = base_trace_ckpt_dir / f"chess_{timestamp}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}"
    trace_ckpt_dir.mkdir(exist_ok=True)
    
    # Create visualization directory if needed
    if visualize:
        vis_dir = Path("visualizations") / f"chess_{timestamp}"
        vis_dir.mkdir(exist_ok=True, parents=True)
    else:
        vis_dir = None
    
    try:
        rewards = []
        optimization_data = []
        logger.info("Chess Policy Optimization Starts")
        
        for i in range(n_optimization_steps):
            env.init()
            traj, error = rollout(env, horizon, policy)

            # Visualize the game if requested
            if visualize and error is None:
                game_vis_dir = vis_dir / f"iteration_{i:03d}" if vis_dir else None
                visualize_game(traj, output_dir=game_vis_dir)
            
            # Debug the policy at regular intervals
            if i % debug_interval == 0 and error is None and len(traj["observations"]) > 0:
                debug_dir = vis_dir / f"debug_iteration_{i:03d}" if vis_dir else None
                debug_policy_decision(policy, traj["observations"][0], output_dir=debug_dir)

            if error is None:
                # Test the policy more thoroughly
                test_results = test_policy(policy, num_games=5)
                
                feedback = (f"Game ended after {traj['steps']} moves with final reward: {traj['rewards'][-1]:.2f}. "
                           f"Win rate: {test_results['win_rate']:.2f}, "
                           f"Wins: {test_results['wins']}, Draws: {test_results['draws']}, Losses: {test_results['losses']}")
                
                if test_results['win_rate'] > 0.8:
                    feedback += "\nExcellent! Your chess strategy is very effective against Stockfish."
                elif test_results['win_rate'] > 0.5:
                    feedback += "\nGood job! You're winning more games than you're losing."
                elif test_results['win_rate'] > 0.2:
                    feedback += "\nYou're making progress, but still need to improve your strategy."
                else:
                    feedback += "\nYour strategy needs significant improvement to compete with Stockfish, try using a different strategy."
                
                target = traj['observations'][-1]
                rewards.append(traj['rewards'][-1])
                
                optimization_data.append({
                    "Optimization Step": i,
                    "Win Rate": test_results['win_rate'],
                    "Avg Reward": test_results['avg_reward'],
                    "Avg Moves": test_results['avg_moves'],
                    "Wins": test_results['wins'],
                    "Draws": test_results['draws'],
                    "Losses": test_results['losses']
                })
                
                df = pd.DataFrame(optimization_data)
                df.to_csv(perf_csv_filename, index=False)
                
                # Save the game as PGN
                if vis_dir:
                    env.save_game_pgn(vis_dir / f"game_iteration_{i:03d}.pgn")
            else:
                feedback = error.exception_node.create_feedback()
                target = error.exception_node
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}")
            policy.save(os.path.join(trace_ckpt_dir, f"{i}.pkl"))

            instruction = "In chess, you are playing as White against Stockfish (Black). "
            instruction += "The goal is to checkmate the opponent's king while protecting your own. "
            instruction += "You need to develop a strategy that considers piece development, material advantage, king safety, and tactical opportunities. "
            instruction += "Analyze the trace to understand why your moves succeed or fail, and optimize your code to make better chess decisions. "
            instruction += "Remember key chess principles: control the center, develop pieces efficiently, castle early for king safety, and look for tactical opportunities."
            
            optimizer.objective = optimizer.default_objective + instruction 
            
            optimizer.zero_feedback()
            optimizer.backward(target, feedback, visualize=True)
            logger.info(optimizer.problem_instance(optimizer.summarize()))
            
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                optimizer.step(verbose=verbose)
                llm_output = stdout_buffer.getvalue()
                if llm_output:
                    logger.info(f"LLM response:\n {llm_output}")
            
            logger.info(f"Iteration: {i}, Feedback: {feedback}")
            
            # Plot performance metrics if visualization is enabled
            if visualize and len(optimization_data) > 1:
                plt.figure(figsize=(12, 8))
                
                # Plot win rate
                plt.subplot(2, 2, 1)
                plt.plot([d["Optimization Step"] for d in optimization_data], 
                         [d["Win Rate"] for d in optimization_data], 'b-o')
                plt.title("Win Rate vs. Optimization Step")
                plt.xlabel("Optimization Step")
                plt.ylabel("Win Rate")
                plt.grid(True)
                
                # Plot average reward
                plt.subplot(2, 2, 2)
                plt.plot([d["Optimization Step"] for d in optimization_data], 
                         [d["Avg Reward"] for d in optimization_data], 'g-o')
                plt.title("Average Reward vs. Optimization Step")
                plt.xlabel("Optimization Step")
                plt.ylabel("Average Reward")
                plt.grid(True)
                
                # Plot wins, draws, losses
                plt.subplot(2, 2, 3)
                plt.plot([d["Optimization Step"] for d in optimization_data], 
                         [d["Wins"] for d in optimization_data], 'g-o', label="Wins")
                plt.plot([d["Optimization Step"] for d in optimization_data], 
                         [d["Draws"] for d in optimization_data], 'b-o', label="Draws")
                plt.plot([d["Optimization Step"] for d in optimization_data], 
                         [d["Losses"] for d in optimization_data], 'r-o', label="Losses")
                plt.title("Game Outcomes vs. Optimization Step")
                plt.xlabel("Optimization Step")
                plt.ylabel("Count")
                plt.legend()
                plt.grid(True)
                
                # Plot average moves per game
                plt.subplot(2, 2, 4)
                plt.plot([d["Optimization Step"] for d in optimization_data], 
                         [d["Avg Moves"] for d in optimization_data], 'm-o')
                plt.title("Average Moves per Game vs. Optimization Step")
                plt.xlabel("Optimization Step")
                plt.ylabel("Average Moves")
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(vis_dir / f"performance_metrics_step_{i}.png")
                plt.close()
    finally:
        if env is not None:
            env.close()
    
    if rewards:
        logger.info(f"Final Average Reward: {sum(rewards) / len(rewards)}")
    return rewards

if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Set up file logging
    log_file = log_dir / f"chess_ai_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting Chess AI training...")
    rewards = optimize_policy(
        horizon=100,  # Chess games can be longer, but we'll limit for training
        n_optimization_steps=20,
        memory_size=5,
        verbose='output',
        logger=logger,
        visualize=True,  # Enable visualization
        debug_interval=2  # Debug every 2 iterations
    )
    logger.info("Training completed.")
