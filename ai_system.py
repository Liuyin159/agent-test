"""
五子棋AI难度系统 - 人类对战AI版本
实现4个难度级别：Beginner, Intermediate, Advance, Expert
用于人类与AI对战
"""

from __future__ import annotations
import random
import math
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np

class AIDifficulty(Enum):
    """AI难度级别"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCE = "advance"
    EXPERT = "expert"
    
class HumanVsAI:
    """人类与AI对战系统"""
    
    def __init__(self):
        self.current_difficulty = AIDifficulty.INTERMEDIATE
        self.ai_color = 2  # 默认AI执白棋（后手）
        self.human_color = 1  # 人类执黑棋（先手）
        self.board_size = 15
        self.max_depth = 2  # 添加默认的 max_depth 属性
        
    def set_difficulty(self, difficulty: str) -> str:
        """设置AI难度"""
        try:
            self.current_difficulty = AIDifficulty(difficulty.lower())
            return f"AI难度已设置为: {self.current_difficulty.value}"
        except ValueError:
            return f"无效难度: {difficulty}，可用: beginner, intermediate, advance, expert"
    
    def set_ai_color(self, color: str) -> str:
        """设置AI执棋颜色"""
        if color.lower() in ['black', '黑棋', '先手']:
            self.ai_color = 1
            self.human_color = 2
            return "AI执黑棋（先手）"
        else:
            self.ai_color = 2  
            self.human_color = 1
            return "AI执白棋（后手）"
    
    def get_ai_move(self, board: List[List[int]]) -> Tuple[int, int]:
        """获取AI走法"""
        ai = GomokuAI(self.current_difficulty)
        return ai.get_move(board, self.ai_color)
    
    def is_ai_turn(self, current_player: int) -> bool:
        """判断是否是AI的回合"""
        return current_player == self.ai_color

class GomokuAI:
    """五子棋AI核心类"""
    
    def __init__(self, difficulty: AIDifficulty = AIDifficulty.INTERMEDIATE):
        self.difficulty = difficulty
        self.board_size = 15
        self.max_depth = self._get_max_depth()
        
    def _get_max_depth(self) -> int:
        """根据难度设置搜索深度"""
        depth_map = {
            AIDifficulty.BEGINNER: 0,
            AIDifficulty.INTERMEDIATE: 2,
            AIDifficulty.ADVANCE: 4,
            AIDifficulty.EXPERT: 6
        }
        return depth_map.get(self.difficulty, 2)
    
    def get_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """获取AI的走子位置"""
        if self.difficulty == AIDifficulty.BEGINNER:
            return self._beginner_move(board, current_player)
        elif self.difficulty == AIDifficulty.INTERMEDIATE:
            return self._intermediate_move(board, current_player)
        elif self.difficulty == AIDifficulty.ADVANCE:
            return self._advance_move(board, current_player)
        else:  # EXPERT
            return self._expert_move(board, current_player)

    def _beginner_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """初级AI：随机走法，偶尔防守"""
        empty_positions = self._get_empty_positions(board)
        
        # 30%概率进行简单防守
        if random.random() < 0.3:
            defensive_move = self._find_defensive_move(board, current_player)
            if defensive_move:
                return defensive_move
        
        # 否则随机走
        return random.choice(empty_positions)
    
    def _intermediate_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """中级AI：使用Minimax搜索，深度2"""
        best_score = -math.inf
        best_move = None
        
        # 获取候选位置（只考虑有邻居的位置）
        candidate_moves = self._get_candidate_moves(board)
        
        for move in candidate_moves:
            row, col = move
            board[row][col] = current_player
            
            score = self._minimax(board, 2, False, current_player, -math.inf, math.inf)
            
            board[row][col] = 0  # 撤销
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or candidate_moves[0]
    
    def _advance_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """高级AI：使用Minimax+Alpha-Beta剪枝，深度4"""
        best_score = -math.inf
        best_move = None
        
        candidate_moves = self._get_candidate_moves(board)
        
        for move in candidate_moves:
            row, col = move
            board[row][col] = current_player
            
            score = self._minimax(board, 4, False, current_player, -math.inf, math.inf)
            
            board[row][col] = 0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or candidate_moves[0]
    
    def _expert_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """专家级AI：使用增强的搜索和模式识别"""
        # 首先检查是否有立即获胜的机会
        winning_move = self._find_winning_move(board, current_player)
        if winning_move:
            return winning_move
        
        # 检查是否需要防守（阻止对手获胜）
        defensive_move = self._find_winning_move(board, 3 - current_player)
        if defensive_move:
            return defensive_move
        
        # 使用增强的Minimax搜索
        return self._enhanced_minimax_move(board, current_player)
    
    def _minimax(self, board: List[List[int]], depth: int, is_maximizing: bool, 
                 player: int, alpha: float, beta: float) -> float:
        """Minimax算法 with Alpha-Beta剪枝"""
        if depth == 0 or self._is_game_over(board):
            return self._evaluate_board(board, player)
        
        if is_maximizing:
            max_eval = -math.inf
            for move in self._get_candidate_moves(board):
                row, col = move
                board[row][col] = player
                eval = self._minimax(board, depth - 1, False, player, alpha, beta)
                board[row][col] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            opponent = 3 - player
            for move in self._get_candidate_moves(board):
                row, col = move
                board[row][col] = opponent
                eval = self._minimax(board, depth - 1, True, player, alpha, beta)
                board[row][col] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def _enhanced_minimax_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """增强的Minimax搜索（专家级使用）"""
        best_score = -math.inf
        best_move = None
        
        candidate_moves = self._get_priority_moves(board, current_player)
        
        for move in candidate_moves:
            row, col = move
            board[row][col] = current_player
            
            score = self._minimax(board, 6, False, current_player, -math.inf, math.inf)
            # 添加位置权重
            score += self._position_weight(row, col)
            
            board[row][col] = 0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move or candidate_moves[0]
    
    def _get_empty_positions(self, board: List[List[int]]) -> List[Tuple[int, int]]:
        """获取所有空位置"""
        positions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    positions.append((i, j))
        return positions
    
    def _get_candidate_moves(self, board: List[List[int]]) -> List[Tuple[int, int]]:
        """获取候选走法（只考虑有邻居的位置）"""
        candidates = []
        empty_positions = self._get_empty_positions(board)
        
        for pos in empty_positions:
            if self._has_neighbors(board, pos):
                candidates.append(pos)
        
        # 如果没有候选位置，返回中心附近的空位
        if not candidates and empty_positions:
            center = self.board_size // 2
            candidates = [pos for pos in empty_positions 
                         if abs(pos[0]-center) <= 2 and abs(pos[1]-center) <= 2]
        
        return candidates or empty_positions
    
    def _get_priority_moves(self, board: List[List[int]], player: int) -> List[Tuple[int, int]]:
        """获取优先走法（专家级使用）"""
        candidates = self._get_candidate_moves(board)
        
        # 根据威胁级别排序
        scored_moves = []
        for move in candidates:
            score = self._evaluate_move_threat(board, move, player)
            scored_moves.append((score, move))
        
        # 按分数降序排序
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        # 返回前10个候选位置
        return [move for _, move in scored_moves[:10]]
    
    def _has_neighbors(self, board: List[List[int]], pos: Tuple[int, int]) -> bool:
        """检查位置是否有邻居棋子"""
        row, col = pos
        for i in range(max(0, row-2), min(self.board_size, row+3)):
            for j in range(max(0, col-2), min(self.board_size, col+3)):
                if (i != row or j != col) and board[i][j] != 0:
                    return True
        return False
    
    def _find_defensive_move(self, board: List[List[int]], current_player: int) -> Optional[Tuple[int, int]]:
        """寻找防守走法"""
        opponent = 3 - current_player
        
        # 检查对手是否有活四
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    # 模拟对手走子
                    board[i][j] = opponent
                    if self._check_five_in_row(board, i, j, opponent):
                        board[i][j] = 0
                        return (i, j)  # 必须防守
                    board[i][j] = 0
        
        return None
    
    def _find_winning_move(self, board: List[List[int]], player: int) -> Optional[Tuple[int, int]]:
        """寻找立即获胜的走法"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    board[i][j] = player
                    if self._check_five_in_row(board, i, j, player):
                        board[i][j] = 0
                        return (i, j)
                    board[i][j] = 0
        return None
    
    def _evaluate_board(self, board: List[List[int]], player: int) -> float:
        """评估棋盘局面"""
        score = 0
        opponent = 3 - player
        
        # 检查各种棋型
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    score += self._evaluate_position(board, i, j, player)
                elif board[i][j] == opponent:
                    score -= self._evaluate_position(board, i, j, opponent)
        
        return score
    
    def _evaluate_position(self, board: List[List[int]], row: int, col: int, player: int) -> float:
        """评估单个位置的价值"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            score += self._evaluate_direction(board, row, col, dr, dc, player)
        
        return score
    
    def _evaluate_direction(self, board: List[List[int]], row: int, col: int, 
                          dr: int, dc: int, player: int) -> float:
        """评估特定方向的价值"""
        # 棋型分数
        five = 100000
        live_four = 10000
        dead_four = 1000
        live_three = 1000
        dead_three = 100
        live_two = 100
        dead_two = 10
        
        # 分析棋型
        count = self._count_consecutive(board, row, col, dr, dc, player)
        
        if count >= 5:
            return five
        elif count == 4:
            return live_four
        elif count == 3:
            return live_three
        elif count == 2:
            return live_two
        
        return 0
    
    def _evaluate_move_threat(self, board: List[List[int]], move: Tuple[int, int], 
                            player: int) -> float:
        """评估走法的威胁级别"""
        row, col = move
        threat_score = 0
        
        # 模拟走子
        board[row][col] = player
        
        # 检查是否形成威胁
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = self._count_consecutive(board, row, col, dr, dc, player)
            if count >= 4:
                threat_score += 1000
            elif count == 3:
                threat_score += 100
            elif count == 2:
                threat_score += 10
        
        # 撤销走子
        board[row][col] = 0
        
        return threat_score
    
    def _count_consecutive(self, board: List[List[int]], row: int, col: int,
                         dr: int, dc: int, player: int) -> int:
        """统计连续棋子数量"""
        count = 1
        
        # 正向
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        
        # 反向
        r, c = row - dr, col - dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        
        return count
    
    def _position_weight(self, row: int, col: int) -> float:
        """位置权重（中心位置更有价值）"""
        center = self.board_size // 2
        distance = math.sqrt((row - center)**2 + (col - center)**2)
        return (center - distance) * 10
    
    def _check_five_in_row(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否形成五连"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            # 正向
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
                count += 1
                r += dr
                c += dc
            
            # 反向
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def _is_game_over(self, board: List[List[int]]) -> bool:
        """检查游戏是否结束"""
        # 简化实现：检查是否有五连
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] != 0:
                    if self._check_five_in_row(board, i, j, board[i][j]):
                        return True
        return False


# 全局人类对战AI实例
_human_vs_ai_instance: Optional[HumanVsAI] = None

def get_human_vs_ai() -> HumanVsAI:
    """获取人类对战AI实例"""
    global _human_vs_ai_instance
    if _human_vs_ai_instance is None:
        _human_vs_ai_instance = HumanVsAI()
    return _human_vs_ai_instance