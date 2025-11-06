"""
五子棋局面评估工具
提供局面分析和最佳走法建议
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from .gomoku_game import get_current_board, Player


def evaluate_position() -> str:
    """评估当前局面
    
    Returns:
        局面评估结果
    """
    board = get_current_board()
    
    if board.game_over:
        return "游戏已结束，无需评估"
    
    # 评估当前玩家和对手的威胁
    current_player = board.current_player
    opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
    
    # 检查对手的威胁（需要防守的点）
    opponent_threats = _find_threats(board, opponent)
    # 检查己方的机会（可以形成威胁的点）
    my_opportunities = _find_threats(board, current_player)
    
    result = "局面评估:\n"
    result += f"当前玩家: {'黑棋' if current_player == Player.BLACK else '白棋'}\n"
    result += f"已走步数: {len(board.move_history)}\n"
    
    if opponent_threats:
        result += f"\n⚠️ 发现 {len(opponent_threats)} 个威胁点（对方可能成五连，需要防守）\n"
    else:
        result += "\n✓ 未发现对方威胁\n"
        
    if my_opportunities:
        result += f"✓ 发现 {len(my_opportunities)} 个机会点（己方可能成五连）\n"
    else:
        result += "未发现己方机会\n"
        
    return result


def _find_threats(board, player: Player) -> List[Tuple[int, int]]:
    """查找威胁位置（即将成五连的位置）"""
    threats = []
    size = board.size
    
    # 检查每个空位置，如果放子后能形成五连
    for i in range(size):
        for j in range(size):
            if board.board[i][j] == Player.EMPTY:
                # 临时放置棋子检查
                board.board[i][j] = player
                if _check_pattern_at(board, i, j, player, 4):  # 四连
                    threats.append((i, j))
                board.board[i][j] = Player.EMPTY
                
    return threats


def _check_pattern_at(board, row: int, col: int, player: Player, target_count: int) -> bool:
    """检查指定位置是否形成指定数量的连续棋子"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        count = 1
        
        # 正向
        for i in range(1, 5):
            nr, nc = row + dr * i, col + dc * i
            if 0 <= nr < board.size and 0 <= nc < board.size and board.board[nr][nc] == player:
                count += 1
            else:
                break
                
        # 反向
        for i in range(1, 5):
            nr, nc = row - dr * i, col - dc * i
            if 0 <= nr < board.size and 0 <= nc < board.size and board.board[nr][nc] == player:
                count += 1
            else:
                break
                
        if count >= target_count:
            return True
            
    return False


def suggest_moves(max_suggestions: int = 5) -> str:
    """建议最佳走法
    
    Args:
        max_suggestions: 最多建议数量
        
    Returns:
        走法建议
    """
    board = get_current_board()
    
    if board.game_over:
        return "游戏已结束，无法建议走法"
    
    suggestions = []
    
    # 1. 检查是否有必胜走法
    winning_moves = _find_winning_moves(board)
    if winning_moves:
        return f"发现必胜走法: {winning_moves[0]} (建议立即走此步！)"
    
    # 2. 检查是否需要防守
    opponent = Player.WHITE if board.current_player == Player.BLACK else Player.BLACK
    blocking_moves = _find_blocking_moves(board, opponent)
    if blocking_moves:
        suggestions.extend([(move, "防守", 10) for move in blocking_moves[:3]])
    
    # 3. 寻找攻击机会
    attacking_moves = _find_attacking_moves(board, board.current_player)
    if attacking_moves:
        suggestions.extend([(move, "攻击", 8) for move in attacking_moves[:3]])
    
    # 4. 中心位置优先
    center_moves = _find_center_moves(board)
    if center_moves:
        suggestions.extend([(move, "中心", 5) for move in center_moves[:2]])
    
    if not suggestions:
        # 如果没有特殊建议，返回中心附近的位置
        center = board.size // 2
        suggestions = [
            ((center, center), "中心", 3),
            ((center + 1, center), "中心附近", 2),
            ((center, center + 1), "中心附近", 2),
        ]
    
    # 去重并排序
    unique_suggestions = {}
    for move, reason, score in suggestions:
        if move not in unique_suggestions or unique_suggestions[move][1] < score:
            unique_suggestions[move] = (reason, score)
    
    sorted_moves = sorted(
        unique_suggestions.items(),
        key=lambda x: x[1][1],
        reverse=True
    )[:max_suggestions]
    
    result = f"走法建议（当前玩家: {'黑棋' if board.current_player == Player.BLACK else '白棋'}）:\n"
    for idx, (move, (reason, score)) in enumerate(sorted_moves, 1):
        result += f"{idx}. 位置 {move}: {reason} (优先级: {score})\n"
        
    return result.strip()


def _find_winning_moves(board) -> List[Tuple[int, int]]:
    """查找必胜走法"""
    winning_moves = []
    size = board.size
    
    for i in range(size):
        for j in range(size):
            if board.board[i][j] == Player.EMPTY:
                board.board[i][j] = board.current_player
                if board._check_win(i, j):
                    winning_moves.append((i, j))
                board.board[i][j] = Player.EMPTY
                
    return winning_moves


def _find_blocking_moves(board, opponent: Player) -> List[Tuple[int, int]]:
    """查找防守走法（阻止对方获胜）"""
    blocking_moves = []
    size = board.size
    
    for i in range(size):
        for j in range(size):
            if board.board[i][j] == Player.EMPTY:
                # 检查如果对方走这里是否会获胜
                board.board[i][j] = opponent
                if board._check_win(i, j):
                    blocking_moves.append((i, j))
                board.board[i][j] = Player.EMPTY
                
    return blocking_moves


def _find_attacking_moves(board, player: Player) -> List[Tuple[int, int]]:
    """查找攻击走法（形成威胁）"""
    attacking_moves = []
    size = board.size
    
    for i in range(size):
        for j in range(size):
            if board.board[i][j] == Player.EMPTY:
                board.board[i][j] = player
                if _check_pattern_at(board, i, j, player, 3):  # 形成三连或四连
                    attacking_moves.append((i, j))
                board.board[i][j] = Player.EMPTY
                
    return attacking_moves


def _find_center_moves(board) -> List[Tuple[int, int]]:
    """查找中心区域可走位置"""
    center_moves = []
    center = board.size // 2
    size = board.size
    
    for i in range(max(0, center - 2), min(size, center + 3)):
        for j in range(max(0, center - 2), min(size, center + 3)):
            if board.board[i][j] == Player.EMPTY:
                center_moves.append((i, j))
                
    return center_moves


def analyze_pattern(pattern_type: str = "all") -> str:
    """分析棋局模式
    
    Args:
        pattern_type: 模式类型 (all, double_threes, fours)
        
    Returns:
        模式分析结果
    """
    board = get_current_board()
    
    if board.game_over:
        return "游戏已结束"
    
    result = f"模式分析（类型: {pattern_type}）:\n"
    
    # 查找双三、活四等模式
    player = board.current_player
    
    # 简单的模式检测
    threats = _find_threats(board, player)
    result += f"当前玩家威胁点: {len(threats)} 个\n"
    
    opponent = Player.WHITE if player == Player.BLACK else Player.BLACK
    opponent_threats = _find_threats(board, opponent)
    result += f"对方威胁点: {len(opponent_threats)} 个\n"
    
    return result.strip()


def get_game_statistics() -> str:
    """获取当前游戏的详细统计信息"""
    from tools.gomoku_game import get_current_board
    
    board = get_current_board()
    
    if len(board.move_history) == 0:
        return "游戏尚未开始，暂无统计数据"
    
    stats = []
    stats.append("📊 棋局统计分析")
    stats.append("=" * 40)
    
    # 基础统计
    total_moves = len(board.move_history)
    black_moves = len([m for i, m in enumerate(board.move_history) if i % 2 == 0])
    white_moves = total_moves - black_moves
    
    stats.append(f"总走子数: {total_moves}")
    stats.append(f"黑棋走子: {black_moves} ({black_moves/total_moves*100:.1f}%)")
    stats.append(f"白棋走子: {white_moves} ({white_moves/total_moves*100:.1f}%)")
    
    # 走子分布分析
    stats.append("\n🎯 走子分布分析")
    center = board.size // 2
    regions = {
        "中心区域(3x3)": 0,
        "内圈区域(5x5)": 0,  
        "外圈区域": 0,
    }
    
    for move in board.move_history:
        row, col = move
        distance = max(abs(row - center), abs(col - center))
        if distance <= 1:
            regions["中心区域(3x3)"] += 1
        elif distance <= 2:
            regions["内圈区域(5x5)"] += 1
        else:
            regions["外圈区域"] += 1
    
    for region, count in regions.items():
        percentage = count / total_moves * 100
        stats.append(f"  {region}: {count}手 ({percentage:.1f}%)")
    
    # 时间线分析
    stats.append("\n⏰ 游戏阶段分析")
    if total_moves >= 20:
        opening_moves = min(8, total_moves)
        midgame_moves = min(12, total_moves - opening_moves)
        endgame_moves = total_moves - opening_moves - midgame_moves
        
        stats.append(f"  开局阶段: 前{opening_moves}手")
        stats.append(f"  中局阶段: 第{opening_moves+1}-{opening_moves+midgame_moves}手")
        if endgame_moves > 0:
            stats.append(f"  残局阶段: 最后{endgame_moves}手")
    else:
        stats.append("  游戏尚处于早期阶段")
    
    # 热点区域分析
    stats.append("\n🔥 热点区域分析")
    hot_spots = _analyze_hot_spots(board)
    if hot_spots:
        for i, (pos, count) in enumerate(hot_spots[:3], 1):
            stats.append(f"  热点{i}: {pos} (相关走子: {count}次)")
    
    # 胜负关键点分析
    stats.append("\n🎯 胜负关键点")
    if total_moves >= 5:
        last_5_moves = board.move_history[-5:]
        stats.append(f"  最后5手: {last_5_moves}")
    
    # 当前局面评估
    stats.append("\n📈 当前局面简评")
    if total_moves < 5:
        stats.append("  🟡 开局阶段 - 争夺中心控制权")
    elif total_moves < 15:
        stats.append("  🟢 中局前期 - 发展阵型，寻找机会")
    elif total_moves < 25:
        stats.append("  🟠 中局后期 - 激烈对抗，寻找突破口")
    else:
        stats.append("  🔴 残局阶段 - 精确计算，决胜时刻")
    
    # 游戏持续时间估算
    stats.append("\n⏱️ 游戏时长估算")
    if total_moves <= 10:
        stats.append("  预计游戏时长: 5-10分钟")
    elif total_moves <= 20:
        stats.append("  预计游戏时长: 10-20分钟") 
    else:
        stats.append("  预计游戏时长: 20-30分钟")
    
    return "\n".join(stats)

def _analyze_hot_spots(board) -> List[Tuple[Tuple[int, int], int]]:
    """分析棋盘上的热点区域"""
    from collections import defaultdict
    
    hot_spot_count = defaultdict(int)
    size = board.size
    
    # 分析每个棋子周围的热度
    for move in board.move_history:
        row, col = move
        # 统计周围3x3区域的热度
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size:
                    hot_spot_count[(nr, nc)] += 1
    
    # 按热度排序，排除已有棋子的位置
    sorted_spots = []
    for pos, count in hot_spot_count.items():
        if board.board[pos[0]][pos[1]] == 0:  # 只考虑空位
            sorted_spots.append((pos, count))
    
    return sorted(sorted_spots, key=lambda x: x[1], reverse=True)
