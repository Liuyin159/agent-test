"""
五子棋AI Agent构建模块
基于LangChain ReAct框架，构建自主下棋的智能体
"""

from __future__ import annotations
from typing import List
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.caches import BaseCache  # noqa: F401
from langchain_openai import ChatOpenAI
import httpx
from langchain.agents import create_react_agent, AgentExecutor

from config import load_config
from tools import (
    init_game,
    make_move,
    get_board_state,
    save_game,
    load_game,
    reset_game,
    download_gomoku_dataset,
    load_dataset,
    analyze_opening,
    evaluate_position,
    suggest_moves,
    analyze_pattern,
)

def _parse_and_make_move(pos_str: str) -> str:
    """解析位置字符串并执行走子"""
    try:
        pos_str = pos_str.strip().lower()
        
        # 处理 "row=7,col=7" 格式
        if 'row=' in pos_str and 'col=' in pos_str:
            import re
            row_match = re.search(r'row=(\d+)', pos_str)
            col_match = re.search(r'col=(\d+)', pos_str)
            if row_match and col_match:
                row = int(row_match.group(1))
                col = int(col_match.group(1))
            else:
                return "错误：格式应为 'row=7,col=7'"
        else:
            # 原有逻辑
            if ',' in pos_str:
                parts = pos_str.split(',')
            else:
                parts = pos_str.split()
                
            if len(parts) < 2:
                return "错误：需要提供行和列坐标，格式为 'row,col'，例如 '7,7'"
                
            row = int(parts[0].strip())
            col = int(parts[1].strip())
            
        return make_move(row, col)
    except ValueError as e:
        return f"错误：坐标格式无效，请使用 'row,col' 格式，例如 '7,7'。错误: {e}"
    except Exception as e:
        return f"走子失败: {e}"
    
def _parse_download_args(args: str) -> str:
    """解析下载参数"""
    try:
        args = args.strip()
        # 支持多种分隔符
        if ',' in args:
            parts = [p.strip() for p in args.split(',', 1)]
        elif ' ' in args:
            parts = args.split(maxsplit=1)
            if len(parts) == 1:
                parts.append('games')
        else:
            # 只有路径，默认games类型
            parts = [args, 'games']
            
        save_path = parts[0].strip().strip('"\'')
        dataset_type = parts[1].strip().strip('"\'').lower() if len(parts) > 1 else 'games'
        
        # 验证dataset_type
        if dataset_type not in ['games', 'openings']:
            dataset_type = 'games'
            
        return download_gomoku_dataset(save_path, dataset_type)
    except Exception as e:
        return f"下载失败: {e}"


def start_human_vs_ai(difficulty: str = "intermediate", ai_color: str = "white") -> str:
    """开始人类与AI对战
    
    Args:
        difficulty: AI难度 (beginner, intermediate, advance, expert)
        ai_color: AI执棋颜色 (black/white)
        
    Returns:
        对战开始信息
    """
    from ai_system import get_human_vs_ai
    from tools.gomoku_game import init_game
    
    # 初始化游戏
    init_game(15)
    
    human_vs_ai = get_human_vs_ai()
    
    # 设置难度
    result = human_vs_ai.set_difficulty(difficulty)
    result += "\n" + human_vs_ai.set_ai_color(ai_color)
    
    human_color = "黑棋" if human_vs_ai.human_color == 1 else "白棋"
    ai_color_str = "黑棋" if human_vs_ai.ai_color == 1 else "白棋"
    
    result += f"\n\n对战开始！"
    result += f"\n- 人类执{human_color}"
    result += f"\n- AI执{ai_color_str} (难度: {difficulty})"
    result += f"\n- 搜索深度: {human_vs_ai.max_depth}"
    
    if human_vs_ai.human_color == 1:
        result += f"\n\n您执黑棋先手，请开始走子！"
    else:
        result += f"\n\nAI执黑棋先手，请等待AI走子..."
    
    return result

def ai_make_move() -> str:
    """让AI执行走子"""
    from .ai_system import get_human_vs_ai
    from tools.gomoku_game import get_current_board, Player, make_move
    
    human_vs_ai = get_human_vs_ai()
    board = get_current_board()
    
    if board.game_over:
        return "游戏已结束，无法走子"
    
    # 检查是否是AI的回合
    current_player_value = 1 if board.current_player == Player.BLACK else 2
    if not human_vs_ai.is_ai_turn(current_player_value):
        human_color = "黑棋" if human_vs_ai.human_color == 1 else "白棋"
        return f"现在轮到人类({human_color})走子，不是AI的回合"
    
    # 将棋盘转换为AI可用的格式
    ai_board = []
    for i in range(board.size):
        row = []
        for j in range(board.size):
            if board.board[i][j] == Player.BLACK:
                row.append(1)
            elif board.board[i][j] == Player.WHITE:
                row.append(2)
            else:
                row.append(0)
        ai_board.append(row)
    
    # 获取AI走法
    move = human_vs_ai.get_ai_move(ai_board)
    
    # 执行走子
    result = make_move(move[0], move[1])
    
    # 添加AI信息
    difficulty = human_vs_ai.current_difficulty.value
    result = f"AI({difficulty})走子: {result}"
    
    return result

def get_ai_difficulty_info() -> str:
    """获取当前AI难度信息"""
    from .ai_system import get_human_vs_ai
    
    human_vs_ai = get_human_vs_ai()
    
    human_color = "黑棋" if human_vs_ai.human_color == 1 else "白棋"
    ai_color = "黑棋" if human_vs_ai.ai_color == 1 else "白棋"
    
    info = f"当前人类对战AI设置:\n"
    info += f"- AI难度: {human_vs_ai.current_difficulty.value}\n"
    info += f"- 搜索深度: {human_vs_ai.max_depth}\n"
    info += f"- 人类执{human_color}\n"
    info += f"- AI执{ai_color}\n"
    
    from tools.gomoku_game import get_current_board, Player
    board = get_current_board()
    if not board.game_over:
        current_player_value = 1 if board.current_player == Player.BLACK else 2
        if human_vs_ai.is_ai_turn(current_player_value):
            info += f"- 当前回合: AI"
        else:
            info += f"- 当前回合: 人类"
    
    return info

def bulid_tools()->List[Tool]:
    """构建五子棋Agent的工具集"""
    return [
        Tool(
            name="initGame",
            description=(
                "初始化新的五子棋游戏。"
                "输入: 棋盘大小（可选，默认15），例如 '15' 或 'size=15'"
                "返回: 初始化结果信息"
            ),
            func=lambda size_str: init_game(int(size_str.strip()) if size_str.strip().isdigit() else 15),
        ),
        Tool(
            name="makeMove",
            description=(
                "在棋盘上落子。"
                "输入: 位置坐标，格式为 'row,col' 或 'row col'，例如 '7,7' 或 '7 7'"
                "坐标范围: 0-14（15x15棋盘）"
                "返回: 走子结果，包括是否成功、下一步玩家、游戏状态等"
            ),
            func=lambda pos: _parse_and_make_move(pos),
        ),
        Tool(
            name="getBoardState",
            description=(
                "获取当前棋盘状态的完整信息。"
                "输入: 任意文本（通常为 'current' 或 'state'）"
                "返回: 棋盘可视化、当前玩家、走子历史、游戏状态等详细信息"
            ),
            func=lambda _: get_board_state(),
        ),
        Tool(
            name="evaluatePosition",
            description=(
                "评估当前局面的优劣。"
                "输入: 任意文本（通常为 'evaluate' 或 'analyze'）"
                "返回: 局面评估，包括威胁点、机会点、当前状态等"
            ),
            func=lambda _: evaluate_position(),
        ),
        Tool(
            name="suggestMoves",
            description=(
                "获取最佳走法建议。"
                "输入: 可选的最大建议数量（默认为5），例如 '5'"
                "返回: 按照优先级排序的走法建议列表，包括位置、原因、优先级评分"
            ),
            func=lambda n: suggest_moves(int(n.strip()) if n.strip().isdigit() else 5),
        ),
        Tool(
            name="downloadDataset",
            description=(
                "下载五子棋数据集（棋谱、开局库等）。"
                "输入: 保存路径和dataset_type，格式必须为 'save_path,dataset_type'"
                "示例: 'output/gomoku_dataset.json,games' 或 'data/openings.json,openings'"
                "dataset_type必须是: 'games' 或 'openings'"
                "返回: 下载结果，包括保存路径和数据统计"
            ),
            func=_parse_download_args,
        ),
        Tool(
            name="loadDataset",
            description=(
                "加载并查看数据集信息。"
                "输入: 数据集文件路径"
                "返回: 数据集摘要信息，包括条目数量、格式等"
            ),
            func=load_dataset,
        ),
        Tool(
            name="analyzeOpening",
            description=(
                "分析开局模式和走法统计。"
                "输入: 数据集文件路径"
                "返回: 开局走法统计，包括常见第一步走法等"
            ),
            func=analyze_opening,
        ),
        Tool(
            name="saveGame",
            description=(
                "保存当前游戏状态到文件。"
                "输入: 保存文件路径，例如 'games/game1.json'"
                "返回: 保存结果和文件路径"
            ),
            func=save_game,
        ),
        Tool(
            name="loadGame",
            description=(
                "从文件加载游戏状态。"
                "输入: 游戏文件路径"
                "返回: 加载结果信息"
            ),
            func=load_game,
        ),
        Tool(
            name="resetGame",
            description=(
                "重置当前游戏，清空棋盘，黑棋先行。"
                "输入: 任意文本（通常为 'reset'）"
                "返回: 重置确认信息"
            ),
            func=lambda _: reset_game(),
        ),
        Tool(
            name="startHumanVsAI",
            description=(
                "开始人类与AI对战。"
                "输入: 难度和AI颜色，格式为 'difficulty,ai_color' 或 'difficulty'"
                "示例: 'expert,black' 或 'intermediate'"
                "难度: beginner, intermediate, advance, expert"
                "AI颜色: black(先手), white(后手)，默认white"
                "返回: 对战开始信息和设置"
            ),
            func=lambda args: start_human_vs_ai(*args.split(',')) if ',' in args else start_human_vs_ai(args),
        ),
        Tool(
            name="aiMakeMove",
            description=(
                "让AI执行走子（在AI的回合调用）。"
                "输入: 任意文本（通常为 'move' 或 'go'）"
                "返回: AI走子结果和位置信息"
            ),
            func=lambda _: ai_make_move(),
        ),
        Tool(
            name="getAIDifficultyInfo", 
            description=(
                "获取当前AI难度设置信息。"
                "输入: 任意文本"
                "返回: AI难度、执棋颜色、当前回合等信息"
            ),
            func=lambda _: get_ai_difficulty_info(),
        ),
    ]


def build_agent() -> AgentExecutor:
    """构建五子棋AI Agent"""
    # 兼容部分环境下 Pydantic 类未完全构建的问题
    try:
        ChatOpenAI.model_rebuild(force=True)
    except Exception:
        pass
    
    cfg = load_config()

    http_client = httpx.Client(timeout=60.0, http2=False)

    llm = ChatOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=0.3,  # 降低温度以保持策略稳定性
        max_retries=2,
        http_client=http_client,
    )

    tools = bulid_tools()

    # 使用标准的 ReAct prompt，必须使用英文关键词以确保 LangChain 正确解析
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**重要规则：**
1. Action 和 Action Input 必须分开写，格式如下：
   Action: tool_name
   Action Input: input_value
   
2. 不要写成 Action: tool_name('input') 这样的格式，这是错误的！

3. 一旦任务完成，立即给出 Final Answer，不要再调用工具

4. 对于简单任务（如初始化游戏），调用工具后直接给出 Final Answer

**示例格式：**
Question: 初始化一个15x15的五子棋游戏
Thought: 我需要调用 initGame 工具来初始化游戏
Action: initGame
Action Input: 15
Observation: 已初始化 15x15 五子棋游戏，黑棋先行
Thought: 游戏已成功初始化，任务完成
Final Answer: 游戏已成功初始化为15x15大小，黑棋先行

**工具说明：**
- initGame: 初始化新游戏，输入为棋盘大小（如 '15'）
- getBoardState: 查看当前棋盘状态，输入任意文本（如 'current'）
- makeMove: 执行走子，输入格式为 'row,col'（如 '7,7'）
- evaluatePosition: 评估当前局面，输入任意文本
- suggestMoves: 获取最佳走法建议，输入建议数量（如 '5'）
- downloadDataset: 下载数据集，输入格式为 'path,type'（如 'path.json,games'）
- 其他工具请参考工具描述

Question: {input}
Thought: {agent_scratchpad}
        """.strip()
    )
    
    agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)

    # 自定义解析错误处理函数
    def handle_parsing_error(error:Exception)->str:
        """处理解析错误，返回友好的错误信息"""
        return f"解析错误，请严格按照格式输出。错误：{str(error)[:100]}"
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=50,
        handle_parsing_errors=handle_parsing_error,
        return_intermediate_steps=True,
        max_execution_time=600,
        early_stopping_method="force"
    )
    return executor