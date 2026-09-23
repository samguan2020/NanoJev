#!/usr/bin/env python3
"""Board/move <-> text conversions used to build NanoJev `/api/evaluate` requests."""
import xiangqi_board as xq

PIECE_NAMES = {
    ("red", "G"): "帅", ("red", "A"): "仕", ("red", "E"): "相", ("red", "H"): "马",
    ("red", "R"): "车", ("red", "C"): "炮", ("red", "S"): "兵",
    ("black", "G"): "将", ("black", "A"): "士", ("black", "E"): "象", ("black", "H"): "马",
    ("black", "R"): "车", ("black", "C"): "炮", ("black", "S"): "卒",
}


def render_board_text(board, side_to_move):
    """A plain-text board grid plus whose turn it is; this is the NanoJev `state`."""
    rows = []
    for r in range(9, -1, -1):
        cells = []
        for c in range(9):
            occ = board.get((r, c))
            cells.append(PIECE_NAMES[occ] if occ else "．")
        rows.append(f"row{r}: " + "".join(cells))
    turn = "红方（Red）" if side_to_move == xq.RED else "黑方（Black）"
    return "中国象棋棋盘（行号从上到下为9到0；列号从左到右为0到8；河界在第4行与第5行之间）：\n" \
        + "\n".join(rows) + f"\n轮到{turn}走棋。"


def describe_move(board, move):
    (fr, fc), (tr, tc) = move
    side, piece = board[(fr, fc)]
    name = PIECE_NAMES[(side, piece)]
    target = board.get((tr, tc))
    action = f"吃掉{PIECE_NAMES[target]}" if target else "移动到空格"
    return f"{name} 从 (行{fr},列{fc}) 到 (行{tr},列{tc})：{action}"


def moves_to_choice_criteria(board, moves):
    """Stable move_id -> description mapping for a NanoJev `choice` question."""
    ids = [f"move_{i}" for i in range(len(moves))]
    criteria = {mid: describe_move(board, mv) for mid, mv in zip(ids, moves)}
    return dict(zip(ids, moves)), criteria


def move_to_square_pairs(move):
    (fr, fc), (tr, tc) = move
    return {"from": {"row": fr, "col": fc}, "to": {"row": tr, "col": tc}}


def board_to_json(board):
    return [{"row": r, "col": c, "side": side, "piece": piece} for (r, c), (side, piece) in board.items()]
