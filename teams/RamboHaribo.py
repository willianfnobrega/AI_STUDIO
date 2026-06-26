class GomokuAgent:
    def __init__(self, agent_symbol, blank_symbol, opponent_symbol):
        self.name = __name__
        self.agent_symbol = agent_symbol
        self.blank_symbol = blank_symbol
        self.opponent_symbol = opponent_symbol

    def play(self, board):
        import math
        size = len(board)
        AGENT = self.agent_symbol
        OPP   = self.opponent_symbol
        BLNK  = self.blank_symbol
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        DEPTH = 1

        # --------------------------
        # Winner detection
        # --------------------------
        def detect_winner(bd):
            for r in range(size):
                for c in range(size):
                    sym = bd[r][c]
                    if sym == BLNK: 
                        continue
                    for dr, dc in directions:
                        cnt = 1
                        rr, cc = r+dr, c+dc
                        while 0<=rr<size and 0<=cc<size and bd[rr][cc]==sym:
                            cnt += 1; rr += dr; cc += dc
                        rr, cc = r-dr, c-dc
                        while 0<=rr<size and 0<=cc<size and bd[rr][cc]==sym:
                            cnt += 1; rr -= dr; cc -= dc
                        if cnt >= 5: 
                            return sym
            return None

        # --------------------------
        # Line evaluation
        # --------------------------
        def eval_line(bd, r, c, sym):
            val = 0
            for dr, dc in directions:
                cnt = 1
                open_ends = 0

                # forward
                rr, cc = r+dr, c+dc
                while 0<=rr<size and 0<=cc<size and bd[rr][cc] == sym:
                    cnt += 1; rr += dr; cc += dc
                if 0<=rr<size and 0<=cc<size and bd[rr][cc] == BLNK:
                    open_ends += 1

                # backward
                rr, cc = r-dr, c-dc
                while 0<=rr<size and 0<=cc<size and bd[rr][cc] == sym:
                    cnt += 1; rr -= dr; cc -= dc
                if 0<=rr<size and 0<=cc<size and bd[rr][cc] == BLNK:
                    open_ends += 1

                # aggressive scoring
                if cnt >= 5:
                    return 1_000_000
                if cnt == 4:
                    val += 50_000 if open_ends == 2 else 5_000
                elif cnt == 3:
                    val += 5_000 if open_ends == 2 else 500
                elif cnt == 2:
                    val += 500 if open_ends == 2 else 50
                elif cnt == 1:
                    val += 5
            return val

        # --------------------------
        # Board heuristic
        # --------------------------
        def heuristic(bd):
            w = detect_winner(bd)
            if w == AGENT: return float('inf')
            if w == OPP:   return float('-inf')
            score = 0
            for r in range(size):
                for c in range(size):
                    if bd[r][c] == AGENT:
                        score += eval_line(bd, r, c, AGENT)
                    elif bd[r][c] == OPP:
                        score -= eval_line(bd, r, c, OPP)
            return score

        # --------------------------
        # Candidate move generation
        # --------------------------
        def generate_candidates(bd):
            cand = set()
            for r in range(size):
                for c in range(size):
                    if bd[r][c] != BLNK:
                        for dr in (-2,-1,0,1,2):
                            for dc in (-2,-1,0,1,2):
                                rr, cc = r+dr, c+dc
                                if 0<=rr<size and 0<=cc<size and bd[rr][cc]==BLNK:
                                    cand.add((rr,cc))
            # sort by proximity to center (better pruning)
            center = size // 2
            return sorted(cand, key=lambda m: abs(m[0]-center) + abs(m[1]-center))

        # --------------------------
        # Minimax search
        # --------------------------
        def max_value(bd, α, β, depth):
            if detect_winner(bd) == AGENT: return float('inf')
            if detect_winner(bd) == OPP:   return float('-inf')
            if depth == 0: return heuristic(bd)
            v = -math.inf
            for (r,c) in generate_candidates(bd):
                bd[r][c] = AGENT
                v = max(v, min_value(bd, α, β, depth-1))
                bd[r][c] = BLNK
                if v >= β: return v
                α = max(α, v)
            return v

        def min_value(bd, α, β, depth):
            if detect_winner(bd) == AGENT: return float('inf')
            if detect_winner(bd) == OPP:   return float('-inf')
            if depth == 0: return heuristic(bd)
            v = math.inf
            for (r,c) in generate_candidates(bd):
                bd[r][c] = OPP
                v = min(v, max_value(bd, α, β, depth-1))
                bd[r][c] = BLNK
                if v <= α: return v
                β = min(β, v)
            return v

        # --------------------------
        # Opening move
        # --------------------------
        if all(cell == BLNK for row in board for cell in row):
            mid = size // 2
            return (mid, mid)

        # --------------------------
        # Immediate win or block
        # --------------------------
        candidates = generate_candidates(board)
        for (r,c) in candidates:
            board[r][c] = AGENT
            if detect_winner(board) == AGENT:
                board[r][c] = BLNK
                return (r, c)
            board[r][c] = BLNK

        for (r,c) in candidates:
            board[r][c] = OPP
            if detect_winner(board) == OPP:
                board[r][c] = BLNK
                return (r, c)
            board[r][c] = BLNK

        # --------------------------
        # Minimax search
        # --------------------------
        best_score = -math.inf
        best_move  = None
        for (r,c) in candidates:
            board[r][c] = AGENT
            score = min_value(board, -math.inf, math.inf, DEPTH-1)
            board[r][c] = BLNK
            if score > best_score:
                best_score, best_move = score, (r, c)

        if not best_move:
            best_move = candidates[0] if candidates else (size//2, size//2)
        return best_move
