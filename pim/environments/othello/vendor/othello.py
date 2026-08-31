"""Vendored from Li et al.'s othello_world repo (MIT, Copyright (c) 2023 Kenneth Li).

Source: https://github.com/likenneth/othello_world @ data/othello.py, vendored 2026-08-31.
This copy keeps ONLY what this project calls — the board rules (`OthelloBoardState`), the
synthetic-game generator (`get_ood_game`), and their module-level constants — and strips
the championship-data loader (`Othello`) and the seaborn/matplotlib plotting, whose
module-level imports (pgn, psutil, seaborn, tqdm) the original forced on every importer.
Nothing kept here is modified: the retained lines are byte-identical to upstream, so
numbers produced through this copy are numbers produced through their code.
See vendor/LICENSE for the full license text.
"""

import random
from copy import copy, deepcopy

import numpy as np

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

mask = np.zeros(64).reshape(8, 8)
mask[3, 3] = 1
mask[3, 4] = 1
mask[4, 3] = 1
mask[4, 4] = 1
mask = mask.astype(bool)

def permit(s):
    s = s.lower()
    if len(s) != 2:
        return -1
    if s[0] not in rows or s[1] not in columns:
        return -1
    return rows.index(s[0]) * 8 + columns.index(s[1])

def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])

start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

wanna_use = "othello_synthetic"

def get_ood_game(_):
    tbr = []
    ab = OthelloBoardState()
    possible_next_steps = ab.get_valid_moves()
    while possible_next_steps:
        next_step = random.choice(possible_next_steps)
        tbr.append(next_step)
        ab.update([next_step, ])
        possible_next_steps = ab.get_valid_moves()
    return tbr


class OthelloBoardState():
    # 1 is black, -1 is white
    def __init__(self, board_size = 8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []

    def get_occupied(self, ):
        board = self.state
        tbr = board.flatten() != 0
        return tbr.tolist()
    def get_state(self, ):
        board = self.state + 1  # white 0, blank 1, black 2
        tbr = board.flatten()
        return tbr.tolist()
    def get_age(self, ):
        return self.age.flatten().tolist()
    def get_next_hand_color(self, ):
        return (self.next_hand_color + 1) // 2
    
    def update(self, moves, prt=False):
        # takes a new move or new moves and update state
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()

    def umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
                
        self.age += 1
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)
        
    def __print__(self, ):
        print("-"*20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
            # tbp.append("\n")
            print(" ".join([a[k]] + tbp))
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-"*20)
        
    def plot_hm(self, ax, heatmap, pdmove, logit=False):
        raise NotImplementedError(
            "plotting stripped from the vendored copy; use upstream othello_world")

    def tentative_move(self, move):
        # tentatively put a piece, do nothing to state
        # returns 0 if this is not a move at all: occupied or both player have to forfeit
        # return 1 if regular move
        # return 2 if forfeit happens but the opponent can drop piece at this place
        r, c = move // 8, move % 8
        if not self.state[r, c] == 0:
            return 0
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) != 0:
            return 1
        else:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            # self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
            if len(tbf) == 0:
                return 0
            else:
                return 2
        
    def get_valid_moves(self, ):
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
            else:
                pass
        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
 
    def get_gt(self, moves, func, prt=False):
        # takes a new move or new moves and update state
        container = []
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            container.append(getattr(self, func)())  
            # to predict first y, we need already know the first x
            if prt:
                self.__print__()
        return container

if __name__ == "__main__":
    pass
