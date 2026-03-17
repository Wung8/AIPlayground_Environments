import numpy as np
import cv2, math, time, random
import keyboard as k
from colorsys import hsv_to_rgb

class TicTacToeEnv:
    num_players = 2
    framerate=20
    resolution = 800, 400

    checks = [
        # rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]

    turn_to_player = {
        "x":"p1",
        "o":"p2"
    }

    def __init__(self, **kwargs):
        self.grid = None
    
    def reset(self):
        self.grid = "........."
        self.turn = "x"
        self.cursor = 0, 0

        self.last_keyboard = {}

        self.frame_count = 0
        self.sketch_cache = {}
        self.last_frame = time.time()

    def getInputs(self):
        return {
            "p1": {
                "grid": self.grid,
                "your_position": self.cursor,
                "your_turn": self.turn == "x",
            },
            "p2": {
                "grid": self.grid,
                "your_position": self.cursor,
                "your_turn": self.turn == "o",
            },
        }

    def getState(self):
        # convert "........." into 3x3 grid
        grid_2d = [
            [self.grid[3*r + c] for c in range(3)]
            for r in range(3)
        ]

        # initialize animation counters if missing
        if not hasattr(self, "frame_count"):
            self.frame_count = 0
        if not hasattr(self, "sketch_cache"):
            self.sketch_cache = {}

        # regenerate sketch jitter every 4 frames
        if self.frame_count % 4 == 0:
            import random
            self.sketch_cache = {}

            def jitter():
                return random.randint(-2, 2)

            # grid lines
            for i in range(4):
                self.sketch_cache[f"v{i}_1"] = (jitter(), jitter())
                self.sketch_cache[f"v{i}_2"] = (jitter(), jitter())
                self.sketch_cache[f"h{i}_1"] = (jitter(), jitter())
                self.sketch_cache[f"h{i}_2"] = (jitter(), jitter())

            # marks
            for i in range(9):
                self.sketch_cache[f"x{i}a_1"] = (jitter(), jitter())
                self.sketch_cache[f"x{i}a_2"] = (jitter(), jitter())
                self.sketch_cache[f"x{i}b_1"] = (jitter(), jitter())
                self.sketch_cache[f"x{i}b_2"] = (jitter(), jitter())
                self.sketch_cache[f"o{i}"] = (jitter(), jitter())

            # cursor
            for k in range(1, 5):
                self.sketch_cache[f"c{k}_1"] = (jitter(), jitter())
                self.sketch_cache[f"c{k}_2"] = (jitter(), jitter())

        self.frame_count += 1

        return {
            "grid": grid_2d,
            "cursor": self.cursor,
            "turn": self.turn,
            "sketch": self.sketch_cache
        }
    
    def checkWinner(self, grid):
        for check in self.checks:
            s = {grid[i] for i in check}
            if len(s) == 1:
                if s == {"x"}: return "p1"
                elif s == {"o"}: return "p2"
        return None

    def getNeighbors(self, grid, turn):
        neighbors = []
        for i in range(9):
            if grid[i] != ".":
                neighbors.append(None)
                continue
            nbr = grid[:i] + turn + grid[i+1:]
            neighbors.append(nbr)
        return neighbors
            

    def step(self, actions, keyboard={}, display=False):
        player = self.turn_to_player[self.turn]
        action = actions[player]
        if action == "keyboard":
            action = [0, 0, 0]
            if player == "p1": # certified best coding practice
                if keyboard.get('w') and not self.last_keyboard.get('w'): action[0] = -1
                if keyboard.get('a') and not self.last_keyboard.get('a'): action[1] = -1
                if keyboard.get('s') and not self.last_keyboard.get('s'): action[0] = 1
                if keyboard.get('d') and not self.last_keyboard.get('d'): action[1] = 1
                if keyboard.get('q') and not self.last_keyboard.get('q'): action[2] = 1
            if player == "p2":
                if keyboard.get('ArrowUp') and not self.last_keyboard.get('ArrowUp'): action[0] = -1
                if keyboard.get('ArrowLeft') and not self.last_keyboard.get('ArrowLeft'): action[1] = -1
                if keyboard.get('ArrowDown') and not self.last_keyboard.get('ArrowDown'): action[0] = 1
                if keyboard.get('ArrowRight') and not self.last_keyboard.get('ArrowRight'): action[1] = 1
                if keyboard.get('.') and not self.last_keyboard.get('.'): action[2] = 1
        
        self.cursor = (
            max(0, min(2, self.cursor[0] + action[0])), 
            max(0, min(2, self.cursor[1] + action[1]))
        )

        if action[2]:
            neighbors = self.getNeighbors(self.grid, self.turn)
            cursor_idx = 3*self.cursor[0] + self.cursor[1]
            nbr = neighbors[cursor_idx]
            if nbr:
                self.grid = nbr
                self.turn = "x" if self.turn == "o" else "o"

        self.last_keyboard = keyboard

        return 0, 0, 0
    

    def display(self):
        width, height = self.resolution
        
        # Paper background
        img = np.full((height, width, 3), (230, 225, 210), dtype=np.uint8)

        board_size = min(width, height) - 120
        cell_size = board_size // 3

        offset_x = (width - board_size) // 2
        offset_y = (height - board_size) // 2

        pencil_color = (60, 60, 60)

        # ---- Frame-controlled sketch randomness ----
        if self.frame_count % 4 == 0:
            self.sketch_cache = {}
        self.frame_count += 1

        def get_jitter(key, amount=2):
            if key not in self.sketch_cache:
                self.sketch_cache[key] = (
                    random.randint(-amount, amount),
                    random.randint(-amount, amount)
                )
            return self.sketch_cache[key]

        def sketch_line(p1, p2, key, thickness=2):
            jx1, jy1 = get_jitter(key + "_1")
            jx2, jy2 = get_jitter(key + "_2")

            pt1 = (p1[0] + jx1, p1[1] + jy1)
            pt2 = (p2[0] + jx2, p2[1] + jy2)

            cv2.line(img, pt1, pt2, pencil_color,
                    thickness, cv2.LINE_AA)

        # ---- Grid ----
        for i in range(4):
            x = offset_x + i * cell_size
            y = offset_y + i * cell_size

            sketch_line((x, offset_y),
                        (x, offset_y + board_size),
                        f"v{i}")

            sketch_line((offset_x, y),
                        (offset_x + board_size, y),
                        f"h{i}")

        # ---- Marks ----
        for row in range(3):
            for col in range(3):
                idx = 3 * row + col
                value = self.grid[idx]

                cx = offset_x + col * cell_size + cell_size // 2
                cy = offset_y + row * cell_size + cell_size // 2

                if value == "x":
                    size = cell_size // 3
                    sketch_line((cx - size, cy - size),
                                (cx + size, cy + size),
                                f"x{idx}a", 3)
                    sketch_line((cx - size, cy + size),
                                (cx + size, cy - size),
                                f"x{idx}b", 3)

                elif value == "o":
                    radius = cell_size // 3
                    jx, jy = get_jitter(f"o{idx}", 2)
                    cv2.circle(img,
                            (cx + jx, cy + jy),
                            radius,
                            pencil_color,
                            2,
                            cv2.LINE_AA)

        # ---- Cursor ----
        if self.turn == "x":
            cursor_color = (80, 80, 200)
        else:
            cursor_color = (200, 100, 50)

        cursor_row, cursor_col = self.cursor
        x1 = offset_x + cursor_col * cell_size
        y1 = offset_y + cursor_row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        sketch_line((x1, y1), (x2, y1), "c1")
        sketch_line((x2, y1), (x2, y2), "c2")
        sketch_line((x2, y2), (x1, y2), "c3")
        sketch_line((x1, y2), (x1, y1), "c4")

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), cursor_color, -1)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        # ---- Text ----
        turn_text = "X's Turn" if self.turn == "x" else "O's Turn"
        cv2.putText(img, turn_text,
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (70, 70, 70),
                    2,
                    cv2.LINE_AA)

        cv2.imshow("TicTacToe", img)

        this_frame = time.time()
        delay = max(int(1000/self.framerate - (this_frame-self.last_frame)), 20)
        cv2.waitKey(delay)
        self.last_frame = this_frame

if __name__ == "__main__":      
    env = TicTacToeEnv(difficulty="hard")
    env.reset()
    while True:
        keyboard = {}
        for key in {"w", "a", "s", "d", "q", "up", "left", "down", "right", "."}:
            if k.is_pressed(key): keyboard[key] = True
        if k.is_pressed('r'): env.reset()
        inputs, r, done = env.step({"p1":"keyboard", "p2":"keyboard"}, keyboard=keyboard)
        env.display()

        if done:
            env.reset()
