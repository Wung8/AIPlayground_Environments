import numpy as np
import cv2, time, random
import keyboard as k

class SudokuEnv:
    num_players = 1
    framerate = 20
    resolution = 800, 800

    def __init__(self, **kwargs):
        self.grid = None
        self.original = None
        self.cursor = [0, 0]

    def reset(self):
        # Simple fixed puzzle (you can randomize later)
        self.original = np.array([
            [5,3,0, 0,7,0, 0,0,0],
            [6,0,0, 1,9,5, 0,0,0],
            [0,9,8, 0,0,0, 0,6,0],

            [8,0,0, 0,6,0, 0,0,3],
            [4,0,0, 8,0,3, 0,0,1],
            [7,0,0, 0,2,0, 0,0,6],

            [0,6,0, 0,0,0, 2,8,0],
            [0,0,0, 4,1,9, 0,0,5],
            [0,0,0, 0,8,0, 0,7,9]
        ], dtype=np.int8)

        self.grid = self.original.copy()
        self.cursor = [0, 0]
        self.prev_keys = {}
        self.last_frame = time.time()

    def getState(self):
        return {
            "grid": self.grid.tolist(),
            "original": self.original.tolist(),
            "cursor": list(self.cursor),
        }

    def getInputs(self):
        return {"p1": {
            "grid": self.grid.tolist(),
            "original": self.original.tolist(),
            "cursor": list(self.cursor),
        }}

    # -------------------------
    # Game Logic
    # -------------------------

    def is_valid(self, row, col, val):
        if val == 0:
            return True

        # Row check
        if val in self.grid[row]:
            return False

        # Column check
        if val in self.grid[:, col]:
            return False

        # 3x3 box check
        box_r, box_c = row // 3 * 3, col // 3 * 3
        if val in self.grid[box_r:box_r+3, box_c:box_c+3]:
            return False

        return True

    def is_solved(self):
        return np.all(self.grid != 0)

    # -------------------------
    # Step Function
    # -------------------------

    def step(self, actions, keyboard={}, display=False):

        action = actions["p1"]

        if action == "keyboard":

            # Movement
            move_keys = {
                'w': (-1, 0),
                's': (1, 0),
                'a': (0, -1),
                'd': (0, 1),
                'ArrowUp': (-1, 0),
                'ArrowDown': (1, 0),
                'ArrowLeft': (0, -1),
                'ArrowRight': (0, 1)
            }

            for key, (dr, dc) in move_keys.items():
                if keyboard.get(key) and not self.prev_keys.get(key):
                    self.cursor[0] = int(np.clip(self.cursor[0] + dr, 0, 8))
                    self.cursor[1] = int(np.clip(self.cursor[1] + dc, 0, 8))

            # Number input
            for num in range(1, 10):
                key = str(num)
                if keyboard.get(key) and not self.prev_keys.get(key):
                    r, c = self.cursor
                    if self.original[r, c] == 0:
                        self.grid[r, c] = num

            # Clear cell
            if keyboard.get('0') and not self.prev_keys.get('0'):
                r, c = self.cursor
                if self.original[r, c] == 0:
                    self.grid[r, c] = 0

        self.prev_keys = keyboard.copy()

        done = self.is_solved()
        return self.grid, 0, done

    # -------------------------
    # Rendering
    # -------------------------

    def has_conflict(self, row, col):
        val = self.grid[row, col]
        if val == 0:
            return False

        # Check row
        for c in range(9):
            if c != col and self.grid[row, c] == val:
                return True

        # Check column
        for r in range(9):
            if r != row and self.grid[r, col] == val:
                return True

        # Check 3x3 box
        box_r, box_c = row // 3 * 3, col // 3 * 3
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if (r != row or c != col) and self.grid[r, c] == val:
                    return True

        return False

    def display(self):

        SIZE = 720
        CELL = SIZE // 9
        img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 235
        hovered_num = self.grid[*self.cursor]
        if hovered_num == 0: hovered_num = None

        font = cv2.FONT_HERSHEY_SIMPLEX

        # -----------------------
        # 3x3 Box Shading
        # -----------------------
        for box_r in range(3):
            for box_c in range(3):
                if (box_r + box_c) % 2 == 0:
                    color = (220, 220, 220)  # soft teal tint
                    r0 = box_r * 3 * CELL
                    c0 = box_c * 3 * CELL
                    cv2.rectangle(
                        img,
                        (c0, r0),
                        (c0 + 3*CELL, r0 + 3*CELL),
                        color,
                        -1
                    )

        primary = (160, 180, 100)
        secondary = (200, 210, 170)
        r, c = self.cursor
        for nr in range(0, 9):
            cv2.rectangle(
                img,
                (c*CELL, nr*CELL),
                ((c+1)*CELL, (nr+1)*CELL),
                secondary,  
                -1
            )
        for nc in range(0, 9):
            cv2.rectangle(
                img,
                (nc*CELL, r*CELL),
                ((nc+1)*CELL, (r+1)*CELL),
                secondary,  
                -1
            )

        for nr in range(9):
            for nc in range(9):
                if self.grid[nr, nc] == hovered_num:
                    cv2.rectangle(
                        img,
                        (nc*CELL, nr*CELL),
                        ((nc+1)*CELL, (nr+1)*CELL),
                        (200, 150, 200),  
                        -1
                    )
        
        cv2.rectangle(
            img,
            (c*CELL, r*CELL),
            ((c+1)*CELL, (r+1)*CELL),
            primary,  
            -1
        )

        # -----------------------
        # Grid Lines
        # -----------------------
        for i in range(10):
            thickness = 3 if i % 3 == 0 else 1
            color = (80, 80, 80)
            cv2.line(img, (0, i*CELL), (SIZE, i*CELL), color, thickness)
            cv2.line(img, (i*CELL, 0), (i*CELL, SIZE), color, thickness)

        # -----------------------
        # Numbers
        # -----------------------
        for i in range(9):
            for j in range(9):
                val = self.grid[i, j]
                if val != 0:

                    if self.original[i, j] != 0:
                        color = (20, 20, 20)  # original numbers (dark gray)
                    else:
                        if self.has_conflict(i, j):
                            color = (40, 40, 200)  # red conflict
                        else:
                            color = (200, 60, 0)  # blue user number (BGR)

                    text = str(val)
                    (w, h), _ = cv2.getTextSize(text, font, 1.8, 3)
                    x = j*CELL + (CELL - w)//2
                    y = i*CELL + (CELL + h)//2

                    cv2.putText(
                        img,
                        text,
                        (x, y),
                        font,
                        1.8,
                        color,
                        3,
                        cv2.LINE_AA
                    )

        cv2.imshow("Sudoku", img)

        this_frame = time.time()
        cv2.waitKey(max(int(1000/self.framerate-(this_frame-self.last_frame)), 20))
        self.last_frame = this_frame


if __name__ == "__main__":
# -------------------------
# Run Loop (same style)
# -------------------------

    env = SudokuEnv()
    env.reset()

    while True:
        keyboard_input = {}

        for key in list("wasd123456789"):
            if k.is_pressed(key):
                keyboard_input[key] = True

        if k.is_pressed('backspace'):
            keyboard_input['backspace'] = True

        if k.is_pressed('r'):
            env.reset()

        _, _, done = env.step({"p1": "keyboard"}, keyboard=keyboard_input)
        env.display()

        if done:
            print("Solved!")
            time.sleep(2)
            env.reset()