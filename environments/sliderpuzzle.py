import numpy as np
import cv2, math, time, random
import keyboard as k
from colorsys import hsv_to_rgb

class SliderPuzzleEnv:
    num_players = 1
    framerate=20
    resolution = 800, 400

    frames = 5

    def __init__(self, difficulty="medium", **kwargs):
        self.grid = None

        match difficulty:
            case "easy":
                self.N = 3
                self.solved_grid = [ 1,  2,  3,
                                     4,  5,  6,
                                     7,  8,  0 ]
            case "medium":
                self.N = 4
                self.solved_grid = [ 1,  2,  3,  4,
                                     5,  6,  7,  8,
                                     9,  10, 11, 12,
                                     13, 14, 15, 0 ]
            case "hard":
                self.N = 5
                self.solved_grid = [ 1,  2,  3,  4,  5,
                                     6,  7,  8,  9, 10,
                                     11, 12, 13, 14, 15,
                                     16, 17, 18, 19, 20,
                                     21, 22, 23, 24,  0 ]
    
    def reset(self):
        self.grid = self.solved_grid[:]

        while True:
            random.shuffle(self.grid)
            if self.is_solvable(self.grid):
                break
        
        self.prev_keys = {}

        TILE_SIZE = 120
        self.textures = [self.create_wood_texture(TILE_SIZE) for i in range(25)]
        self.moving_tile = None
        self.animation = 3
        self.animation_tick = 0
        self.last_frame = time.time()

    def is_solvable(self, grid):
        arr = [x for x in grid if x != 0]
        inversions = 0
        
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inversions += 1

        N = int(len(grid)**0.5)
        if N % 2 == 1:
            return inversions % 2 == 0
        else:
            return (inversions + N-grid.index(0)//N) % 2 == 1

    def is_solved(self, grid):
        return grid == self.solved_grid

    def get_neighbors(self, grid):
        neighbors = [None for i in range(4)] # up down left right
        dirs = ((1,0), (-1,0), (0,1), (0,-1))
        hole = grid.index(0)
        r, c = divmod(hole, self.N)

        for n, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.N and 0 <= nc < self.N:
                swap_idx = nr * self.N + nc
                new_grid = grid[:]
                new_grid[hole], new_grid[swap_idx] = new_grid[swap_idx], new_grid[hole]
                neighbors[n] = new_grid

        return neighbors


    def getInputs(self):
        return {
            "p1": {
                "grid": self.grid[:],
            }
        }
    
    def getState(self):
        return {
            "grid": self.grid[:],
            "moving_tile": self.moving_tile,
            "animation_tick": self.animation_tick,
            "animation": self.animation,
            "solved": self.is_solved(self.grid)
        }

    def step(self, actions, keyboard={}, display=False):

        action = actions[f"p1"]
        neighbors = self.get_neighbors(self.grid)

        if action == "keyboard":
            keys = {'w':0, 's':1, 'a':2, 'd':3,
                    'ArrowUp':0, 'ArrowDown':1, 'ArrowLeft':2, 'ArrowRight':3}
            for key in keys:
                if keyboard.get(key) and not self.prev_keys.get(key): 
                    action = [keys[key] + 1]
                    break
            else:
                action = [0]
        self.prev_keys = keyboard

        new_grid = neighbors[action[0]-1] if action[0] else None
        if new_grid:
            self.moving_tile = self.grid.index(0)
            self.grid = new_grid
            self.animation_tick = self.animation
        
        done = self.is_solved(self.grid)
        
        if self.animation_tick:
            self.animation_tick -= 1

        return self.grid, 0, 0

    def create_wood_texture(self, size):
        base = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Base wood color
        base[:] = (60, 120, 180)  # BGR (warm brown)
        
        # Add vertical grain lines
        for i in range(0, size, 4):
            color_variation = np.random.randint(-10, 10)
            cv2.line(base, (i, 0), (i, size),
                    (60+color_variation, 120+color_variation, 180+color_variation), 1)
        
        # Slight blur for smoother grain
        base = cv2.GaussianBlur(base, (5,5), 0)
        return base

    def display(self):
        grid = self.grid
        N = self.N
        TILE_SIZE = 120
        PADDING = 10
        BOARD_SIZE = N * TILE_SIZE + (N + 1) * PADDING

        img = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
        img[:] = (40, 80, 120)  # darker wood background

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(N):
            for j in range(N):
                tile = i*N + j
                hole = self.grid.index(0)
                if tile == self.moving_tile:
                    dist = int((TILE_SIZE + PADDING) * (self.animation_tick / self.animation)**2)
                    offset_x = -(tile%N - hole%N) * dist
                    offset_y = -(tile//N - hole//N) * dist
                else:
                    offset_x, offset_y = 0, 0

                value = grid[tile]
                
                x = PADDING + j*(TILE_SIZE + PADDING) + offset_x
                y = PADDING + i*(TILE_SIZE + PADDING) + offset_y

                if value != 0:
                    # Draw shadow
                    cv2.rectangle(img,
                                (x+5, y+5),
                                (x+TILE_SIZE+5, y+TILE_SIZE+5),
                                (20, 40, 60), -1)


                    img[y:y+TILE_SIZE, x:x+TILE_SIZE] = self.textures[value]

                    # Bevel effect (light top-left, dark bottom-right)
                    cv2.line(img, (x, y), (x+TILE_SIZE, y), (100,150,200), 2)
                    cv2.line(img, (x, y), (x, y+TILE_SIZE), (100,150,200), 2)
                    cv2.line(img, (x, y+TILE_SIZE), (x+TILE_SIZE, y+TILE_SIZE), (30,30,30), 2)
                    cv2.line(img, (x+TILE_SIZE, y), (x+TILE_SIZE, y+TILE_SIZE), (30,30,30), 2)

                    # Draw number
                    text = str(value)
                    (w, h), _ = cv2.getTextSize(text, font, 1.5, 3)
                    text_x = x + (TILE_SIZE - w)//2
                    text_y = y + (TILE_SIZE + h)//2
                    cv2.putText(img, text, (text_x, text_y),
                                font, 1.5, (30,30,30), 3, cv2.LINE_AA)

        cv2.imshow('img', img)        
        cv2.waitKey(1)
        

if __name__ == "__main__":
    import keyboard as k
    import time

    player1 = "keyboard"

    #from YourPyScript import Agent as player1
                
    game = SliderPuzzleEnv(difficulty="easy") # easy, medium, hard
    game.reset()

    if player1 != "keyboard": player1 = player1()

    while True:
        keys = {}
        inputs = game.getInputs()
        actions1 = [0]
        
        if player1 == "keyboard":
            for key in "wasd":
                if k.is_pressed(key): keys[key] = True
            actions1 = "keyboard"
        else:
            actions1 = player1.getAction(inputs["p1"])

        game.step({"p1":actions1}, keyboard=keys)
        game.display()

        time.sleep(1/20)

