import numpy as np
import cv2, math, time, random
import keyboard as k
from colorsys import hsv_to_rgb

class MazeEnv:
    num_players = 1
    framerate=20
    resolution = 800, 400

    difficulty_settings = {
        "easy": (16, 8),
        "medium": (32, 16),
        "hard": (64, 32)
    }

    def __init__(self, size=(32, 16), difficulty=None, **kwargs):
        if difficulty:
            self.size = self.difficulty_settings[difficulty]
        else:
            self.size = size
        
        self.grid = None
        self.player = None
        self.goal = None
        self.colors = None
    
    def reset(self):
        hue = random.random() * 100
        bg_hsv = (hue, 0.3, 50)
        wall_hsv = (hue, 0.3, 100)
        goal_hsv = (hue, 0.5, 200)

        self.colors = {
            "bg": hsv_to_rgb(*bg_hsv),
            "wall": hsv_to_rgb(*wall_hsv),
            "player": (200, 200, 200),
            "goal": hsv_to_rgb(*goal_hsv),
            "complete": (230, 230, 100)
        }

        self.grid = np.ones((2*self.size[0]+1, 2*self.size[1]+1))
        stack = [(random.randint(0, self.size[0]-1), random.randint(0, self.size[1]-1))]
        while stack:
            curr = stack[-1]
            nbrs = [nbr for nbr in self.getNeighbors(curr) if self.grid[self.posToGrid(nbr)]==1]
            
            if not nbrs:
                stack.pop()
                continue
            nbr = random.choice(nbrs)
            grid_pos_curr = self.posToGrid(curr)
            grid_pos_nbr = self.posToGrid(nbr)
            grid_pos_mid = self.posToGrid(np.mean((curr, nbr), axis=0))
            self.grid[grid_pos_curr] = 0
            self.grid[grid_pos_nbr] = 0
            self.grid[grid_pos_mid] = 0
            stack.append(nbr)
        
        self.player = [1,1]
        self.goal = (2*self.size[0]-1, 2*self.size[1]-1)
        self.tick = 0

        self.last_frame = time.time()

    def getInputs(self):
        return {
            "p1": {
                "grid": self.grid.tolist(),
                "your_position": self.player
            }
        }

    def getState(self):
        state = {
            "grid": self.grid.tolist(), 
            "player": self.player,
            "goal": self.goal,
            "colors": self.colors
        }
        return state

    def step(self, actions, keyboard={}, display=False):
        self.tick += 1
        action = actions[f"p1"]
        if action == "keyboard":
            action = [0,0]
            if keyboard.get('w'): action[1] = -1
            if keyboard.get('a'): action[0] = -1
            if keyboard.get('s'): action[1] = 1
            if keyboard.get('d'): action[0] = 1
            if keyboard.get('ArrowUp'): action[1] = -1
            if keyboard.get('ArrowLeft'): action[0] = -1
            if keyboard.get('ArrowDown'): action[1] = 1
            if keyboard.get('ArrowRight'): action[0] = 1
        new_pos = self.player[:]
        new_pos[0] += action[0]
        if self.grid[*new_pos] == 0:
            self.player = new_pos
        new_pos = self.player[:]
        new_pos[1] += action[1]
        if self.grid[*new_pos] == 0:
            self.player = new_pos
        
        done = tuple(self.player) == tuple(self.goal)
        time = self.tick * 0.05 
        return self.getInputs(), time, done
    
    def posToGrid(self, pos):
        return int(2*pos[0]+1), int(2*pos[1]+1)

    def getNeighbors(self, pos):
        nbrs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dy != 0:
                    continue
                new_pos = (pos[0] + dx, pos[1] + dy)
                if self.outOfBounds(new_pos):
                    continue
                nbrs.append(new_pos)
        return nbrs
        
    def outOfBounds(self, pos):
        if not(0 <= pos[0] < self.size[0]):
            return True
        if not(0 <= pos[1] < self.size[1]):
            return True
        return False

    def display(self):
        img = np.array([[self.colors['bg']]], dtype=np.uint8)
        img = img.repeat(self.size[0]*2+1,axis=0).repeat(self.size[1]*2+1,axis=1)
        img[self.grid==1] = self.colors['wall']
        img[*self.player] = self.colors['player']
        img[*self.goal] = self.colors['goal']
        if tuple(self.player) == tuple(self.goal):
            img[*self.player] = self.colors['complete']

        img = img.transpose(1,0,2)

        '''
        mapping = {
            0: "  ",
            1: "XX",
        }
        for row in self.grid:
            s = "".join(mapping[int(tile)] for tile in row)
            print(s)
        '''

        scale = 160 / self.size[0]
        img = img.repeat(scale,axis=0).repeat(scale,axis=1)
        
        cv2.imshow('img', img)        
        cv2.waitKey(1)


if __name__ == "__main__":
    import keyboard as k
    import time

    player1 = "keyboard"

    #from YourPyScript import Agent as player1
                
    game = MazeEnv(difficulty="easy") # easy, medium, hard
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
