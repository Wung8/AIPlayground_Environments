import numpy as np
import cv2, math, time, random
import keyboard as k

def add(l1, l2): return [a+b for a,b in zip(l1,l2)]
def subtract(l1, l2): return [a-b for a,b in zip(l1,l2)]
def scale(lst, s): return [a*s for a in lst]
def norm(lst): return [i/(sum(map(abs, lst))) for i in lst]
def mag(lst): return math.sqrt(sum(i**2 for i in lst))
def turn_int(lst): return [int(i) for i in lst]

class SlimeVolleyballEnv:
    num_players = 2
    resolution = 800, 400

    def __init__(self, render_mode="none", **kwargs):
        self.render_mode = render_mode
        self.score = [0,0]
        self.last_frame = time.time()
        self.framerate = 20
        self.dt = 0.1 # in game time passed per frame
        self.skip_frames = 4 # number of physics frames processed between displaying
        self.screen_size = (800,400) # width, height
        self.timestep = 0

        self.ground_level = 320
        self.net_level = self.ground_level-40
        self.net_width = 5 # width from center, so net is net_width*2 px wide
        self.slime_left = Slime(side=0)
        self.slime_right = Slime(side=1)
        self.ball = Ball()

        self.colors = {'bg': (245,248,254),
                       'ground': (124,227,150),
                       'net': (241,211,124),
                       'ball': (245,191,0),
                       'left': (243,74,0),
                       'right': (0,147,255)}
        for color in self.colors.keys():
            self.colors[color] = self.colors[color][::-1] # convert rgb to bgr
            

    def reset(self):
        self.timestep = 0
        self.slime_left.reset()
        self.slime_right.reset()
        self.ball.reset()
        return self.getInputs()

    def scale(self, lst, s):
        return [x*s for x in lst]

    def shift(self, pos, s):
        return (pos[0]+s, pos[1])

    def flip_pos(self, pos):
        return (self.screen_size[0] - pos[0], pos[1])

    def flip_vel(self, vel):
        return (-vel[0], vel[1])

    def norm_pos(self, pos):
        return (pos[0] - self.screen_size[0]/2, self.ground_level - pos[1])
    
    def norm_vel(self, vel):
        return (vel[0], -vel[1])

    def getInputs(self):
        return {
            "p1": {
                "your_position":self.norm_pos(self.slime_left.pos), 
                "opponent_position":self.norm_pos(self.slime_right.pos), 
                "ball_position":self.norm_pos(self.ball.pos), 
                "ball_velocity":self.norm_vel(self.ball.vel)
            },
            "p2": {
                "your_position":self.norm_pos(self.slime_right.pos), 
                "opponent_position":self.norm_pos(self.slime_left.pos), 
                "ball_position":self.norm_pos(self.ball.pos), 
                "ball_velocity":self.norm_vel(self.ball.vel)
            }
        }

    def step(self, actions, keyboard={}, display=True, is_skip=False):
        self.timestep += 1
        if self.timestep < 20:
            return self.getInputs(), 0, 0

        ball_side = 2*int(self.ball.pos[0] > self.screen_size[0]/2) - 1
        
        # process skip frames
        if is_skip == False:
            for i in range(self.skip_frames):
                result = self.step(actions, keyboard=keyboard, display=False, is_skip=True)
                if result == -1:
                    ball_side = 2*int(self.ball.pos[0] > self.screen_size[0]/2) - 1
                    self.score[(1-ball_side)//2] += 1
                    return self.getInputs(), (ball_side, -ball_side), True

        # slime controls
        for i in range(2):
            action = actions[f"p{i+1}"]

            if action == "keyboard":
                match i:
                    case 0:
                        action = [0, 0]
                        if keyboard.get('w'): action[1] += 1
                        if keyboard.get('a'): action[0] -= 1
                        if keyboard.get('d'): action[0] += 1
                    case 1:
                        action = [0, 0]
                        if keyboard.get("ArrowUp"): action[1] += 1
                        if keyboard.get("ArrowLeft"): action[0] -= 1
                        if keyboard.get("ArrowRight"): action[0] += 1
            
            # verify action
            if not action[0] in {-1, 0, 1}:
                raise ValueError
            if not action[1] in {0, 1}:
                raise ValueError

            slime = [self.slime_left, self.slime_right][i]

            slime.vel[0] = slime.move_speed * action[0]
            if action[1] and slime.pos[1] == self.ground_level:
                slime.vel[1] = -slime.jump_height

            slime.pos = add(slime.pos, scale(slime.vel, self.dt))
            slime.vel = add(slime.vel, scale([0,slime.gravity], self.dt))

            # keep slime in bounds
            if slime.pos[0] < 0 + (400+self.net_width) * i + slime.radius:
                slime.pos[0] = 0 + (400+self.net_width) * i + slime.radius
            if slime.pos[0] > 400-self.net_width + (400+self.net_width) * i - slime.radius:
                slime.pos[0] = 400-self.net_width + (400+self.net_width) * i - slime.radius
                

            # if slime on floor
            if slime.pos[1] >= self.ground_level:
                slime.pos[1] = self.ground_level
                slime.vel[1] = 0

        # apply physics of closer slime last to override
        if self.ball.pos[0] < self.screen_size[0]: slimes = [self.slime_right, self.slime_left]
        else: slimes = [self.slime_left, self.slime_right]

        # detect collisions with slimes
        slime_hit = False # overrides game end
        for slime in slimes:
            # collision is when bottom of slime is below the ball and the slime and ball are close enough together
            if slime.pos[1] > self.ball.pos[1] and math.dist(slime.pos, self.ball.pos) < slime.radius + self.ball.radius:
                slime_hit = True
                angle = norm(subtract(self.ball.pos, slime.pos))
                new_vel = add(scale(angle, self.ball.bounce_vel), slime.vel)
                if new_vel[0] > self.ball.max_vel_x: new_vel[0] = self.ball.max_vel_x
                if new_vel[0] < -self.ball.max_vel_x: new_vel[0] = -self.ball.max_vel_x
                if new_vel[1] < -self.ball.max_vel_y: new_vel[1] = -self.ball.max_vel_y
                self.ball.vel = new_vel

        # all physics values are scaled by dt
        self.ball.pos = add(self.ball.pos, scale(self.ball.vel, self.dt)) # apply velocity
        self.ball.vel = add(self.ball.vel, scale([0,self.ball.gravity], self.dt)) # apply gravity, + is down

                
        # bounce off walls
        if self.ball.pos[0]-self.ball.radius <= 0:
            self.ball.vel[0] = -self.ball.vel[0]
            self.ball.pos[0] = 0 + self.ball.radius + 1
        if self.ball.pos[0]+self.ball.radius >= self.screen_size[0]:
            self.ball.vel[0] = -self.ball.vel[0]
            self.ball.pos[0] = self.screen_size[0] - self.ball.radius - 1

        # bounce off net
        if abs(self.ball.pos[0]-self.screen_size[0]/2) <= self.net_width+self.ball.radius and self.ball.pos[1] >= self.net_level-self.ball.radius:
            # if ball is going up then it hit the side
            if self.ball.vel[0] > 0: side = self.screen_size[0]/2 - self.net_width
            else: side = self.screen_size[0]/2 + self.net_width
            if self.ball.vel[1] < 0 and (self.ball.pos[1] - self.net_level) < abs(self.ball.pos[0] - side):
                self.ball.vel[0] = -self.ball.vel[0]
                self.ball.pos[0] = side + [-1,1][side>self.screen_size[0]/2] * self.ball.radius
            else:
                iy = self.ball.radius-abs(self.net_level - self.ball.pos[1])
                ix = self.ball.radius-abs(side - self.ball.pos[0])
                dx, dy = self.ball.vel
                if abs(ix/dx) < abs(iy/dy):
                    self.ball.vel[0] = -self.ball.vel[0]
                    self.ball.pos[0] = side + [-1,1][side>self.screen_size[0]/2] * self.ball.radius
                else:
                    self.ball.vel[1] = -self.ball.vel[1]
                    self.ball.pos[1] = self.net_level - self.ball.radius - 1

        if display: self.display()
            
        # if touching ground
        if not slime_hit and self.ball.pos[1]+self.ball.radius > self.ground_level:
            if is_skip: return -1
            else: 
                self.score[(1-ball_side)//2] += 1
                return self.getInputs(), (ball_side, -ball_side), True

        bs_scale = 0
        r1, r2 = bs_scale * ball_side, bs_scale * -ball_side
        new_ball_side = 2*int(self.ball.pos[0] > self.screen_size[0]/2) - 1
        if ball_side != new_ball_side:
            if ball_side < 0:
                r1 += 0.01
            else:
                r2 += 0.01
            
        return self.getInputs(), (r1, r2), False

    def getState(self):
        state = {
            "left": {
                "x": self.slime_left.pos[0],
                "y": self.slime_left.pos[1]
            },
            "right": {
                "x": self.slime_right.pos[0],
                "y": self.slime_right.pos[1]
            },
            "ball": {
                "x": self.ball.pos[0],
                "y": self.ball.pos[1]
            },
            "score": self.score
        }
        return state

    def display(self):
        # fill in bg
        img = np.array([[self.colors['bg']]], dtype=np.uint8)
        img = img.repeat(self.screen_size[1],axis=0).repeat(self.screen_size[0],axis=1)

        # display score
        if self.score != -1:
            for score, pos in zip(self.score, [(200,75),(600,75)]):
                text = str(score)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1.2
                color = (0, 0, 0)
                thickness = 2
                text_width, text_height = cv2.getTextSize(text, font, fontScale, thickness)[0]
                org = np.subtract(pos, (text_width/2, text_height/2))
                img = cv2.putText(img, text, (int(org[0]),int(org[1])), font,
                                  fontScale, color, thickness, cv2.LINE_AA)

        # draw net
        img = cv2.rectangle(img,
                           (self.screen_size[0]//2-self.net_width, self.net_level),
                           (self.screen_size[0]//2+self.net_width, self.ground_level),
                           self.colors['net'], thickness=-1)

        # fill in ground
        img = cv2.rectangle(img,
                            (0, self.ground_level),
                            self.screen_size,
                            self.colors['ground'], thickness=-1)

        # draw slimes as half circles
        for slime in [self.slime_left, self.slime_right]:
            img = cv2.ellipse(img,
                              turn_int(slime.pos),
                              [slime.radius, slime.radius],
                              0, 180, 360,
                              self.colors[('left','right')[slime.side]], thickness=-1)

        # draw ball
        img = cv2.circle(img, turn_int(self.ball.pos), self.ball.radius, self.colors['ball'], thickness=-1)

        cv2.imshow('img', img)
        cv2.waitKey(1)

    def close(self):
        if self.render_mode == "human":
            cv2.destroyWindow("img")

    

class Slime:
    def __init__(self, side):
        self.gravity = 10
        self.radius = 30
        self.side = side
        self.pos = [0,0]
        self.vel = [0,0]
        self.jump_height = 40
        self.move_speed = 30
        
        self.reset()

    def reset(self):
        self.pos = [200+400*self.side,320]
        self.vel = [0,0]


class Ball:
    def __init__(self):
        self.gravity = 10
        self.radius = 15
        self.pos = [0,0]
        self.vel = [0,0]
        self.bounce_vel = 50
        self.max_vel_x = 75
        self.max_vel_y = 50
        
        self.reset()

    def reset(self):
        side = random.randint(0,1)
        #side = 0
        self.pos = [200+400*side,200]
        self.vel = [0,0]


if __name__ == '__main__':
    import keyboard as k
    import time

    player1 = "keyboard"
    player2 = "keyboard"

    #from YourPyScript import Agent as player1
    #from YourPyScript import Agent as player2

    game = SlimeVolleyballEnv()
    game.reset()

    if player1 != "keyboard": player1 = player1()
    if player2 != "keyboard": player2 = player2()
    while True:
        inputs = game.getInputs()
        actions1 = [0,0]
        actions2 = [0,0]

        if player1 == "keyboard":
            if k.is_pressed('a'): actions1[0] -= 1
            if k.is_pressed('d'): actions1[0] += 1
            if k.is_pressed('w'): actions1[1] += 1
        else:
            actions1 = player1.getAction(inputs["p1"])
        if player2 == "keyboard":
            if k.is_pressed('left'): actions2[0] -= 1
            if k.is_pressed('right'): actions2[0] += 1
            if k.is_pressed('up'): actions2[1] += 1
        else:
            actions2 = player2.getAction(inputs["p2"])
        _, _, done = game.step({"p1":actions1, "p2":actions2})
        game.display()

        time.sleep(1/20)

        if done:
            game.reset()
        




        