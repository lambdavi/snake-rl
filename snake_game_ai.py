import random
from collections import namedtuple, deque
from enum import Enum

import numpy as np
import pygame

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0, 153, 0)

BLOCK_SIZE = 20
SPEED = 60

class SnakeGameAI:
    """
        This is an agent controlled game.
    """
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.cycle = deque(maxlen=20)
        self.reset()
        

    def reset(self):
        """
            Init or reset the game state.
        """
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        # keep track of frame iteration
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def distance_from_food(self):
        dist = ((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)**(0.5)
        return dist
    
    def check_for_cycles(self):
        moves = dict()
        for move in self.cycle:
            if moves.get(move):
                moves[move] += 1
            else:
                moves[move] = 1
        
        for v in moves.values():
            if v >= 5:
                print("cycle detected")
                self.cycle.clear()
                return True
        return False
    
    def play_step(self, action):
        # 0. update the time step
        self.frame_iteration += 1

        initial_distance = self.distance_from_food()
        # 1. collect user input only to check if we want to stop the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move: update the head of the snake
        self._move(action)
        self.snake.insert(0, self.head)
        final_distance = self.distance_from_food()

        self.cycle.append(self.head)

        # 3. check if game over
        # set reward to return to the agent
        reward = 0
        game_over = False
        # check if collided or if the snake doesn't do anything for too long (longer the snake, more time the app has)
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

            if final_distance < initial_distance:
                reward += 0.1
            else:
                reward += -0.1

            if self.check_for_cycles():
                reward -= 2
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    
    def is_collision(self, point=None):
        """
            Check for collision. if 'point' is None it just checks if head collided, 
            otherwise we can use it to know if danger is nearby passing a point different from the head.
        """
        if point is None:
            point = self.head

        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        pt = self.head
        pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        """
            Action is a list of the form: [0/1, 0/1, 0/1]:
            - [1, 0, 0] = go straight
            - [0, 1, 0] = right turn
            - [0, 0, 1] = left turn
        """

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        direction_index = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[direction_index]
        elif np.array_equal(action, [0, 1, 0]):
            next_direction_index = (direction_index + 1) % 4 # mod 4 used for start again from 0 after 3
            new_direction = clock_wise[next_direction_index] # r -> d -> l -> u -> r ..
        else:
            next_direction_index = (direction_index - 1) % 4 
            new_direction = clock_wise[next_direction_index] # r -> u -> l -> d -> r ..

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)