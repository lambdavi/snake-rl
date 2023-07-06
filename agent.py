import os
import random
from collections import deque

import numpy as np
import torch

from model import Linear_QNet, QTrainer
from snake_game_ai import Direction, Point

# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        # Parameter to control the randomness
        self.eps = 0
        # Discount rate
        self.gamma = 0.9
        # If we exceed the memory deque pop from the left, very handy
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Set model and trainer
        self.model = Linear_QNet(11, 256, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        """
        We calculate the state from the game. The state is composed by 11 values.
        """

        # Get the snake head point
        head = game.snake[0]

        # Get points nearby
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        point_ru = Point(head.x+ 20, head.y - 20)
        point_lu = Point(head.x - 20, head.y - 20)
        point_ld = Point(head.x - 20, head.y + 20)
        point_rd = Point(head.x + 20, head.y + 20)


        # Get the direction we are moving towards (one-hot encoding like)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        """# Danger diagonal straight
            (dir_r and (game.is_collision(point_ru) or dir_r and game.is_collision(point_rd))) or
            (dir_l and (game.is_collision(point_lu) or dir_l and game.is_collision(point_ld))),

            # Danger diagonal left
            (dir_l and (game.is_collision(point_ld) or game.is_collision(point_rd))) or
            (dir_r and (game.is_collision(point_lu) or game.is_collision(point_ru))),

            # Danger diagonal right
            (dir_r and (game.is_collision(point_rd) or game.is_collision(point_ld))) or
            (dir_l and (game.is_collision(point_ru) or game.is_collision(point_lu))),"""
        """
        (dir_r and (game.is_collision(point_r) or game.is_collision(point_ru) or game.is_collision(point_rd))) or
            (dir_l and (game.is_collision(point_l) or game.is_collision(point_lu) or game.is_collision(point_ld))) or
            (dir_u and (game.is_collision(point_u) or game.is_collision(point_ru) or game.is_collision(point_lu))) or
            (dir_d and (game.is_collision(point_d) or game.is_collision(point_ld) or game.is_collision(point_rd))),

            # Danger right
            (dir_u and (game.is_collision(point_r) or game.is_collision(point_ru) or game.is_collision(point_rd))) or
            (dir_r and (game.is_collision(point_d) or game.is_collision(point_rd) or game.is_collision(point_ld))) or
            (dir_l and (game.is_collision(point_u) or game.is_collision(point_ru) or game.is_collision(point_lu))) or
            (dir_d and (game.is_collision(point_l) or game.is_collision(point_lu) or game.is_collision(point_ld))),

            # Danger left
            (dir_l and (game.is_collision(point_d) or game.is_collision(point_ld) or game.is_collision(point_rd))) or
            (dir_r and (game.is_collision(point_u) or game.is_collision(point_lu) or game.is_collision(point_ru))) or
            (dir_u and (game.is_collision(point_l) or game.is_collision(point_lu) or game.is_collision(point_ld))) or
            (dir_d and (game.is_collision(point_r) or game.is_collision(point_ru) or game.is_collision(point_rd))),
        """
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move direction
            dir_l, 
            dir_r,
            dir_u, 
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, gameover):
        # pop left if max memory is reached
        self.memory.append((state, action, reward, next_state, gameover))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE) # list of BATCH_SIZE tuples
        else:
            batch = self.memory

        states, actions, rewards, next_states, game_overs = zip(*batch)
        self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(game_overs))
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        self.eps = 80 - self.n_games # random function
        final_move = [0,0,0]
        if random.randint(0, 200) < self.eps:
            # increasing n_games we don't get moves anymore
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() 
            final_move[move] = 1
        
        return final_move
    
