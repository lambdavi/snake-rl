from helper import plot
from agent import Agent
from snake_game_ai import SnakeGameAI

import random

def train():
    """
        Function to orchestrate all the training
    """
    # utilities for plotting
    plot_scores = []
    plot_mean_scores = []

    # variables
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    # Create the loop
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move based on the current state
        final_move = agent.get_action(state_old)

        # perform the move
        reward, game_over, score = game.play_step(final_move)
        # get new state
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train experience replay memory: train on all previous moves
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"--- Game: {agent.n_games} - Score: {score} - Record: {record} ---")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    print("Game started.")
    train()
    print("Game finished")