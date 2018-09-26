import os
from PyGameLearningEnvironment.ple.games import  Pong
from PyGameLearningEnvironment.ple import PLE
from pong_agent import PongAgent
import matplotlib.pyplot as plt
import numpy as np

show_game = True

if not show_game:
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

game = Pong()

p = PLE(game, fps=30, display_screen=show_game, force_fps=not show_game)
p.init()

myAgent = PongAgent(p.getActionSet()[::-1], load_from_file=True, gama=0.9, learning_ratio=0.01, epsilon=0.01)

nb_frames = 10000
k = 1
old_diff = 0
diff = 0
reward = 0
rewards = []

try:
    for f in range(1000):
        print('Episode {}'.format(k))
        k += 1
        total_reward = 0
        for f in range(nb_frames):
            if p.game_over(): #check if the game is over
                p.reset_game()
                break

            # get current state
            state = game.getGameState()
            # pick action based on q_function and current state
            action = myAgent.pick_action(state)
            # apply action and get reward
            p.act(action)
            diff = state['player_y'] - state['ball_y']
            if diff == 0:
                reward = 10
            elif diff < old_diff:
                reward = 5
            elif diff > old_diff:
                reward = -20
            old_diff = diff
            total_reward += reward
            # get next state
            next_state = game.getGameState()
            # update q_function with state, next_state, action and reward
            myAgent.update_q_function(state, action, reward, next_state)

        rewards.append(total_reward)
finally:
    plt.plot(np.array(list(range(len(rewards)))), np.array(rewards))
    plt.show()
    myAgent.save_to_file()
    pass
