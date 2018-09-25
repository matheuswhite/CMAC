from random import random, randint
from cmac import State, CMAC, Dimension

'''
self.player_y = int(state['player_y'])
self.player_vel = int(state['player_velocity'])
self.cpu_y = int(state['cpu_y'])
self.ball_x = int(state['ball_x'])
self.ball_y = int(state['ball_y'])
self.ball_vel_x = int(state['ball_velocity_x'])
self.ball_vel_y = int(state['ball_velocity_y'])
'''


class PongState(State):

    def __init__(self, state, action_index):
        self.ver_distance = abs(int(state['ball_y']) - int(state['player_y']))
        self.ver_distance_signal = 1 if int(state['ball_y']) - int(state['player_y']) > 0 else 0
        self.hor_distance = int(state['ball_x'])
        self.ball_vel_x = 1 if int(state['ball_velocity_y']) > 0 else 0
        self.action_index = action_index

        super().__init__([self.ver_distance, self.ver_distance_signal, self.hor_distance, self.ball_vel_x, self.action_index])


class PongAgent:

    def __init__(self, action_set, learning_ratio=0.01, gama=0.5, epsilon=0.01, load_from_file=False):
        self.action_set = action_set
        self.learning_ratio = learning_ratio
        self.gama = gama
        self.epsilon = epsilon
        self.filename = 'pong_agent_cmac'

        offsets = [8*1, 0, 8*3, 0, 0]
        dimensions = [Dimension(tile_width=8, minn=0, maxx=48), Dimension(tile_width=1, minn=0, maxx=2),
                      Dimension(tile_width=8, minn=0, maxx=64), Dimension(tile_width=1, minn=0, maxx=2),
                      Dimension(tile_width=1, minn=0, maxx=3)]

        self.q_func = CMAC(offsets=offsets, dimensions=dimensions, n_tilings=4)

        if load_from_file:
            self.q_func.load_from_file(self.filename)

    def __get_action_index(self, action):
        for x in range(0, len(self.action_set)):
            if action == self.action_set[x]:
                return x
        return None

    def __choose_best_action_index(self, state):
        best_state = PongState(state, 0)
        best_reward = self.q_func.get_weight(best_state)

        for a in range(1, len(self.action_set)):
            current_state = PongState(state, a)
            current_reward = self.q_func.get_weight(current_state)

            if current_reward > best_reward:
                best_state = current_state
                best_reward = current_reward

        return best_state.action_index

    def pick_action(self, state):
        best_action_index = self.__choose_best_action_index(state)

        if random() < self.epsilon:
            best_action_index = randint(0, len(self.action_set)-1)

        return self.action_set[best_action_index]

    # TODO: Remake
    def update_q_function(self, state, action, reward, next_state):
        best_next_action_index = self.__choose_best_action_index(next_state)
        best_action_index = self.__get_action_index(action)

        pong_state = PongState(state, best_action_index)
        pong_next_state = PongState(next_state, best_next_action_index)

        now_q_func = self.q_func.get_weight(pong_state)
        next_q_func = self.q_func.get_weight(pong_next_state)
        expected_reward = reward + self.gama*next_q_func

        new_weight = now_q_func + self.learning_ratio*(expected_reward - now_q_func)
        self.q_func.set_weight(pong_state, new_weight)

    def save_to_file(self):
        self.q_func.save_to_file(self.filename)
