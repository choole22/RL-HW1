import random

from gym import Env
from gym import spaces
import pygame
import imageio.v2 as imageio
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class GoalVsHoleEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4, gamma=0.9, epsilon=1, epsilon_decay=1, epsilon_min=0.1):
        self.size = size
        self.window_size = 512
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self._agent_location = np.array([0, 0])
        self._agent_next_location = np.array([0, 0])
        self._target_location = np.array([2, 2])
        self._hole_locations = (np.array([1, 2]), np.array([2, 1]), np.array([3, 3]))
        self._q_table = [[0 for x in range(4)] for y in range(16)]
        self._prev_state = 0
        self._curr_state = 0
        self._next_state = 0
        self.total_goals = 0
        self.total_holes = 0

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0])  # Reset agent to space 0
        self._curr_state = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        self._agent_next_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if np.array_equal(self._agent_next_location, np.array([0, 0])):
            self._next_state = 0
        elif np.array_equal(self._agent_next_location, np.array([0, 1])):
            self._next_state = 4
        elif np.array_equal(self._agent_next_location, np.array([0, 2])):
            self._next_state = 8
        elif np.array_equal(self._agent_next_location, np.array([0, 3])):
            self._next_state = 12
        elif np.array_equal(self._agent_next_location, np.array([1, 0])):
            self._next_state = 1
        elif np.array_equal(self._agent_next_location, np.array([1, 1])):
            self._next_state = 5
        elif np.array_equal(self._agent_next_location, np.array([1, 2])):
            self._next_state = 9
        elif np.array_equal(self._agent_next_location, np.array([1, 3])):
            self._next_state = 13
        elif np.array_equal(self._agent_next_location, np.array([2, 0])):
            self._next_state = 2
        elif np.array_equal(self._agent_next_location, np.array([2, 1])):
            self._next_state = 6
        elif np.array_equal(self._agent_next_location, np.array([2, 2])):
            self._next_state = 10
        elif np.array_equal(self._agent_next_location, np.array([2, 3])):
            self._next_state = 14
        elif np.array_equal(self._agent_next_location, np.array([3, 0])):
            self._next_state = 3
        elif np.array_equal(self._agent_next_location, np.array([3, 1])):
            self._next_state = 7
        elif np.array_equal(self._agent_next_location, np.array([3, 2])):
            self._next_state = 11
        elif np.array_equal(self._agent_next_location, np.array([3, 2])):
            self._next_state = 15

        # An episode is done iff the agent has reached a goal or hole
        terminated = np.array_equal(self._agent_next_location, self._target_location) \
                     or np.array_equal(self._agent_next_location, self._hole_locations[0]) \
                     or np.array_equal(self._agent_next_location, self._hole_locations[1]) \
                     or np.array_equal(self._agent_next_location, self._hole_locations[2])

        if terminated and np.array_equal(self._agent_next_location, self._target_location):  # if in goal
            reward = 100
            self.total_goals += 1
        elif terminated:  # if in hole
            reward = -100
            self.total_holes += 1
        else:  # non-terminal state
            reward = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                256, 256,
                pix_square_size, pix_square_size
            )
        )

        # Hole, state 6
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                256, 130,
                pix_square_size, pix_square_size
            )
        )

        # Hole, state 9
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                130, 256,
                pix_square_size, pix_square_size
            )
        )

        # Hole, state 15
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                385, 385,
                pix_square_size, pix_square_size
            )
        )

        # print("Agent Loc: ", self._agent_location, type(self._agent_location))
        # self._agent_location += 1

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def print_q_table(self):
        q_table_rowsize = len(self._q_table)

        print('----------------------')
        print(f'{"Q-TABLE": ^21}')
        print('----------------------')
        print('State   #', f'{"(R, D, L, U)": >12}')
        print('----------------------')
        for row in range(q_table_rowsize):
            print(f'State:{row:3}', self._q_table[row])
        print('----------------------')

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def train(self, n=1000):
        episode = 1
        step = 0

        print('\nQ-Table before training:')
        env.print_q_table()

        for i in tqdm(range(n)):
            # Plot the previous state and save it as an image that
            # will be later patched together as a .gif
            img = plt.imshow(env.render())
            plt.title("Episode: {}, Step: {}".format(episode, step))
            plt.axis('off')
            plt.savefig("./temp/{}.png".format(i))
            plt.close()
            filenames.append("./temp/{}.png".format(i))

            rand = np.random.rand()
            while True:
                if rand < self.epsilon:
                    action_not_valid = True
                    while action_not_valid:
                        '''
                        0: right
                        1: down
                        2: left
                        3: up
                        '''
                        action = env.action_space.sample()
                        if self._curr_state == 0 and action in [0, 1]:
                            action_not_valid = False
                        elif self._curr_state == 3 and action in [1, 2]:
                            action_not_valid = False
                        elif self._curr_state == 12 and action in [0, 3]:
                            action_not_valid = False
                        elif self._curr_state == 15 and action in [2, 3]:
                            action_not_valid = False

                        elif self._curr_state in [1, 2] and action in [0, 1, 2]:
                            action_not_valid = False
                        elif self._curr_state in [4, 8] and action in [0, 1, 3]:
                            action_not_valid = False
                        elif self._curr_state in [7, 11] and action in [1, 2, 3]:
                            action_not_valid = False
                        elif self._curr_state in [13, 14] and action in [0, 2, 3]:
                            action_not_valid = False
                        elif self._curr_state in [5, 6, 9, 10]:
                            action_not_valid = False

                # exploitative action; consult q-table
                else:
                    # corners, restrict actions
                    if self._curr_state == 0:
                        if self._q_table[self._curr_state][0] > self._q_table[self._curr_state][1]:
                            action = 0
                        elif self._q_table[self._curr_state][0] == self._q_table[self._curr_state][1]:
                            if random.randrange(2) == 0:
                                action = 0
                            else:
                                action = 1
                        else:
                            action = 1

                    elif self._curr_state == 3:
                        if self._q_table[self._curr_state][1] > self._q_table[self._curr_state][2]:
                            action = 1
                        elif self._q_table[self._curr_state][1] == self._q_table[self._curr_state][2]:
                            if random.randrange(2) == 0:
                                action = 1
                            else:
                                action = 2
                        else:
                            action = 2

                    elif self._curr_state == 12:
                        if self._q_table[self._curr_state][0] > self._q_table[self._curr_state][3]:
                            action = 0
                        elif self._q_table[self._curr_state][0] == self._q_table[self._curr_state][3]:
                            if random.randrange(2) == 0:
                                action = 0
                            else:
                                action = 3
                        else:
                            action = 3

                    # non-corner side states
                    elif self._curr_state in [1, 2]:
                        acts = [0, 1, 2]
                        for a in range(2):
                            if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                                action = acts[a]
                            elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][
                                acts[a + 1]]:
                                if random.randrange(2) == 0:
                                    action = acts[a]
                                else:
                                    action = acts[a + 1]
                            else:
                                action = acts[a + 1]

                    elif self._curr_state in [4, 8]:
                        acts = [0, 1, 3]
                        for a in range(2):
                            if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                                action = acts[a]
                            elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][
                                acts[a + 1]]:
                                if random.randrange(2) == 0:
                                    action = acts[a]
                                else:
                                    action = acts[a + 1]
                            else:
                                action = acts[a + 1]

                    elif self._curr_state in [7, 11]:
                        acts = [1, 2, 3]
                        for a in range(2):
                            if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                                action = acts[a]
                            elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][
                                acts[a + 1]]:
                                if random.randrange(2) == 0:
                                    action = acts[a]
                                else:
                                    action = acts[a + 1]
                            else:
                                action = acts[a + 1]
                    elif self._curr_state in [13, 14]:
                        acts = [0, 2, 3]
                        if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                            action = acts[a]
                        elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][acts[a + 1]]:
                            if random.randrange(2) == 0:
                                action = acts[a]
                            else:
                                action = acts[a + 1]
                        else:
                            action = acts[a + 1]
                    else:
                        for j in range(3):
                            if self._q_table[self._curr_state][j] > self._q_table[self._curr_state][j + 1]:
                                action = j
                            elif self._q_table[self._curr_state][j] == self._q_table[self._curr_state][j + 1]:
                                if random.randrange(2) == 0:
                                    action = j
                                else:
                                    action = j + 1
                            else:
                                action = j + 1

                observation, reward, terminated, truncated = env.step(action)

                # update q-table
                self._q_table[self._curr_state][action] = reward + self.gamma * np.amax(self._q_table[self._next_state])

                # update agent_location
                self._agent_location = self._agent_next_location

                if self._prev_state != self._next_state:
                    break

            self._prev_state = self._curr_state
            if np.array_equal(self._agent_location, np.array([0, 0])):
                self._curr_state = 0
            elif np.array_equal(self._agent_location, np.array([0, 1])):
                self._curr_state = 4
            elif np.array_equal(self._agent_location, np.array([0, 2])):
                self._curr_state = 8
            elif np.array_equal(self._agent_location, np.array([0, 3])):
                self._curr_state = 12
            elif np.array_equal(self._agent_location, np.array([1, 0])):
                self._curr_state = 1
            elif np.array_equal(self._agent_location, np.array([1, 1])):
                self._curr_state = 5
            elif np.array_equal(self._agent_location, np.array([1, 2])):
                self._curr_state = 9
            elif np.array_equal(self._agent_location, np.array([1, 3])):
                self._curr_state = 13
            elif np.array_equal(self._agent_location, np.array([2, 0])):
                self._curr_state = 2
            elif np.array_equal(self._agent_location, np.array([2, 1])):
                self._curr_state = 6
            elif np.array_equal(self._agent_location, np.array([2, 2])):
                self._curr_state = 10
            elif np.array_equal(self._agent_location, np.array([2, 3])):
                self._curr_state = 14
            elif np.array_equal(self._agent_location, np.array([3, 0])):
                self._curr_state = 3
            elif np.array_equal(self._agent_location, np.array([3, 1])):
                self._curr_state = 7
            elif np.array_equal(self._agent_location, np.array([3, 2])):
                self._curr_state = 11
            elif np.array_equal(self._agent_location, np.array([3, 2])):
                self._curr_state = 15

            step += 1
            if terminated or truncated:
                episode += 1
                step = 0
                observation, info = env.reset()

            # update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def test(self, n=50):
        episode = 1
        step = 0
        self.total_holes = 0
        self.total_goals = 0

        for i in tqdm(range(n)):
            # Plot the previous state and save it as an image that
            # will be later patched together as a .gif
            img = plt.imshow(env.render())
            plt.title("Episode: {}, Step: {}\n Goals: {}, Holes: {}".format(episode, step, self.total_goals, self.total_holes))
            plt.axis('off')
            plt.savefig("./temp/{}.png".format(i))
            plt.close()
            filenames.append("./temp/{}.png".format(i))

            rand = np.random.rand()
            # random action
            if rand >= 0.9:
                action_not_valid = True
                while action_not_valid:
                    '''
                    0: right
                    1: down
                    2: left
                    3: up
                    '''
                    action = env.action_space.sample()
                    if self._curr_state == 0 and action in [0, 1]:
                        action_not_valid = False
                    elif self._curr_state == 3 and action in [1, 2]:
                        action_not_valid = False
                    elif self._curr_state == 12 and action in [0, 3]:
                        action_not_valid = False
                    elif self._curr_state == 15 and action in [2, 3]:
                        action_not_valid = False

                    elif self._curr_state in [1, 2] and action in [0, 1, 2]:
                        action_not_valid = False
                    elif self._curr_state in [4, 8] and action in [0, 1, 3]:
                        action_not_valid = False
                    elif self._curr_state in [7, 11] and action in [1, 2, 3]:
                        action_not_valid = False
                    elif self._curr_state in [13, 14] and action in [0, 2, 3]:
                        action_not_valid = False
                    elif self._curr_state in [5, 6, 9, 10]:
                        action_not_valid = False

            # exploitative action; consult q-table
            else:
                # corners, restrict actions
                if self._curr_state == 0:
                    if self._q_table[self._curr_state][0] > self._q_table[self._curr_state][1]:
                        action = 0
                    elif self._q_table[self._curr_state][0] == self._q_table[self._curr_state][1]:
                        if random.randrange(2) == 0:
                            action = 0
                        else:
                            action = 1
                    else:
                        action = 1

                elif self._curr_state == 3:
                    if self._q_table[self._curr_state][1] > self._q_table[self._curr_state][2]:
                        action = 1
                    elif self._q_table[self._curr_state][1] == self._q_table[self._curr_state][2]:
                        if random.randrange(2) == 0:
                            action = 1
                        else:
                            action = 2
                    else:
                        action = 2

                elif self._curr_state == 12:
                    if self._q_table[self._curr_state][0] > self._q_table[self._curr_state][3]:
                        action = 0
                    elif self._q_table[self._curr_state][0] == self._q_table[self._curr_state][3]:
                        if random.randrange(2) == 0:
                            action = 0
                        else:
                            action = 3
                    else:
                        action = 3

                # non-corner side states
                elif self._curr_state in [1, 2]:
                    acts = [0, 1, 2]
                    for a in range(2):
                        if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                            action = acts[a]
                        elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][
                            acts[a + 1]]:
                            if random.randrange(2) == 0:
                                action = acts[a]
                            else:
                                action = acts[a + 1]
                        else:
                            action = acts[a + 1]

                elif self._curr_state in [4, 8]:
                    acts = [0, 1, 3]
                    for a in range(2):
                        if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                            action = acts[a]
                        elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][
                            acts[a + 1]]:
                            if random.randrange(2) == 0:
                                action = acts[a]
                            else:
                                action = acts[a + 1]
                        else:
                            action = acts[a + 1]

                elif self._curr_state in [7, 11]:
                    acts = [1, 2, 3]
                    for a in range(2):
                        if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                            action = acts[a]
                        elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][
                            acts[a + 1]]:
                            if random.randrange(2) == 0:
                                action = acts[a]
                            else:
                                action = acts[a + 1]
                        else:
                            action = acts[a + 1]
                elif self._curr_state in [13, 14]:
                    acts = [0, 2, 3]
                    if self._q_table[self._curr_state][acts[a]] > self._q_table[self._curr_state][acts[a + 1]]:
                        action = acts[a]
                    elif self._q_table[self._curr_state][acts[a]] == self._q_table[self._curr_state][acts[a + 1]]:
                        if random.randrange(2) == 0:
                            action = acts[a]
                        else:
                            action = acts[a + 1]
                    else:
                        action = acts[a + 1]
                else:
                    for j in range(3):
                        if self._q_table[self._curr_state][j] > self._q_table[self._curr_state][j + 1]:
                            action = j
                        elif self._q_table[self._curr_state][j] == self._q_table[self._curr_state][j + 1]:
                            if random.randrange(2) == 0:
                                action = j
                            else:
                                action = j + 1
                        else:
                            action = j + 1

            observation, reward, terminated, truncated = env.step(action)

            # update agent_location
            self._agent_location = self._agent_next_location

            self._prev_state = self._curr_state
            if np.array_equal(self._agent_location, np.array([0, 0])):
                self._curr_state = 0
            elif np.array_equal(self._agent_location, np.array([0, 1])):
                self._curr_state = 4
            elif np.array_equal(self._agent_location, np.array([0, 2])):
                self._curr_state = 8
            elif np.array_equal(self._agent_location, np.array([0, 3])):
                self._curr_state = 12
            elif np.array_equal(self._agent_location, np.array([1, 0])):
                self._curr_state = 1
            elif np.array_equal(self._agent_location, np.array([1, 1])):
                self._curr_state = 5
            elif np.array_equal(self._agent_location, np.array([1, 2])):
                self._curr_state = 9
            elif np.array_equal(self._agent_location, np.array([1, 3])):
                self._curr_state = 13
            elif np.array_equal(self._agent_location, np.array([2, 0])):
                self._curr_state = 2
            elif np.array_equal(self._agent_location, np.array([2, 1])):
                self._curr_state = 6
            elif np.array_equal(self._agent_location, np.array([2, 2])):
                self._curr_state = 10
            elif np.array_equal(self._agent_location, np.array([2, 3])):
                self._curr_state = 14
            elif np.array_equal(self._agent_location, np.array([3, 0])):
                self._curr_state = 3
            elif np.array_equal(self._agent_location, np.array([3, 1])):
                self._curr_state = 7
            elif np.array_equal(self._agent_location, np.array([3, 2])):
                self._curr_state = 11
            elif np.array_equal(self._agent_location, np.array([3, 2])):
                self._curr_state = 15


            step += 1
            if terminated or truncated:
                episode += 1
                step = 0
                observation, info = env.reset()


'''
================================
'''
# number of timesteps
max_steps = 400

# Since we pass render_mode="human", you should see a window pop up rendering the environment.
render_mode = 'rgb_array'  # or, 'human'
env = GoalVsHoleEnv()
env.render_mode = render_mode
# Initialize empty buffer for the images that will be stiched to a gif
# Create a temp directory
filenames = []
try:
    os.mkdir("./temp")
except:
    pass

env.action_space.seed(42)
observation, info = env.reset(seed=42)

env.train(n=max_steps)
print("Q-Table after training...")
env.print_q_table()
# Stitch the images together to produce a .gif
# with imageio.get_writer('./GVH_Train.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)

# Cleanup the images for the next run
for f in filenames:
    os.remove(f)

''' TEST '''
filenames = []
env.action_space.seed(42)
env.test(n=100)
# Stitch the images together to produce a .gif
with imageio.get_writer('./GVH_Test.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup the images for the next run
for f in filenames:
    os.remove(f)

env.close()
