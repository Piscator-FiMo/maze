from typing import Optional
import numpy as np
import gymnasium as gym
import pygame
from Tools.demo.sortvisu import steps


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size + 2
        self.render_mode = "invisible"
        self.window_size= 500
        self.window = None
        self.clock = None
        self.metadata = {"render_fps": 10}
        self.steps = 0
        self.pillar_locations = [np.array([2, 3]), np.array([2, 4]), np.array([4, 3]), np.array([4, 2])]

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.render_mode = options["render_mode"]
        self.steps = 0
        # Choose the agent's location uniformly at random

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(1, self.size-1, size=2, dtype=int)
        while self._has_pillar_collision(self._agent_location):
            self._agent_location = self.np_random.integers(1, self.size-1, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location) or self._has_pillar_collision(self._target_location):
            self._target_location = self.np_random.integers(
                1, self.size-1, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _has_pillar_collision(self, pos: np.array):
        for pillar in self.pillar_locations:
            if np.array_equal(pos, pillar):
                return True
        return False

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        new_location = self._agent_location + direction
        self._agent_location = np.clip(
            new_location, 1, self.size - 2
        )
        if not np.array_equal(self._agent_location, new_location):
            hit = True

        #if agent hits a pillar, reset to previous position
        if self._has_pillar_collision(self._agent_location):
            hit = True
            self._agent_location = self._agent_location - direction

        # An environment is completed if and only if the agent has reached the target
        reached_target = np.array_equal(self._agent_location, self._target_location)

        self.steps += 1
        truncated = False
        if self.steps > 10000:
            truncated = True
            print("truncated")
        if reached_target:
            print(f"reached target after {self.steps} steps")
        terminated = reached_target
        reward = self.calculate_reward(reached_target, self.steps)  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def calculate_reward(self, reached_target: bool, steps_taken: int):
        if reached_target:
            return 10000
        return -1

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


        #draw border
        self.draw_border_rect(canvas, pix_square_size=pix_square_size, width=self.window_size, height=pix_square_size,
                              position=np.array([0, 0], dtype=np.int32))
        self.draw_border_rect(canvas, pix_square_size=pix_square_size, width=pix_square_size, height=self.window_size,
                              position=np.array([0, 0], dtype=np.int32))
        self.draw_border_rect(canvas, pix_square_size=pix_square_size, width=pix_square_size, height=self.window_size,
                              position=np.array([self.size-1, 0], dtype=np.int32))
        self.draw_border_rect(canvas, pix_square_size=pix_square_size, width=self.window_size, height=pix_square_size,
                              position=np.array([0, self.size-1], dtype=np.int32))

        # Pillars
        for pillar in self.pillar_locations:
            pygame.draw.rect(
                canvas,
                (50, 50, 50),
                pygame.Rect(
                    pix_square_size * pillar,
                    (pix_square_size, pix_square_size),
                ),
            )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
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

    def draw_border_rect(self, canvas, pix_square_size, width, height, position):
        pygame.draw.rect(
            canvas,
            (50, 50, 50),
            pygame.Rect(
                pix_square_size * position,
                (width, height),
            ),
        )