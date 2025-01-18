from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

from Labyrinth import Labyrinth, convert


class GridWorldEnv(gym.Env):

    def __init__(self, labyrinth: Labyrinth):
        self.labyrinth = labyrinth

        # The size of the square grid
        self.render_mode = "invisible"
        self.window = None
        self.clock = None
        self.metadata = {"render_fps": 5}
        self.steps = 0

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        low = np.array([0, 0])
        high = np.array([self.labyrinth.columns - 1, self.labyrinth.rows - 1])
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low=low, high=high, shape=(2,), dtype=np.int32),
                "target": gym.spaces.Box(low=low, high=high, shape=(2,), dtype=np.int32),
                "neighbours": gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32),
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
        def get_location(delta):
            location = self._agent_location + delta
            tile = self.labyrinth.get_tile_at(location[0], location[1])
            return convert(tile.value)

        neighbours = np.array([get_location(delta) for delta in self._action_to_direction.values()], dtype=np.int32)
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "neighbours": neighbours
        }

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

        self.labyrinth.regenerate_start_and_end()
        self._agent_location = np.array(self.labyrinth.start.position, dtype=np.int32)
        self._target_location = np.array(self.labyrinth.end.position, dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        new_location = self._agent_location + direction
        hit = False

        # if agent hits a wall, don't move
        if self.labyrinth.is_wall_at(new_location[0], new_location[1]):
            hit = True
            new_location = self._agent_location

        self._agent_location = new_location

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
        cell_size = 50
        width = self.labyrinth.columns * cell_size
        height = self.labyrinth.rows * cell_size
        size = (width, height)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(size)
        # The size of a single grid square in pixels
        canvas.fill((0, 0, 0))

        # Finally, add some gridlines
        for rows in self.labyrinth.tiles:
            for tile in rows:
                color = None
                if tile.value == '.':
                    color = (255, 255, 255)
                elif tile.value == 'S':
                    color = (0, 0, 255)
                elif tile.value == 'E':
                    color = (0, 255, 0)
                if color:
                    pygame.draw.rect(canvas, color, (cell_size * tile.x + 1, cell_size *
                                     tile.y - 1, cell_size - 2, cell_size - 2))

        # Now we draw the agent
        pygame.draw.circle(canvas, (255, 0, 0), (self._agent_location + 0.5) * cell_size, cell_size / 3)

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


if __name__ == "__main__":
    lab = Labyrinth(15, 10)
    env = GridWorldEnv(lab)
    env.reset(options={"render_mode": "human"})
    while True:
        env._render_frame()
