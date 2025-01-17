#Aufgabestellung:
# Einen Roboter  erstellen, der in einem Labyrinth selbständig vom Start- zum Endpunkt findet.

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Definiere das Labyrinth
        self.maze = [
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            ['#', 'S', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '#', '#', '#', '#', '#', '#', '#', '.', '#'],
            ['#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '.', '#', '#', '#', '.', '#', '#', '#', '#'],
            ['#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '#', '#', '#', '.', '#', '#', '#', '#', '#'],
            ['#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '#', '#', '#', '#', '#', '#', '#', 'E', '#'],
        ]

        self.start = (1, 1)  # Startposition (S)
        self.end = (8, 8)  # Endposition (E)
        self.current_position = self.start

        # Definiere die Aktionsräume (Bewegungen: oben, unten, links, rechts)
        self.action_space = spaces.Discrete(4)  # 0: oben, 1: unten, 2: links, 3: rechts

        # Definiere den Beobachtungsraum
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.maze), len(self.maze[0])), dtype=np.float32)

    def reset(self):
        self.current_position = self.start
        return self._get_observation()

    def step(self, action):
        # Bewege den Roboter basierend auf der Aktion
        x, y = self.current_position

        if action == 0:  # oben
            new_position = (x - 1, y)
        elif action == 1:  # unten
            new_position = (x + 1, y)
        elif action == 2:  # links
            new_position = (x, y - 1)
        elif action == 3:  # rechts
            new_position = (x, y + 1)

        # Überprüfe, ob die neue Position gültig ist
        if self._is_valid_move(new_position):
            self.current_position = new_position

        # Überprüfe, ob das Ziel erreicht wurde
        done = self.current_position == self.end
        reward = 1 if done else 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Erstelle eine Beobachtung des Labyrinths
        obs = np.zeros((len(self.maze), len(self.maze[0])), dtype=np.float32)
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == '#':
                    obs[i][j] = 1  # Wand
                elif (i, j) == self.current_position:
                    obs[i][j] = 0.5  # Aktuelle Position
                elif (i, j) == self.end:
                    obs[i][j] = 0.75  # Endposition
        return obs

    def _is_valid_move(self, position):
        x, y = position
        return (0 <= x < len(self.maze) and
                0 <= y < len(self.maze[0]) and
                self.maze[x][y] != '#')

    def render(self):
        # Visualisiere das Labyrinth
        plt.imshow(self._get_observation(), cmap='grey', vmin=0, vmax=1)
        plt.xticks([])  # Keine x-Achsen-Beschriftungen
        plt.yticks([])  # Keine y-Achsen-Beschriftungen
        plt.title("Labyrinth")
        plt.show()


# Beispiel für die Verwendung der Umgebung
if __name__ == "__main__":
    env = MazeEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Zufällige Aktion
        obs, reward, done, _ = env.step(action)
        env.render()  # Labyrinth nach jedem Schritt render