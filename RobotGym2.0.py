import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MazeEnv(gym.Env):
  

    def __init__(self):
        super(MazeEnv, self).__init__()

        # Labyrinth-Layout
        # - '#' ist eine Wand (Hindernis).
        # - 'S' ist die Startposition.
        # - 'E' ist die Zielposition.
        # - '.' sind begehbare Felder.
        self.maze = [
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
            ['#', 'S', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '#', '#', '#', '#', '.', '#', '#', '#', '#', '#', '.', '#', '.', '#'],
            ['#', '.', '.', '.', '#', '.', '.', '.', '.', '#', '.', '.', '#', '.', '#'],
            ['#', '.', '#', '.', '#', '#', '#', '.', '#', '#', '#', '.', '#', '.', '#'],
            ['#', '.', '#', '.', '.', '.', '.', '.', '.', '.', '#', '.', '#', '.', '#'],
            ['#', '.', '#', '#', '#', '#', '#', '#', '#', '.', '#', '.', '#', '.', '#'],
            ['#', '.', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '#', '.', '#'],
            ['#', '#', '#', '#', '#', '#', '.', '#', '#', '#', '#', '.', '#', '.', '#'],
            ['#', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '#', '.', '#'],
            ['#', '.', '#', '#', '.', '#', '#', '#', '#', '#', '#', '#', '#', '.', '#'],
            ['#', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', 'E', '#'],
        ]

        # Start- und Zielposition
        self.start = (1, 1)  # Startposition des Spielers
        self.end = (12, 13)  # Zielposition des Spiels
        self.current_position = self.start

        # Aktionsraum (Bewegungen: oben, unten, links, rechts)
        self.action_space = spaces.Discrete(4)  # 0: oben, 1: unten, 2: links, 3: rechts

        # Beobachtungsraum (2D-Matrix, die den Zustand des Labyrinths darstellt)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.maze), len(self.maze[0])), dtype=np.float32
        )

    def reset(self):
        """
        Setzt die Umgebung zurück.
        - Der Spieler wird auf die Startposition gesetzt.
        - Die initiale Beobachtung wird zurückgegeben.
        """
        self.current_position = self.start
        return self._get_observation()

    def step(self, action):
        """
        Führt eine Aktion aus und aktualisiert den Zustand der Umgebung.
        - Berechnet die neue Position basierend auf der Aktion.
        - Überprüft, ob die Bewegung gültig ist (kein Hindernis, innerhalb der Grenzen).
        - Aktualisiert die Position des Spielers, falls die Bewegung gültig ist.
        - Überprüft, ob das Ziel erreicht wurde (done = True).
        - Gibt die neue Beobachtung, die Belohnung, den `done`-Status und zusätzliche Infos zurück.

        Args:
            action (int): Die gewählte Aktion (0 = oben, 1 = unten, 2 = links, 3 = rechts).

        Returns:
            tuple: (observation, reward, done, info)
        """
        x, y = self.current_position
        new_position = {
            0: (x - 1, y),  # oben
            1: (x + 1, y),  # unten
            2: (x, y - 1),  # links
            3: (x, y + 1),  # rechts
        }.get(action, self.current_position)

        # Überprüfen, ob die neue Position gültig ist
        if self._is_valid_move(new_position):
            self.current_position = new_position

        # Überprüfen, ob das Ziel erreicht wurde
        done = self.current_position == self.end
        reward = 1 if done else 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """
        Erstellt eine Beobachtungsmatrix, die den Zustand des Labyrinths darstellt.
        - `1` für Wände.
        - `0.25` für die Startposition.
        - `0.75` für die Zielposition.
        - `0.5` für die aktuelle Position des Spielers.
        - `0` für freie Felder.

        Returns:
            np.ndarray: Die Beobachtungsmatrix des aktuellen Zustands.
        """
        obs = np.zeros((len(self.maze), len(self.maze[0])), dtype=np.float32)
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == '#':
                    obs[i][j] = 1  # Wand
        x, y = self.current_position
        obs[x][y] = 0.5  # Aktuelle Position
        sx, sy = self.start
        ex, ey = self.end
        obs[sx][sy] = 0.25  # Startposition
        obs[ex][ey] = 0.75  # Zielposition
        return obs

    def _is_valid_move(self, position):
        """
        Überprüft, ob eine Bewegung gültig ist.
        - Die Position muss innerhalb der Grenzen des Labyrinths liegen.
        - Die Position darf kein Hindernis (`#`) sein.

        Args:
            position (tuple): Die zu überprüfende Position.

        Returns:
            bool: True, wenn die Bewegung gültig ist, sonst False.
        """
        x, y = position
        return (
            0 <= x < len(self.maze)
            and 0 <= y < len(self.maze[0])
            and self.maze[x][y] != '#'
        )

    def render(self):
        """
        Visualisiert das Labyrinth
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                color = 'black' if cell == '#' else 'white'
                if (i, j) == self.start:
                    color = 'red'
                elif (i, j) == self.end:
                    color = 'green'
                elif (i, j) == self.current_position:
                    color = 'blue'

                ax.add_patch(
                    patches.Rectangle((j, len(self.maze) - i - 1), 1, 1, color=color)
                )

        ax.set_xlim(0, len(self.maze[0]))
        ax.set_ylim(0, len(self.maze))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

# Hauptprogramm: Beispielverwendung der Umgebung
if __name__ == "__main__":
    env = MazeEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Zufällige Aktion
        obs, reward, done, _ = env.step(action)
        env.render()
