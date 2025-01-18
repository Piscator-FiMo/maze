from enum import Enum
import random
import sys
from typing import List, Optional, Self, Tuple

import numpy as np
import matplotlib.pyplot as plt

# (0,0) is the top left corner


class Orientation(Enum):
    # North / Up
    NORTH = (0, -1)
    # East / Right
    EAST = (1, 0)
    # South / Down
    SOUTH = (0, 1)
    # West / Left
    WEST = (-1, 0)

    def opposite(self):
        opposites = {
            Orientation.NORTH: Orientation.SOUTH,
            Orientation.EAST: Orientation.WEST,
            Orientation.SOUTH: Orientation.NORTH,
            Orientation.WEST: Orientation.EAST
        }
        return opposites[self]

    @staticmethod
    def get_shuffled():
        values = list(Orientation)
        random.shuffle(values)
        return values


class Tile:

    def __init__(self, x, y, value='#') -> None:
        self.x = x
        self.y = y
        self.position = (x, y)
        self.value = value

    def get_neighbour(self, orientation: Orientation) -> Tuple[int, int]:
        x, y = self.position
        dx, dy = orientation.value
        return x + dx, y + dy

    def get_diagonal_neighbour(self, a: Orientation, b: Orientation) -> Tuple[int, int]:
        x, y = self.position
        if a == b or a == b.opposite():
            return x, y
        dx_a, dy_a = a.value
        dx_b, dy_b = b.value
        return x + dx_a + dx_b, y + dy_a + dy_b

    def get_all_neighbours_oriented(self, orientation: Orientation) -> List[Tuple[int, int]]:
        neighbours = []
        for orient in Orientation:
            if orient != orientation.opposite():
                neighbours.append(self.get_neighbour(orient))
                if orient != orientation:
                    neighbours.append(
                        self.get_diagonal_neighbour(orientation, orient))

        return neighbours

    def __str__(self) -> str:
        return f"{self.position} {self.value}"

    def __repr__(self) -> str:
        return self.__str__()


class Labyrinth:

    def __init__(self, columns: int = 10, rows: int = 10, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

        self.columns = columns
        self.rows = rows
        self.tiles = [[Tile(x, y, '#') for x in range(columns)] for y in range(rows)]
        self._recursion(self.get_tile_at(1, 1))

        # set start | ToDo: randomize
        shuffled_cols = [y for y in range(self.columns)]
        random.shuffle(shuffled_cols)
        for y in shuffled_cols:
            t = self.get_tile_at(0, y)
            n = t.get_neighbour(Orientation.EAST)
            n = self.get_tile_at(n[0], n[1])
            if n.value == '.':
                t.value = 'S'
                self.start = t
                break
        # set end
        shuffled_cols = [y for y in range(self.columns)]
        random.shuffle(shuffled_cols)
        for y in shuffled_cols:
            t = self.get_tile_at(columns - 1, y)
            n = t.get_neighbour(Orientation.WEST)
            n = self.get_tile_at(n[0], n[1])
            if n.value == '.':
                t.value = 'E'
                self.end = t
                break

    def get_tile_at(self, x, y) -> Tile:
        if self._is_out_of_bounds(x, y):
            return Tile(x, y, '#')
        return self.tiles[y][x]

    def is_wall_at(self, x, y) -> bool:
        return self._is_out_of_bounds(x, y) or self.get_tile_at(x, y).value == '#'

    def _is_out_of_bounds(self, x, y) -> bool:
        return not (x in range(0, self.columns) and y in range(0, self.rows))

    def get_array(self):
        return [[cell.value for cell in row] for row in self.tiles]

    def _recursion(self, tile: Tile):
        tile.value = '.'
        for orientation in Orientation.get_shuffled():
            next_tile = tile.get_neighbour(orientation)
            next_tile = self.tiles[next_tile[1]][next_tile[0]]
            if next_tile is not None:
                neighbours = next_tile.get_all_neighbours_oriented(orientation)
                usable = True
                for neighbour in neighbours:
                    if self._is_out_of_bounds(neighbour[0], neighbour[1]):
                        neighbour = None
                    else:
                        neighbour = self.get_tile_at(neighbour[0], neighbour[1])
                    if neighbour is None or neighbour.value != '#':
                        usable = False
                        break

                if usable:
                    self._recursion(next_tile)


def convert(symbol):
    match symbol:
        case '#':
            return 1
        case '.':
            return 0
        case 'S':
            return 2
        case 'E':
            return 3


def backup(rows):

    # rows = [[convert(cell) for cell in row.split(', ')]
    #       for row in maze.splitlines() if len(row) > 0]
    rows = [[convert(cell) for cell in row] for row in rows]

    # print(rows)
    a = np.array(rows)
    # print(a)
    plt.imshow(a, interpolation="nearest", origin="upper")
    # plt.colorbar()
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    l = Labyrinth(15, 25)
    [print(row) for row in l.tiles]
    backup(l.get_array())
