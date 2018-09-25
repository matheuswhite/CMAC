import numpy as np
import math


class Dimension:

    def __init__(self, tile_width, minn, maxx):
        self.tile_width = tile_width
        self.min = minn
        self.max = maxx

        self.n_tiles = math.ceil((self.max - self.min)/self.tile_width)


class State:

    def __init__(self, values: list):
        self.values = values
        self.__current_index = 0

    def get_next(self):
        output = None
        if len(self.values) > 0:
            output = self.values[self.__current_index]
            self.__current_index += 1

            if self.__current_index >= len(self.values):
                self.__current_index = 0
        return output


class BoxTiling:

    def __init__(self, dimensions: list, offsets: list):
        self.dimensions = dimensions
        self.offsets = offsets
        self.state_space = np.zeros(tuple(d.n_tiles for d in self.dimensions))

    @staticmethod
    def __get_tile_index(state: float, dim: Dimension, offset: float):
        index = -1
        while state >= dim.min + offset:
            index += 1
            state -= dim.tile_width
        return index

    def __get_tile_all_indexes(self, state: State):
        indexes = []
        offsets = iter(self.offsets)
        for d in self.dimensions:
            index = self.__get_tile_index(state.get_next(), d, next(offsets))
            if index < 0:
                return None
            indexes.append(index)
        return indexes

    def update_tile(self, state: State, new_weight: float):
        indexes = self.__get_tile_all_indexes(state)
        if indexes:
            self.state_space[tuple(indexes)] = new_weight

    def get_tile_weight(self, state: State):
        indexes = self.__get_tile_all_indexes(state)
        weight = 0
        if indexes:
            weight = self.state_space[tuple(indexes)]
        return weight

    def save_to_file(self, filename: str):
        np.save(filename, self.state_space)

    def load_from_file(self, filename: str):
        self.state_space = np.load(filename)


class CMAC:

    def __init__(self, offsets: list, dimensions: list, n_tilings: int):
        self.tilings = []
        for x in range(n_tilings):
            self.tilings.append(BoxTiling(dimensions, offsets))

    def get_weight(self, state: State):
        weight = 0
        for tiling in self.tilings:
            weight += tiling.get_tile_weight(state)
        return weight

    def set_weight(self, state: State, weight: float):
        for tiling in self.tilings:
            tiling.update_tile(state, weight)

    def save_to_file(self, filename: str):
        index = 0
        for tiling in self.tilings:
            tiling.save_to_file(filename + str(index))
            index += 1

    def load_from_file(self, filename: str):
        index = 0
        for tiling in self.tilings:
            tiling.load_from_file(filename + str(index) + '.npy')
            index += 1
