import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

l_empty = 0
l_snake = 1
l_food = 2

class Board(object):
    """
    The game board class in the game snake.

    The game board is represented as a 2-d numpy.array (matrix).
    0 refers to empty position, 1 refers to snake, and 2 refers to snake food.
    """
    def __init__(self, shape):
        self._data = np.zeros(shape)
        self.food_xy = (-1, -1)
        self.shape = tuple(shape)

    def set_label(self, x, y, label):
        self._data[x, y] = label

    def get_label(self, x, y):
        return self._data[x, y]

    def is_occupied(self, x, y):
        return self._data[x, y] != l_empty

    def reset(self):
        self._data = np.zeros(self.shape)

    def __str__(self):
        return self._data.__str__()

    def print(self):
        print(self._data)

    def next_random_avail_position(self):
        bx, by = self.shape
        x = np.random.randint(bx)
        y = np.random.randint(by)
        while (self.is_occupied(x, y)):
            x = np.random.randint(bx)
            y = np.random.randint(by)
        return x, y

    def copy(self):
        t = Board(self.shape)
        t._data = self._data.copy()
        return t

    def draw(self, screen):
        """
        Draw the game board in the screen.

        Parameters
        ----------
        screen : pygame.Surface
            The main screen of the game.
        board: snake.Board
            The game board.
        """
        # some constant values
        white = 255,255,255
        red = (255,0, 0)
        color_snake = white
        color_food = red

        screen_w, screen_h = screen.get_size()
        board_w, board_h = self.shape
        unit_w, unit_h = screen_w // board_w, screen_h // board_h
        for i in range(board_w):
            for j in range(board_h):
                if self.get_label(i, j) == l_snake:
                    rect = [j * unit_h, i * unit_w, unit_w, unit_h]
                    pygame.draw.rect(screen, color_snake, rect, 1)
                elif self.get_label(i, j) == l_food:
                    rect = [j * unit_h, i * unit_w, unit_w, unit_h]
                    pygame.draw.rect(screen, color_food, rect)
        pygame.display.flip()
