import numpy as np
import random
from collections import deque

l_empty = 0
l_snake = 1
l_food = 2

def rand_xy(bx, by):
    x = random.randrange(bx)
    y = random.randrange(by)
    return x, y

class Snake:
    """
    The snake class in the game snake.

    Attributes
    ----------
    body : collectons.deque([int, int])
        The positions of the snake body. The positions are in (x, y)
        corrdinates.
    len : int
        The body length of the snake.
    score : int
        The score of the snake achieved.
    step : int
        The number of steps that the snake achieved. This quantity reflects
        the lifetime of the snake.
    """
    def __init__(self, board, is_random=True):
        self._board = board

        # create snake body
        if is_random:
            x, y = rand_xy(*(self._board.shape))
            while (board.is_occupied(x, y)):
                x, y = rand_xy(*(self._board.shape))
        else:
            x, y = 0, 0
        self.body = deque([(x, y)])
        self.len = 1
        self.score = 0
        self.step = 0

        # update board label
        self._board.set_label(x, y, l_snake)

    def _foward(self, new_head):
        # check is new head okay?
        x, y = new_head
        bd_x, bd_y = self._board.shape
        # check boundary of the game board
        if x < 0 or x >= bd_x:
            print(f"Snake died: X-axis out-of-boundary: x={x}, bd_x={bd_x}")
            return False
        elif y < 0 or y >= bd_y:
            print(f"Snake died: Y-axis out-of-boundary: Y={y}, bd_y={bd_y}")
            return False
        # check if snake collide himself.
        elif self._board.get_label(x, y) == l_snake:
            print("Snake died: bite itself!")
            return False

        # Now it is okay to move one step forward.
        original_label = self._board.get_label(x, y)
        self.body.appendleft(new_head)
        self._board.set_label(x, y, l_snake)
        if original_label == l_food:
            # find food and grow the snake by 1 unit.
            self.len += 1
            # let the board to generate new food
            self._board.new_food()
            # increase the score.
            self.score += 1
        else:
            # no growing, just move one step.
            tail_x, tail_y = self.body.pop()
            self._board.set_label(tail_x, tail_y, l_empty)

        # update moving steps
        self.step += 1
        return True

    def move_up(self):
        head = self.body[0]
        new_head = (head[0]-1, head[1])
        return self._foward(new_head)

    def move_down(self):
        head = self.body[0]
        new_head = (head[0]+1, head[1])
        return self._foward(new_head)

    def move_left(self):
        head = self.body[0]
        new_head = (head[0], head[1]-1)
        return self._foward(new_head)

    def move_right(self):
        head = self.body[0]
        new_head = (head[0], head[1]+1)
        return self._foward(new_head)


class Board:
    """
    The game board class in the game snake.

    The game board is represented as a 2-d numpy.array (matrix).
    0 refers to empty position, 1 refers to snake, and 2 refers to snake food.
    """
    def __init__(self, shape):
        self._data = np.zeros(shape)
        self.shape = shape
        self.snake = Snake(self)
        self.new_food()

    def set_label(self, x, y, label):
        self._data[x, y] = label

    def get_label(self, x, y):
        return self._data[x, y]

    def is_occupied(self, x, y):
        return self._data[x, y] != 0

    def new_food(self):
        while 1:
            x, y = rand_xy(*(self.shape))
            if not self.is_occupied(x, y):
                self._data[x, y] = l_food
                break

    def reset(self):
        self._data = np.zeros(self.shape)
        self.snake = Snake(self)
        self.new_food()

    def print(self):
        print(self._data)
