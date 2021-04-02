import numpy as np
import random
from collections import deque

l_empty = 0
l_snake = 1
l_food = 2


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
    memory : dict
        The memory of how the snake plays on the board.
        'init_board' : Board
            The initial board of the game for this snake.
        'steps' : deque of list, deque([((x, y), val), ...])
            The modification to the board in each step.
    """

    def __init__(self, board, is_random=True, save_memory=True):
        self._board = board
        self.is_random = is_random
        self.save_memory = save_memory
        self._init()

    def _init(self):
        # create snake body
        if self.is_random:
            body_x, body_y = self._board.next_random_xy()
        else:
            body_x, body_y = 0, 0
        self.body = deque([(body_x, body_y)])
        self.len = 1
        self.score = 0
        self.step = 0
        self.memory = {}

        # update board label
        self._board.set_label(body_x, body_y, l_snake)

        # create food
        # Note: food must be created after snake body to avoid food-snake
        # collision.
        self.new_food()

        if self.save_memory:
            t = Board(self._board.shape)
            t._data = self._board._data.copy()
            self.memory['init_board'] = t
            self.memory['steps'] = []

    def reset(self):
        self._board.reset()
        self._init()

    def _foward(self, new_head):
        # check if it is okay to move one step forward.
        x, y = new_head
        bd_x, bd_y = self._board.shape
        # check boundary of the game board
        if x < 0 or x >= bd_x:
            print(f"Snake died: X-axis out-of-boundary: x={x}, bd_x={bd_x}")
            return False
        elif y < 0 or y >= bd_y:
            print(f"Snake died: Y-axis out-of-boundary: Y={y}, bd_y={bd_y}")
            return False
        # check if the snake bites itself.
        elif self._board.get_label(x, y) == l_snake:
            print("Snake died: bite itself!")
            return False

        # Now it is okay to move one step forward.
        record = []
        original_label = self._board.get_label(x, y)
        self.body.appendleft(new_head)
        self._board.set_label(x, y, l_snake)
        record.append((new_head, l_snake))
        if original_label == l_food:
            # find food and grow the snake by 1 unit.
            self.len += 1
            # let the board to generate new food
            food_x, food_y = self.new_food()
            record.append(((food_x, food_y), l_food))
            # increase the score.
            self.score += 1
        else:
            # no growing, just move one step.
            tail_x, tail_y = self.body.pop()
            self._board.set_label(tail_x, tail_y, l_empty)
            record.append(((tail_x, tail_y), l_empty))

        # update moving steps
        self.step += 1

        # update memory with all the operations to board in current step.
        if self.save_memory:
            self.memory['steps'].append(record)

        #print(f'step: {self.step}\n{self._board._data}')
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

    def new_food(self):
        x, y = self._board.next_random_xy()
        self._board.set_label(x, y, l_food)
        return x, y


class Board:
    """
    The game board class in the game snake.

    The game board is represented as a 2-d numpy.array (matrix).
    0 refers to empty position, 1 refers to snake, and 2 refers to snake food.
    """

    def __init__(self, shape):
        self._data = np.zeros(shape)
        self.shape = shape

    def set_label(self, x, y, label):
        self._data[x, y] = label

    def get_label(self, x, y):
        return self._data[x, y]

    def is_occupied(self, x, y):
        return self._data[x, y] != 0

    def reset(self):
        self._data = np.zeros(self.shape)

    def print(self):
        print(self._data)

    def next_random_xy(self):
        bx, by = self.shape
        x = random.randrange(bx)
        y = random.randrange(by)
        while (self.is_occupied(x, y)):
            x = random.randrange(bx)
            y = random.randrange(by)
        return x, y
