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

    def __init__(self, board, is_random=True, save_memory=True, verbose=0):
        self._board = board
        self.is_random = is_random
        self.save_memory = save_memory
        self.verbose = verbose
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

        self._vision_direction = {
            (0, 1) : 'right',
            (0, -1) : 'left',
            (1, 0) : 'down',
            (-1, 0) : 'up',
            (1, 1) : 'down_right',
            (1, -1) : 'down_left',
            (-1, -1) : 'up_left',
            (-1, 1) : 'up_right',
        }
        self._vision_orderd_directions = sorted(list(self._vision_direction.keys()))

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

        if self.verbose > 0:
            print(f'step: {self.step}\n{self._board._data}')
            self.vision()

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
        self._board.food_xy = (x, y)
        return x, y

    def vision(self):
        """
        The vision of the snake.

        The snake looks 8 directions, including (up, down, left, right,
        up_left, up_right, down_left, down_right). In each direction, the snake
        will measure 3 distances, including (head -> wall), (head -> body),
        and (head -> food).

        Returns
        -------
        numpy.array
            A 8 x 3 matrix that represents all the distances.
        """
        rst = np.zeros((8, 3))
        for i, direction in enumerate(self._vision_orderd_directions):
            rst[i, :] = self.vision_in_one_direction(direction)
        return rst

    def vision_in_one_direction(self, direction):
        """
        Parameters
        ----------
        direction : (int, int)
            The unit direction vector. For example direction=(0, 1) means
            the right direction, (1, 1) means the up_right direction and
            (-1, 0) means the left direction.

        Returns
        -------
        numpy.array
            A 1 x 3 array that represents the distance to the wall, body and
            food respectively.
        """
        def measure_dist(self, direction, condition):
            sx, sy = self._board.shape
            x, y = self.body[0]
            dx, dy = direction
            step = 0
            found = False
            while (x >= 0 and x < sx) and (y >= 0 and y < sy):
                if condition(self, (x, y)):
                    found = True
                    break
                else:
                    x += dx
                    y += dy
                    step += 1

            if found:
                rst = step
            else:
                rst = -1
            return rst

        def cond_reach_wall(self, xy):
            x, y = xy
            sx, sy = self._board.shape
            return x in [0, sx - 1] or y in [0, sy - 1]

        def cond_reach_food(self, xy):
            return xy == self._board.food_xy

        def cond_reach_body(self, xy):
            x, y = xy
            return (self._board._data[x, y] == l_snake) and (xy != self.body[0])

        if direction not in self._vision_direction:
            raise Exception('Detect wrong direction in vision.')

        d_wall = measure_dist(self, direction, cond_reach_wall)
        d_body = measure_dist(self, direction, cond_reach_body)
        d_food = measure_dist(self, direction, cond_reach_food)
        rst = np.array([d_wall, d_body, d_food])
        desc = self._vision_direction[direction]
        if self.verbose > 0:
            print(f'{desc : <12} ({direction[0]:>2},{direction[1]:>2}): d_wall={rst[0] : > 2} d_body={rst[1] : > 2} d_food={rst[2] : > 2}')
        return rst

class Board:
    """
    The game board class in the game snake.

    The game board is represented as a 2-d numpy.array (matrix).
    0 refers to empty position, 1 refers to snake, and 2 refers to snake food.
    """

    def __init__(self, shape):
        self._data = np.zeros(shape)
        self.food_xy = (-1, -1)
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
