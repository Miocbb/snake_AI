from collections import deque
from board import Board, l_snake, l_empty, l_food
import sys
import pygame
import numpy as np
import dnn


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

    def __init__(self, board, is_random=True, verbose=0, save_memory=True):
        self._is_random = is_random
        self._verbose = verbose
        self._save_memory = save_memory
        self._board = board
        self._init()

    def _init(self):
        # create snake body
        if self._is_random:
            body_x, body_y = self._board.next_random_avail_position()
        else:
            body_x, body_y = 0, 0
        self.body = deque([(body_x, body_y)])
        self.len = 1
        self.score = 0
        self.step = 0

        # update board label
        self._board.set_label(body_x, body_y, l_snake)

        # create food
        # Note: food must be created after snake body to avoid food-snake
        # collision.
        self.new_food()

        self.memory = {}
        if self._save_memory:
            self.memory['init_board'] = self._board.copy()
            self.memory['steps'] = []

        self._move_ops = (self.move_left, self.move_right,
                          self.move_down, self.move_up)

    def reset(self):
        """
        Reset the game board and the snake.

        The board is reset with `Board.reset` and snake is reset with
        re-initialization.
        """
        self._board.reset()
        self._init()

    def _forward(self, new_head):
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
        if self._save_memory:
            self.memory['steps'].append(record)

        if self._verbose > 0:
            print(f'step: {self.step}')
            self._board.print()
        return True

    def move_up(self):
        head = self.body[0]
        new_head = (head[0]-1, head[1])
        return self._forward(new_head)

    def move_down(self):
        head = self.body[0]
        new_head = (head[0]+1, head[1])
        return self._forward(new_head)

    def move_left(self):
        head = self.body[0]
        new_head = (head[0], head[1]-1)
        return self._forward(new_head)

    def move_right(self):
        head = self.body[0]
        new_head = (head[0], head[1]+1)
        return self._forward(new_head)

    def move(self):
        raise Exception("Don't know how to move: not implemented!")

    def new_food(self):
        x, y = self._board.next_random_avail_position()
        self._board.set_label(x, y, l_food)
        self._board.food_xy = (x, y)
        return x, y


class SnakeKeyboard(Snake):
    """
    A type of snake that supports reading moving instructions from the input
    of keyboard.
    """

    def __init__(self, *args, **kargs):
        super(SnakeKeyboard, self).__init__(*args, **kargs)

    def move(self):
        """
        Move the snake for one step forward based on the keyboard input.

        Returns
        -------
        bool
            True for successful move, and False otherwise.
        """
        running = True
        status = True  # successfully moved or not.
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        status = self.move_up()
                    elif event.key == pygame.K_DOWN:
                        status = self.move_down()
                    elif event.key == pygame.K_LEFT:
                        status = self.move_left()
                    elif event.key == pygame.K_RIGHT:
                        status = self.move_right()
                    running = False
        return status


class SnakeRandom(Snake):
    """
    A type of snake that supports moving randomly.
    """

    def __init__(self, *args, **kargs):
        super(SnakeRandom, self).__init__(*args, **kargs)

    def move(self):
        """
        Randomly move the snake for one step forward.

        Returns
        -------
        bool
            True for successful move, and False otherwise.
        """
        return self._move_ops[np.random.randint(4)]()


class SnakeDNN(Snake):
    """
    A type of snake with artificial intelligence based on deep neural network.

    Attributes
    ----------
    memory : dict
        The memory of how the snake plays on the board.
        'init_board' : board.Board
            The initial board of the game for this snake.
        'steps' : deque of list, deque([((x, y), val), ...])
            The modification to the board in each step.
    brain : dnn.NN
        The internal deep neural network of the snake.
    """

    def __init__(self, *args, dnn_hidden_layers=[250, 100], **kargs):
        """
        Parameters
        ----------
        dnn_hidden_layers: list
            The number of nodes in all the hidden layers, following the
            direction from input layer to output layer.
        """
        super(SnakeDNN, self).__init__(*args, **kargs)

        self._vision_direction = {
            (0, 1): 'right',
            (0, -1): 'left',
            (1, 0): 'down',
            (-1, 0): 'up',
            (1, 1): 'down_right',
            (1, -1): 'down_left',
            (-1, -1): 'up_left',
            (-1, 1): 'up_right',
        }
        self._vision_orderd_directions = sorted(
            list(self._vision_direction.keys()))

        num_input_nodes = self.vision().size
        self._dnn_full_layers = [num_input_nodes] + dnn_hidden_layers + [4]
        self.brain = dnn.NN(self._dnn_full_layers)

    def _forward(self, new_head):
        status = super(SnakeDNN, self)._forward(new_head)
        if self._verbose > 0:
            self.vision()
        return status

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
        if self._verbose > 0:
            print(
                f'{desc : <12} ({direction[0]:>2},{direction[1]:>2}): d_wall={rst[0] : > 2} d_body={rst[1] : > 2} d_food={rst[2] : > 2}')
        return rst

    def move(self):
        """
        Move the snake one step forward based on DNN.

        Returns
        -------
        bool
            True for successful move, and False otherwise.
        """
        x = self.vision()
        x = x.reshape(-1, 1)
        prediction = self.brain.evaluate(x)
        choice = np.argmax(prediction)
        if self._verbose > 0:
            print(
                f'step: {self.step} dnn prediction: {prediction.flatten()} choice: {choice}')
        return self._move_ops[choice]()
