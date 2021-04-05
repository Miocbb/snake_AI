from collections import deque
import sys
import pygame
import numpy as np
import dnn
import pickle

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
    """

    def __init__(self, game_size, how='random', verbose=0, save_memory=True):
        self._width, self._height = game_size
        if self._width <= 2 or self._height <= 2:
            raise Exception('The game board is too small!')

        self._game_size = tuple(game_size)
        self._verbose = verbose
        self._save_memory = save_memory
        self._move_ops = (self.move_left, self.move_right,
                          self.move_down, self.move_up)
        self._init_states(how=how)
        self._original_head_xy = self._body[0]
        self._original_food_xy = self._food_xy

    def _init_states(self, how='random'):
        """Initialize all the states of the snake with default values.

        States of the snake refers to properties or attributes in the snake
        objects that can be modified along the game.
        """
        # ==> Part 1 <==
        # States related to the snake's body, the position of food and
        # game board.
        self._board = np.zeros(self._game_size)
        if how == 'random':
            body_x, body_y = self._next_random_avail_position()
            fx, fy = self._next_random_avail_position()
        elif how == 'zero':
            body_x, body_y = 0, 0
            fx, fy = self._width // 2, self._height // 2
        elif how == 'original':
            body_x, body_y = self._original_head_xy
            fx, fy = self._original_food_xy
        elif how == 'center':
            body_x, body_y = self._width // 2, self._height // 2
            fx, fy = body_x - 2, body_y - 2
        else:
            raise Exception('Implementation error')
        # body
        self._body = deque([(body_x, body_y)])
        self._body_in_tuple = self._body_to_tuple()
        # food
        self._food_xy = (fx, fy)
        # board
        self._board[body_x, body_y] = l_snake
        self._board[fx, fy] = l_food

        # ==> Part 2 <==
        # States related to monitoring and measuring the moving of the snake.

        # The record for all positions of the snake's body during the
        # exploration for one food.
        self._body_record = set()
        self._add_body_record()

        # number of turnings that the snakes made during the exploration to
        # eat current food.
        self._num_turns = 0
        self._total_num_turns = 0

        # number of moving steps during the exploration to find the current
        # food.
        self._num_steps = 0
        self._total_num_steps = 0

        # The status of the snakes. Is it dead or live? How does it die?
        self._status = 'live'

        # The length of the snake
        self._len = 1

        # ==> Part 3 <==
        # Other states
        self.memory = {}
        if self._save_memory:
            self.memory['init_board'] = self._board.copy()
            self.memory['steps'] = []

    def len(self):
        return self._len

    def status(self):
        return self._status

    def last_num_steps(self):
        return self._num_steps

    def total_num_steps(self):
        return self._total_num_steps

    def avg_num_steps(self):
        return self.total_num_steps() / self.len()

    def last_num_turns(self):
        return self._num_turns

    def total_num_turns(self):
        return self._total_num_turns

    def avg_num_turns(self):
        return self.total_num_turns() / self.len()

    def _body_to_tuple(self):
        return tuple(i for x in self._body for i in x)

    def _add_body_record(self):
        self._body_record.add(self._body_to_tuple())

    def _reset_body_record(self):
        self._body_record.clear()
        self._add_body_record()

    def _is_body_in_record(self):
        return self._body_in_tuple in self._body_record

    def _new_food(self):
        "Create a piece of food on the board."
        # make sure there is only one piece of food on board.
        fx, fy = self._food_xy
        if self._board[fx, fy] == l_food:
            self._board[fx, fy] == l_empty

        # create new food.
        fx, fy = self._next_random_avail_position()
        self._board[fx, fy] = l_food
        self._food_xy = (fx, fy)
        return fx, fy

    def _is_occupied(self, x, y):
        "Check if the specified position is occupied or not."
        return self._board[x, y] != l_empty

    def _next_random_avail_position(self):
        """Based on current game board, find the next available position
        randomly."""
        bx, by = self._board.shape
        x = np.random.randint(bx)
        y = np.random.randint(by)
        while (self._is_occupied(x, y)):
            x = np.random.randint(bx)
            y = np.random.randint(by)
        return x, y

    def _overwrite_board(self):
        "Overwrite the board based on current snake and food information."
        self._board = np.zeros_like(self._board)
        x, y = self._food_xy
        self._board[x, y] = l_food
        for x, y in self._body:
            self._board[x, y] = l_snake

    def print(self):
        "Print the game board in matrix form."
        print(self._board)

    def reset_board_random(self):
        """
        Randomly reset the game board, including resetting all the states of
        the snake and the position of food.
        """
        self._init_states(how='random')

    def reset_original_board(self):
        """
        Reset the game board to be original, including resetting all the states
        of the snake and the position of food.
        """
        self._init_states(how='original')
        self._body.clear()
        self._body.append(self._original_head_xy)
        self._body_in_tuple = self._body_to_tuple()
        self._food_xy = self._original_food_xy
        self._overwrite_board()

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
        white = 255, 255, 255
        red = (255, 0, 0)
        color_snake = white
        color_food = red

        screen_w, screen_h = screen.get_size()
        board_w, board_h = self._width, self._height
        unit_w, unit_h = screen_w // board_w, screen_h // board_h
        for i in range(board_w):
            for j in range(board_h):
                if self._board[i, j] == l_snake:
                    rect = [j * unit_h, i * unit_w, unit_w, unit_h]
                    pygame.draw.rect(screen, color_snake, rect, 1)
                elif self._board[i, j] == l_food:
                    rect = [j * unit_h, i * unit_w, unit_w, unit_h]
                    pygame.draw.rect(screen, color_food, rect)
        pygame.display.flip()

    def _move_one_step(self, new_head):
        # check if it is okay to move one step forward.
        x, y = new_head
        bd_x, bd_y = self._board.shape
        # check boundary of the game board
        if x < 0 or x >= bd_x:
            self._status = 'crush_on_wall'
            if self._verbose > 0:
                print(
                    f"Snake died: X-axis out-of-boundary: x={x}, bd_x={bd_x}")
            return False
        elif y < 0 or y >= bd_y:
            self._status = 'crush_on_wall'
            if self._verbose > 0:
                print(
                    f"Snake died: Y-axis out-of-boundary: Y={y}, bd_y={bd_y}")
            return False
        # check if the snake bites itself.
        elif self._board[x, y] == l_snake:
            self._status = 'bite_itself'
            if self._verbose > 0:
                print("Snake died: bite itself!")
            return False

        # Now it is okay to move one step forward.
        board_modification_record = []
        original_label = self._board[x, y]
        self._body.appendleft(new_head)
        self._board[x, y] = l_snake
        board_modification_record.append((new_head, l_snake))
        found_food = (original_label == l_food)
        if found_food:
            # update body in tuple representation.
            self._body_in_tuple = self._body_to_tuple()
            # increase states accordingly
            self._len += 1
            # generate new food and record the modification
            food_x, food_y = self._new_food()
            board_modification_record.append(((food_x, food_y), l_food))
            # reset states for the exploration of the eaten food.
            self._num_steps = 0
            self._num_turns = 0
            self._reset_body_record()
        else:
            # move one step.
            tail_x, tail_y = self._body.pop()
            self._board[tail_x, tail_y] = l_empty
            board_modification_record.append(((tail_x, tail_y), l_empty))
            # update body in tuple representation.
            self._body_in_tuple = self._body_to_tuple()
            # check if the snake is in a visited position.
            # If so, it may circle.
            if self._is_body_in_record():
                self._status = 'killed_by_circling'
                return False
            # update the states for the exploration of the current food.
            self._num_steps += 1
            self._add_body_record()

        # update total moving steps
        self._total_num_steps  += 1

        # update memory with all the operations to board in current step.
        if self._save_memory:
            self.memory['steps'].append(board_modification_record)

        if self._verbose > 0:
            print(f'step: {self._total_num_steps}\n{self._board}')

        return True

    def move_up(self):
        head = self._body[0]
        new_head = (head[0]-1, head[1])
        return self._move_one_step(new_head)

    def move_down(self):
        head = self._body[0]
        new_head = (head[0]+1, head[1])
        return self._move_one_step(new_head)

    def move_left(self):
        head = self._body[0]
        new_head = (head[0], head[1]-1)
        return self._move_one_step(new_head)

    def move_right(self):
        head = self._body[0]
        new_head = (head[0], head[1]+1)
        return self._move_one_step(new_head)

    def move(self):
        raise Exception("Don't know how to move: not implemented!")


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

    def __init__(self, *args, dnn_hidden_layers=[16, 16], **kargs):
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

        num_input_nodes = self._make_nn_input().size
        self._dnn_full_layers = [num_input_nodes] + dnn_hidden_layers + [4]
        self.brain = dnn.NN(self._dnn_full_layers)

    def _move_one_step(self, new_head):
        status = super(SnakeDNN, self)._move_one_step(new_head)
        if self._verbose > 0:
            self._vision()
        return status

    def _vision(self):
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
            rst[i, :] = self._vision_in_one_direction(direction)
        return rst

    def _vision_in_one_direction(self, direction):
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
            x, y = self._body[0]
            dx, dy = direction
            step = 0
            found = False
            while (x >= 0 and x < sx) and (y >= 0 and y < sy):
                x += dx
                y += dy
                step += 1
                if condition(self, (x, y)):
                    found = True
                    break

            if found:
                return step
            else:
                return -1

        def cond_reach_wall(self, xy):
            x, y = xy
            sx, sy = self._board.shape
            return x in [-1, sx] or y in [-1, sy]

        def cond_reach_food(self, xy):
            return xy == self._food_xy

        def cond_reach_body(self, xy):
            x, y = xy
            sx, sy = self._board.shape
            return ((x >= 0 and x < sx) and (y >= 0 and y < sy) and
                    (self._board[x, y] == l_snake) and (xy != self._body[0]))

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

    def _measure_wrt_food(self):
        x, y = self._body[0]
        fx, fy = self._food_xy
        a = np.array([x - fx, y - fy]) / np.array(self._board.shape)
        dist = np.linalg.norm(a) #/ np.linalg.norm(self._board.shape)
        a = a / dist
        a = list(a)
        a.append(dist)
        return a

    def _make_nn_input(self):
        x = self._vision()
        x = np.array(x).reshape(-1, 1)
        return x

    def reset_brain_random(self):
        "Reset the brain of snake randomly."
        self.brain = dnn.NN(self._dnn_full_layers)

    def move(self):
        """
        Move the snake one step forward based on DNN.

        Returns
        -------
        bool
            True for successful move, and False otherwise.
        """
        x = self._make_nn_input()
        prediction = self.brain.evaluate(x)
        choice = np.argmax(prediction)
        if self._verbose > 0:
            print(
                f'step: {self._total_num_steps} dnn prediction: {prediction.flatten()} choice: {choice}')
        return self._move_ops[choice]()

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()

    def load(self, path):
        raise Exception('Not implement load')


class SnakeDNN_v2(SnakeDNN):
    """
    Snake with deep neural network version 2.

    Compare to SnakeDNN (version 1), the version 2 of snake only has three
    choices to descide when it wants to move. It never goes back!
    """
    def __init__(self, *args, **kargs):
        super(SnakeDNN_v2, self).__init__(*args, **kargs)
        self._dnn_full_layers[-1] = 3
        self.brain = dnn.NN(self._dnn_full_layers)

        # ==> states of the snake <==
        # The direction to which the snake is facing.
        self._face_direction = (0, 1)

    def _vision_in_one_direction(self, direction):
        dirx, diry = direction
        x, y = self._body[0]
        fx, fy = self._food_xy
        distance = 1
        input = [0, 0, 0]
        food_found = False
        body_found = False
        while((x != 0) and (x != self._width-1) and (y != 0) and (y != self._height-1)):
            x, y = x + dirx, y + diry
            distance += 1
            if(not food_found and fx == x and fy == y):
                input[0] = 1
                food_found = True
            if(not body_found and self._board[x,y] == l_snake):
                input[1] = 1 / distance
                body_found = True
        input[2] = 1 / distance

        if self._verbose > 0:
            desc = self._vision_direction[direction]
            print(
                f'{desc : <12} food: {input[0]} body: {input[1]} wall: {input[2]}')
        return input

    def _nn_prediction(self):
        x = self._make_nn_input()
        prediction = self.brain.evaluate(x)
        choice = np.argmax(prediction)
        if choice != 0:
            self._num_turns += 1
            self._total_num_turns += 1
        if self._verbose > 0:
            print(
                f'step: {self._total_num_steps} input x: {x.reshape(8, -1)}')
            print(
                f'step: {self._total_num_steps} dnn prediction: {prediction.flatten()} choice: {choice}')

        # interpretation of the choice.
        # 0: keep straight
        # 1: go left
        # 2: go right
        decision = {}
        if self._face_direction == (-1, 0): # face up
            decision = {
                0: self.move_up,
                1: self.move_left,
                2: self.move_right,
            }
        elif self._face_direction == (1, 0): # face down
            decision = {
                0: self.move_down,
                1: self.move_right,
                2: self.move_left,
            }
        elif self._face_direction == (0, -1): # face left
            decision = {
                0: self.move_left,
                1: self.move_down,
                2: self.move_up,
            }
        elif self._face_direction == (0, 1): # face right
            decision = {
                0: self.move_right,
                1: self.move_up,
                2: self.move_down,
            }

        return decision[choice]

    def move(self):
        func_move = self._nn_prediction()
        t = {
            self.move_right: (0, 1),
            self.move_left: (0, -1),
            self.move_up: (-1, 0),
            self.move_down: (1, 0),
        }
        status = func_move()
        # update face direction
        self._face_direction = t[func_move]
        return status
