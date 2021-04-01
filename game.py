import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import sys
import time
import random
import snake

# some constant values
black = 0, 0, 0
white = 255,255,255
green = (0, 255, 0)
red = (255,0, 0)
blue = (0, 0, 128)
color_snake = white
color_food = red
color_bg = black

def draw_board(screen, board):
    """
    Draw the game board in the screen.

    Parameters
    ----------
    screen : pygame.Surface
        The main screen of the game.
    board: snake.Board
        The game board.
    """
    screen_w, screen_h = screen.get_size()
    board_w, board_h = board.shape
    unit_w, unit_h = screen_w // board_w, screen_h // board_h
    for i in range(board_w):
        for j in range(board_h):
            if board.get_label(i, j) == snake.l_snake:
                rect = [j * unit_h, i * unit_w, unit_w, unit_h]
                pygame.draw.rect(screen, color_snake, rect, 1)
            elif board.get_label(i, j) == snake.l_food:
                rect = [j * unit_h, i * unit_w, unit_w, unit_h]
                pygame.draw.rect(screen, color_food, rect)
    pygame.display.flip()

def draw_game_over(screen, msg):
    """
    Draw the game over text in the screen.
    """
    # create a font object.
    # 1st parameter is the font file
    # which is present in pygame.
    # 2nd parameter is size of the font
    font = pygame.font.Font('freesansbold.ttf', 32)

    # create a text surface object,
    # on which text is drawn on it.
    game_over_msg = font.render(msg, True, green, blue)

    # create a rectangular object for the
    # text surface object
    game_over_msg_Rect = game_over_msg.get_rect()
    X, Y = screen.get_size()
    game_over_msg_Rect.center = (X // 2, Y // 2)
    screen.blit(game_over_msg, game_over_msg_Rect)
    pygame.display.flip()

def restart_game_by_key():
    """
    Read keyboard input to determine to restart the game or not.

    Returns
    -------
    restart : bool
        True indicates restarting and False otherwise.
    """
    # restart?
    running = True
    restart = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    restart = True
                elif event.key == pygame.K_n:
                    restart = False
                pygame.event.clear()
                running = False
                break
    return restart

def move_snake_by_key(snake):
    """
    Move the snake for one step forward based on the keyboard input.

    Parameters
    ----------
    snake : snake.Snake
        A snake object on the game board.
    """
    running = True
    status = True # successfully moved or not.
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    status = snake.move_up()
                elif event.key == pygame.K_DOWN:
                    status = snake.move_down()
                elif event.key == pygame.K_LEFT:
                    status = snake.move_left()
                elif event.key == pygame.K_RIGHT:
                    status = snake.move_right()
                running = False
    return status

def move_snake_random(snake):
    """
    Move the snake for one step forward randomly.

    Parameters
    ----------
    snake : snake.Snake
        A snake object on the game board.
    """
    op = [snake.move_left, snake.move_right, snake.move_down, snake.move_up]
    idx = random.randrange(4)
    return op[idx]()


def run_game(nx=10, ny=10, screen_size=(500, 500), plot=True,
             enable_restart=False,
             walk_type='random',
             time_lag=0.1,
             ):
    """
    Run the snake game.

    Parameters
    ----------
    nx : int
        The number of units in X-axis (width) of the game board.
    ny : int
        The number of units in Y-axis (height) of the game board.
    screen_size : (int, int)
        The dimension in pixel (weight, height) of the game display window.
    plot : bool, default=True
        Display the game window or not.
    enable_restart : bool, default=False
        Enable restarting the game or not.
    walk_type : str, options={'random', 'key', 'ai'}
        The type of the snake step forward.
        'random' : forward randomly
        'key' : forward based on keyboard input
        'ai' : forward with AI.
    time_lag : float
        The length of time lag to update the snake in second.


    Returns
    -------
    snake.Snake
        A snake that reaches to the end of the game.
    """
    # create the game board
    board = snake.Board((nx, ny))

    if plot:
        pygame.init()
        # create a pygame window
        screen = pygame.display.set_mode(screen_size)
        # set the pygame window name
        pygame.display.set_caption('Snake AI')
        # draw the screen
        draw_board(screen, board)

    running = True
    while running:
        # control the refresh speed.
        if time_lag > 0:
            time.sleep(abs(time_lag))

        if plot:
            screen.fill(color_bg)
            draw_board(screen, board)
        if walk_type == 'random':
            status = move_snake_random(board.snake)
        elif walk_type == 'key':
            status = move_snake_by_key(board.snake)
        elif walk_type == 'ai':
            raise Exception(f'Snake AI is not done yet.')
        else:
            raise Exception(f'Detect unsupported walk type of snake.')

        # the snake is dead. enable restart or not?
        if not status:
            if plot:
                msg = 'Game Over: restart? <y/n>'
                if not enable_restart:
                    msg = 'Game Over!'

                draw_game_over(screen, msg)
                if enable_restart:
                    running = restart_game_by_key()
                else:
                    break;
            else:
                break; # break the main game loop to exit.
    return board.snake


if __name__ == '__main__':
    snake = run_game(enable_restart=True, walk_type='random', plot=False)
    print(f'Snake score: {snake.score}')
    print(f'Snake length: {snake.len}')
    print(f'Snake moveing steps: {snake.step}')