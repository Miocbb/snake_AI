import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import sys
import time
import snake
import board

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

def read_key_restart_game():
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
                    running = False
                    break
                elif event.key == pygame.K_n:
                    restart = False
                    running = False
                    break
    return restart

def replay_game(snake, repeat=True):
    """
    Replay the game.

    Parameters
    ----------
    snake : snake.Snake
    """
    if not snake.memory:
        raise Exception('Fail to replay the game of snake. Input snake has not memory.')

    original_board = snake.memory['init_board']
    steps = snake.memory['steps']

    pygame.init()
    # create a pygame window
    screen = pygame.display.set_mode(gui_param['screen_size'])
    # set the pygame window name
    pygame.display.set_caption('Snake AI Replay')

    running = True
    while running:
        # initialize and draw board
        board = original_board.copy()
        screen.fill(color_bg)
        draw_board(screen, board)

        # replay the game step by step
        for record in steps:
            # handle exit from keyboard.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break

            # update board
            for (x, y), v in record:
                board.set_label(x, y, v)

            # draw the screen
            screen.fill(color_bg)
            draw_board(screen, board)

            # add time lag between each step.
            time.sleep(gui_param['time_lag'])

        # draw game over message and pause for 1 second.
        draw_game_over(screen, 'Game Over!')
        time.sleep(1)

        # repeat the replay?
        if not repeat:
            running = False

gui_param = {
    # The dimension in pixel (weight, height) of the game display window.
    'screen_size': (500, 500),
    'time_lag': 0.1
}

def run_game(snake, plot=True, enable_restart=False):
    """
    Run the snake game.

    Parameters
    ----------
    snake : snake.Snake
        A snake in the game.
    plot : bool, default=True
        Display the game window or not.
    enable_restart : bool, default=False
        Enable restarting the game or not.
    verbose : int, default=0
        The the print level.

    Returns
    -------
    snake.Snake
        A snake that reaches to the end of the game.
    """
    if plot:
        pygame.init()
        # create a pygame window
        screen = pygame.display.set_mode(gui_param['screen_size'])
        # set the pygame window name
        pygame.display.set_caption('Snake AI')
        # draw the screen
        snake.draw(screen)

    running = True
    while running:
        # control the refresh speed.
        if plot and gui_param['time_lag'] > 0:
            time.sleep(abs(gui_param['time_lag']))
        if plot:
            screen.fill(color_bg)
            snake.draw(screen)
        status = snake.move()

        # the snake is dead. enable restart or not?
        if not status:
            if plot:
                msg = 'Game Over: restart? <y/n>'
                if not enable_restart:
                    msg = 'Game Over!'

                draw_game_over(screen, msg)
                if enable_restart:
                    if read_key_restart_game():
                        snake.reset()
                    else:
                        running = False
                else:
                    running = False;
            else:
                running = False;
    return snake


def create_game(mode='key', game_size=(10, 10), verbose=1):
    game_board = board.Board(game_size)
    if mode == 'key':
        game_snake = snake.SnakeKeyboard(game_board, verbose=verbose)
    elif mode == 'dnn':
        game_snake = snake.SnakeDNN(game_board, verbose=verbose)
    elif mode == 'random':
        game_snake = snake.SnakeRandom(game_board, verbose=verbose)

    return game_board, game_snake


if __name__ == '__main__':
    game_board, game_snake = create_game(mode='random')
    rst_snake = run_game(game_snake, enable_restart=True, plot=True)

    gui_param['time_lag'] = 0.5
    replay_game(rst_snake)
    #print(f'Snake score: {rst_snake.score}')
    #print(f'Snake length: {rst_snake.len}')
    #print(f'Snake moveing steps: {rst_snake.step}')
    #print(f'Snake init board:\n{rst_snake.memory["init_board"]}')
    #print(f'Snake final board:\n{rst_snake._board._data}')
    #print(f'Snake steps memory: {rst_snake.memory["steps"]}')
