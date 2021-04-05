import game
import sys
import random
import pickle

if len(sys.argv) < 3:
    raise Exception('Need snake file and gif path')
file_s = sys.argv[1]
gif_path = sys.argv[2]
with open(file_s, 'rb') as f:
    s = pickle.load(f)

env_seed = s['env_seed']
snake = s['snake']
random.seed(env_seed)
snake.reset_original_board()
game.gui_param['time_lag'] = 0.0
game.run_game(snake, enable_restart=False, plot=True, gif_path=gif_path)
