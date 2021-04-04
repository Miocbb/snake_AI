from numpy import random
from numpy.lib.arraysetops import union1d
from numpy.lib.function_base import _parse_input_dimensions
from numpy.lib.npyio import savez_compressed
import snake
import game
import numpy as np

generation = 50
population = 100

top_parent_percentage = 0.1
bottom_parent_percentage = 0.07
random_parent_percentage = 0.05
mutation_rate = 0.5
mutation_percentage = 0.1
snake_DNN_layers = [16, 16]
game_size = (10, 10)

total_parent_percentage = top_parent_percentage + \
    bottom_parent_percentage + random_parent_percentage
if total_parent_percentage > 1:
    raise Exception('Invalid percentage of parents.')



def create_snake():
    return snake.SnakeDNN(game_size, dnn_hidden_layers=snake_DNN_layers,
                          verbose=0, save_memory=True, how='center')


def create_population():
    snakes = []
    for _ in range(population):
        s = create_snake()
        snakes.append(s)
    return snakes


def rank_snakes(snakes, reverse=True):
    snakes.sort(key=lambda s: (s.score, -s.total_step), reverse=reverse)
    return snakes


def select_parents(snakes):
    "snakes are dead snakes that reach to end of the game."
    # count number of parents
    n_top_parents = int(population * top_parent_percentage)
    n_bottom_parents = int(population * bottom_parent_percentage)
    n_random_parents = int(population * random_parent_percentage)

    def find_idx(arr, num):
        for i in range(len(arr)):
            if arr[i] > num:
                return i

    def select(cum_scores):
        total_score = cum_scores[-1]
        rand_val = np.random.uniform(0, total_score)

        # linear search
        # TODO: binary search
        return find_idx(cum_scores, rand_val)

    # snakes with high scores have priority to mate.
    pool = rank_snakes(snakes)
    top_p = pool[:n_top_parents]
    bottom_p = pool[-n_bottom_parents:]
    random_p = [pool[np.random.randint(population)]
                for _ in range(n_random_parents)]
    parent_snakes = top_p + bottom_p + random_p

    #scores = [s.score for s in pool]
    #cum_scores = np.cumsum(np.array(scores)) + 1
    # for _ in range(n_random_parents):
    #    idx = select(cum_scores)
    #    parent_snakes.append(pool[idx])

    return parent_snakes


def create_children(parent_snakes):
    n_parents = len(parent_snakes)

    def select_random_parent():
        idx = np.random.randint(n_parents)
        return parent_snakes[idx]

    def create_child():
        p1 = select_random_parent()
        p2 = select_random_parent()

        game_snake = create_snake()

        l = len(p1.brain.W)
        for i in range(l):
            # crossover
            # for each coefficient element, choose it from p1 or p2.
            w1, b1 = p1.brain.W[i], p1.brain.b[i]
            w2, b2 = p2.brain.W[i], p2.brain.b[i]
            fw = np.random.randint(0, 2, size=w1.shape)
            fb = np.random.randint(0, 2, size=b1.shape)
            fw_neg = 1 - fw
            fb_neg = 1 - fb
            game_snake.brain.W[i] = fw * w1 + fw_neg * w2
            game_snake.brain.b[i] = fb * b1 + fb_neg * b2

            # linear crossover
            #game_snake.brain.W[i] = (p1.brain.W[i] + p2.brain.W[i]) / 2
            #game_snake.brain.b[i] = (p1.brain.b[i] + p2.brain.b[i]) / 2

        return game_snake

    # create children with mating.
    children = []
    n_children = population - n_parents
    for _ in range(n_children):
        c = create_child()
        children.append(c)

    return children


def mutate_children(child_snakes):
    def mutate_mat(m):
        np_m = np.array(m)
        n_mutations = int(mutation_percentage * np_m.size)
        if len(np_m.shape) == 2:
            row, col = np_m.shape
            for _ in range(n_mutations):
                i = np.random.randint(0, row)
                j = np.random.randint(0, col)
                m[i, j] += np.random.uniform(-mutation_rate, mutation_rate)
        elif len(np_m.shape) == 1:
            row = np_m.shape[0]
            for _ in range(n_mutations):
                i = np.random.randint(0, row)
                m[i] += np.random.uniform(-mutation_rate, mutation_rate)

    def mutate_child(s):
        for i in range(len(s.brain.W)):
            mutate_mat(s.brain.W[i])
            mutate_mat(s.brain.b[i])

    for s in child_snakes:
        mutate_child(s)


def update_population(old_snakes, child_snakes, env_seed):
    all_snakes = old_snakes + child_snakes
    run_snakes(child_snakes, env_seed=env_seed)
    new_snakes = rank_snakes(all_snakes)[:population]
    return new_snakes


def run_snakes(snakes, env_seed=None):
    if not env_seed:
        env_seed = np.random.randint(100000)
    for s in snakes:
        np.random.seed(env_seed)
        s.reset_original_board()
        game.run_game(s, enable_restart=False, plot=False)


def print_generation(generation, snakes):
    print('-' * 10)
    print('Generation: ', generation)
    for i, s in enumerate(snakes[:5]):
        print(f'    Rank: {i}  Score: {s.score} Steps: {s.total_step}')
    print('')


def genetic_algo():
    # create population
    current_snakes = create_population()
    env_seed = np.random.randint(10000)
    run_snakes(current_snakes, env_seed=env_seed)

    for i in range(generation):
        # mating process to give child population
        parent_snakes = select_parents(current_snakes)
        child_snakes = create_children(parent_snakes)
        mutate_children(child_snakes)
        current_snakes = update_population(
            parent_snakes, child_snakes, env_seed)

        print_generation(i, current_snakes)
        game.replay_game(
            current_snakes[0], repeat=False, title_addition=f'Generation {i}')

    return current_snakes


s = genetic_algo()[0]
# s.save_brain('opt_nn.pt')
print(f'best_score: {s.score} steps: {s.total_step}')
game.gui_param['time_lag'] = 0.5
game.replay_game(s)
