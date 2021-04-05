import snake
import game
import numpy as np
import pickle
import copy

# number of generations.
generation = 100
# number of generations.
# For each generation, train it on the same environment for multiple times to
# give them time to evolve.
num_training_per_generation = 40
# population of each generation.
population = 200

# percentage of top ranked parents to the population.
top_parent_percentage = 0.20
# percentage of bottom ranked parents to the population.
bottom_parent_percentage = 0.05
# percentage of randomly generated new children to the population.
random_children_percentage = 0.10
# mutation rate, perturbation in [-n, n].
mutation_rate = 0.2
# percentage of paramters (weights and bias matrix elements).
mutation_percentage = 0.08

# number of nodes in the DNN hidden layers.
snake_DNN_layers = [8]

# size of the game.
game_size = (10, 10)

total_parent_percentage = top_parent_percentage + bottom_parent_percentage
if total_parent_percentage > 1:
    raise Exception('Invalid percentage of parents.')


def create_snake():
    return snake.SnakeDNN_v2(game_size, dnn_hidden_layers=snake_DNN_layers,
                          verbose=0, save_memory=True, how='center')


def create_population():
    snakes = []
    for _ in range(population):
        s = create_snake()
        snakes.append(s)
    return snakes


def rank_snakes(snakes, reverse=True):
    #snakes.sort(key=lambda s: (s.len, -s._current_step - s._num_turns), reverse=reverse)
    #snakes.sort(key=lambda s: (s.len, -s._num_turns), reverse=reverse)
    #snakes.sort(key=lambda s: (s.len, -s._current_step), reverse=reverse)
    snakes.sort(key=lambda s: (s.len, -s._num_turns, s._current_step, -s.total_step/s.len), reverse=reverse)
    return snakes

def select_parents(snakes):
    """Select parents based on their rankings.

    Notes
    -----
    1. The top and bottom parents are selected to form the group of parents.
    2. `snakes` are in sorted and increasing order.
    """
    # count number of parents
    n_top_parents = int(population * top_parent_percentage)
    n_bottom_parents = int(population * bottom_parent_percentage)

    # snakes with high scores have priority to mate.
    pool = rank_snakes(snakes)
    top_p = pool[:n_top_parents]
    bottom_p = pool[-n_bottom_parents:]
    parent_snakes = top_p + bottom_p

    return parent_snakes


def create_children(parent_snakes):
    # Make sure the random generator is sed randomly. There are following
    # reasons:
    # 1. We may need to create a group of randomly generated children,
    # according to `random_children_percentage`, to improve the group diversity.
    # 2. We need to randomly select pairs of parents to mate.
    np.random.seed()
    n_parents = len(parent_snakes)

    def select_random_parent():
        idx = np.random.randint(n_parents)
        return parent_snakes[idx]

    def create_child_from_mating():
        p1 = select_random_parent()
        p2 = select_random_parent()
        game_snake = create_snake()
        l = len(p1.brain.W)
        for i in range(l):
            # crossover
            w1, b1 = p1.brain.W[i], p1.brain.b[i]
            w2, b2 = p2.brain.W[i], p2.brain.b[i]
            # for each coefficient element, choose it from p1 or p2.
            fw = np.random.randint(0, 2, size=w1.shape)
            fb = np.random.randint(0, 2, size=b1.shape)
            fw_neg = 1 - fw
            fb_neg = 1 - fb
            game_snake.brain.W[i] = fw * w1 + fw_neg * w2
            game_snake.brain.b[i] = fb * b1 + fb_neg * b2
        return game_snake

    # create children with mating.
    children = []
    n_children = population - n_parents
    for _ in range(n_children):
        c = create_child_from_mating()
        children.append(c)

    # create children randomly
    n_new_children = int(population * random_children_percentage)
    for _ in range(n_new_children):
        c = create_snake()
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


def update_population(old_snakes, child_snakes, env_seed=None):
    all_snakes = old_snakes + child_snakes
    run_snakes(all_snakes, env_seed=env_seed)
    new_snakes = rank_snakes(all_snakes)[:population]
    return new_snakes


def run_snakes(snakes, env_seed=None):
    if not env_seed:
        env_seed = np.random.randint(100000)
    for s in snakes:
        np.random.seed(env_seed)
        s.reset_original_board()
        game.run_game(s, enable_restart=False, plot=False)
    np.random.seed()


def print_generation(generation, sub_gen, snakes):
    print('-' * 10)
    print(f'Generation: {generation} Sub-gen: {sub_gen}')
    for i, s in enumerate(snakes[:5]):
        print(f'    Rank: {i}  Len: {s.len:<4} LastSteps: {s._current_step:<4} LastTurns: {s._num_turns:<10}')
    print('')


def genetic_algo():
    # create population
    current_snakes = create_population()
    env_seed = np.random.randint(10000)
    run_snakes(current_snakes, env_seed=env_seed)

    top_5_snakes = []

    for gen in range(generation):
        env_seed = np.random.randint(10000)
        for sub_gen in range(num_training_per_generation):
            # mating process to give child population
            parent_snakes = select_parents(current_snakes)
            child_snakes = create_children(parent_snakes)
            mutate_children(child_snakes)
            current_snakes = update_population(parent_snakes, child_snakes, env_seed=env_seed)

            print_generation(gen, sub_gen, current_snakes)
            #game.replay_game(
            #    current_snakes[0], repeat=False, title_addition=f'Generation {i}')
        top_5_snakes.append([copy.deepcopy(s) for s in current_snakes[:5]])

    return current_snakes, top_5_snakes


def save_snakes(top_5_snakes, filename):
    f = open(filename, 'wb')
    pickle.dump(top_5_snakes, f)
    f.close()

s, top_snakes = genetic_algo()
save_snakes(top_snakes, 'save/test_5.pickle')
print(f'best_score: {s[0].len} steps: {s[0]._current_step}')
game.gui_param['time_lag'] = 0.1
game.replay_game(s[0])
