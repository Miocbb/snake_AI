import snake
import game
import numpy as np
import pickle
import copy
import argparse
import random

# ==> Game settings <==

# size of the game.
game_size = (10, 10)

# ==> Population settings <==

# Number of environments. The environment means the sequence of positions of
# foods on the game board.
num_environment = 50
# number of generations per environment.
# For each environment , train a group of snakes generation by generation
# to make them evolve to learn the current environment.
num_generation = 150
# population of each generation.
population = 200

# ==> Evolution settings <==

# percentage of top ranked parents to the population.
top_parent_percentage = 0.20
# percentage of bottom ranked parents to the population.
bottom_parent_percentage = 0.10
# percentage of randomly generated new children to the population.
random_children_percentage = 0.00
# mutation rate, perturbation in [-n, n].
mutation_rate = 0.5
# percentage of paramters (weights and bias matrix elements).
mutation_percentage = 0.15

# ==> Neural Network settings <==

# number of nodes in the DNN hidden layers.
snake_DNN_layers = [8, 3]
#snake_DNN_layers = [10, 10]

total_parent_percentage = top_parent_percentage + bottom_parent_percentage
if total_parent_percentage > 1:
    raise Exception('Invalid percentage of parents.')


def create_snake():
    s = snake.SnakeDNN_v2(game_size, dnn_hidden_layers=snake_DNN_layers,
                          verbose=0, save_memory=True, how='center')
    s.is_run = False
    return s


def create_population():
    snakes = []
    for _ in range(population):
        s = create_snake()
        snakes.append(s)
    return snakes


def rank_snakes(snakes, reverse=True):
    """
    The snake with longer length, less number of turns in the last round,
    more number of steps in the last round, less averaged steps and
    less averaged turns will have a higher rank.
    """
    def rank_func(s): return (s.len(),
                              # -s.total_num_turns(),
                              # -s.avg_num_turns(),
                              -s.last_num_turns(),
                              s.last_num_steps(),
                              # s.avg_num_steps(),
                              # s.total_num_steps(),
                              )
    snakes.sort(key=rank_func, reverse=reverse)
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


def create_children(parent_snakes, n_children=None):
    """
    Let the parent snakes mate to generate children snakes.

    Parameters
    ----------
    n_children : int, default=None
        The number of children snakes to generate. Default number is
        `population * (1 + random_children_percentage) - len(parent_snakes)`.
    """
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
    if not n_children:
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
    """
    Mutate the input children snakes in place.
    """
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


def update_population(old_snakes, child_snakes):
    """
    Combine two groups of snakes and sorted them in order to select the top
    number (the size of population) of snakes.
    """
    all_snakes = old_snakes + child_snakes
    new_snakes = rank_snakes(all_snakes)[:population]
    return new_snakes


def run_snakes(snakes, env_seed=None):
    """
    Run the snakes on the game board.

    Notes
    -----
    All the snakes are running under the same conditions.
    """
    if not env_seed:
        env_seed = np.random.randint(100000)
    for s in snakes:
        if s.is_run:
            continue
        random.seed(env_seed)
        s.reset_original_board()
        game.run_game(s, enable_restart=False, plot=False)
        s.is_run = True
    random.seed()


def print_generation(n_env, n_gen, snakes):
    print('-' * 10)
    print(f'Environ: {n_env} Generation: {n_gen}')
    for i, s in enumerate(snakes[:5]):
        print(f'    Rank: {i}  Len: {s.len():<4} LastTurns: {s.last_num_turns():<4}',
              f' LastSteps: {s.last_num_steps():<4}',
              f' TotalSteps: {s.total_num_steps():<6} TotalTurns: {s.total_num_turns():<6}',
              f' AvgSteps: {s.avg_num_steps():<6.2f} AvgTurns: {s.avg_num_turns():<6.2f}')
    print('')


def genetic_algo():
    """
    Outer loop is the number of environments. Inner loop is the number
    generations. The snakes evolve on a certrain environment.
    """
    top_5_snakes = []
    best_snake = {}

    # create population
    current_snakes = create_population()

    # evolve the group
    for n_env in range(num_environment):
        # The random seed complete controls how the food is generated. So
        # it controls the change of the environment.
        env_seed = np.random.randint(10000)
        for s in current_snakes:
            s.is_run = False

        for n_gen in range(num_generation):
            # let snakes run for food.
            run_snakes(current_snakes, env_seed=env_seed)
            # select parents.
            parent_snakes = select_parents(current_snakes)
            # mate to generate children.
            child_snakes = create_children(parent_snakes)
            # mutate children to bring more diversity.
            mutate_children(child_snakes)
            # let children snakes run for food.
            run_snakes(child_snakes, env_seed=env_seed)
            # update the population to make next generation.
            current_snakes = update_population(parent_snakes, child_snakes)
            # print top some of snakes.
            if n_gen % 10 == 0 and n_gen != 0:
                print_generation(n_env, n_gen, current_snakes)
            # game.replay_game(
            #    current_snakes[0], repeat=False, title_addition=f'Generation {i}')

            current_best = current_snakes[0]
            if not best_snake or current_best.len() > best_snake['snake'].len():
                best_snake = {'env_seed': env_seed,
                              'snake': copy.deepcopy(current_best)}

        # save the top 5 snakes for current environment to history.
        top_5_snakes.append([copy.deepcopy(s) for s in current_snakes[:5]])

    # return last group of snakes, top 5 snakes and the best snake.
    return current_snakes, top_5_snakes, best_snake


def genetic_algo_2():
    """
    Inner loop is the number of environments. Outer loop is the number
    generations. Each generation of snakes learns multiple environments to
    evolve.
    """
    top_5_snakes = []
    best_snake = {}

    # create population
    current_snakes = create_population()

    # evolve the group
    for n_gen in range(num_generation):
        for s in current_snakes:
            s.is_run = False

        # -----------------------
        # start to let current generation of snakes to explore different
        # environments
        parent_snakes = []
        for n_env in range(num_environment):
            # The random seed complete controls how the food is generated. So
            # it controls the change of the environment.
            env_seed = np.random.randint(10000)

            # let snakes run for food.
            run_snakes(current_snakes, env_seed=env_seed)

            # select parents based on current environment and save them into
            # the main parent pool.
            parent_snakes_temp = select_parents(current_snakes)
            for s in parent_snakes_temp:
                parent_snakes.append(copy.deepcopy(s))

        # mate to generate children.
        child_snakes = create_children(
            parent_snakes, max(population, len(parent_snakes)))

        # mutate children to bring more diversity.
        mutate_children(child_snakes)

        # let new children and parent snakes run for food in a new environment for testing performance.
        env_seed = np.random.randint(10000)
        current_snakes = child_snakes + parent_snakes
        run_snakes(current_snakes, env_seed=env_seed)

        # update the population to make next generation based on their ranks.
        rank_snakes(current_snakes)
        current_snakes = current_snakes[:population]

        # print some of top snakes.
        print_generation(0, n_gen, current_snakes)
        # game.replay_game(
        #    current_snakes[0], repeat=False, title_addition=f'Generation {i}')

        # record the best snake so far.
        current_best = current_snakes[0]
        if not best_snake or current_best.len() > best_snake['snake'].len():
            best_snake = {'env_seed': env_seed,
                          'snake': copy.deepcopy(current_best)}

        # save the top 5 snakes for current environment to history.
        top_5_snakes.append([copy.deepcopy(s) for s in current_snakes[:5]])

    # return last group of snakes, top 5 snakes and the best snake.
    return current_snakes, top_5_snakes, best_snake


def save_snakes(snakes, filename):
    f = open(filename, 'wb')
    pickle.dump(snakes, f)
    f.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', help='relative path to save the snakes')
    ap.add_argument('--save_top5_history',
                    help='save all the top 5 snakes in each generation to file.')
    ap.add_argument('--algo_type', type=int, default=1,
                    help='type of genetic algorithms', choices=[1, 2])
    args = ap.parse_args()

    algo_func = None
    if args.algo_type == 2:
        algo_func = genetic_algo_2
    elif args.algo_type == 1:
        algo_func = genetic_algo_2
    else:
        raise Exception('Wrong algo type')

    last_snakes, top_5_snakes, best_snake = algo_func()
    if args.save_top5_history:
        save_snakes(top_5_snakes, args.save_top5_history)
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(best_snake, f)

    print(
        f'BestSnake: Len: {best_snake["snake"].len()} TotalSteps: {best_snake["snake"].total_num_steps()}')
    game.gui_param['time_lag'] = 0.1
    game.replay_game(best_snake['snake'])
