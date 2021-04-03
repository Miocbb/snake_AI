import snake
import board
import dnn
import game
import numpy as np

generation = 100
population = 1000
mating_rate = 0.5
mutation_rate = 0.1

game_size = (10, 10)

snake_DNN_layers = [250, 250, 100, 10]


def run_one_snake(snake):
    game.run_game(snake, plot=False, enable_restart=False)
    return snake


def run_snakes(snakes):
    for i, s in enumerate(snakes):
        #print(f'run snake: {i}')
        run_one_snake(s)


def create_population():
    snakes = []
    for _ in range(population):
        b = board.Board(game_size)
        s = snake.SnakeDNN(b, dnn_hidden_layers=snake_DNN_layers, verbose=0)
        snakes.append(s)
    return snakes


def fitness(snake):
    return snake.score


def select_parents(snakes):
    "snakes are dead snakes that reach to end of the game."
    pool = sorted([(fitness(s), s) for s in snakes], key=lambda x: x[0])
    scores,  _ = zip(*pool)
    cum_scores = np.cumsum(np.array(scores)) + 1

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

    n_parents = int(population * mating_rate)
    parent_snakes = []
    for _ in range(n_parents):
        idx = select(cum_scores)
        parent_snakes.append(pool[idx][1])

    return parent_snakes


def mate(parent_snakes):
    n_parents = len(parent_snakes)

    def select_random_parent():
        idx = np.random.randint(n_parents)
        return parent_snakes[idx]

    def create_child():
        p1 = select_random_parent()
        p2 = select_random_parent()

        game_board = board.Board(game_size)
        game_snake = snake.SnakeDNN(
            game_board, dnn_hidden_layers=snake_DNN_layers, verbose=0)

        l = len(p1.brain.W)
        for i in range(l):
            # crossover
            game_snake.brain.W[i] = (p1.brain.W[i] + p2.brain.W[i]) / 2
            game_snake.brain.b[i] = (p1.brain.b[i] + p2.brain.b[i]) / 2

            # matution
            game_snake.brain.W[i] += np.random.uniform(-mutation_rate,
                                                       mutation_rate)
            game_snake.brain.b[i] += np.random.uniform(-mutation_rate,
                                                       mutation_rate)
        return game_snake

    # create children with mating.
    children = []
    for _ in range(n_parents):
        c = create_child()
        children.append(c)

    return children


def update_population(old_snakes, child_snakes):
    size = len(old_snakes)
    all_snakes = [(fitness(s), s) for s in old_snakes] + \
        [(fitness(s), s) for s in child_snakes]
    all_snakes.sort(key=lambda x: x[0], reverse=True)
    _, new_snakes = zip(*all_snakes[:size])
    return new_snakes, all_snakes[0][1]


def genetic_algo():
    # create population
    current_snakes = create_population()
    run_snakes(current_snakes)
    for i in range(generation):
        # mating process to give child population
        parent_snakes = select_parents(current_snakes)
        child_snakes = mate(parent_snakes)
        run_snakes(child_snakes)

        # update population
        current_snakes, best_snake = update_population(
            parent_snakes, child_snakes)

        if i % 10 == 0:
            print(f'generation: {i} best_score: {fitness(best_snake)}')

    return best_snake


s = genetic_algo()
print('fitness:', fitness(s))
game.gui_param['time_lag'] = 0.5
game.replay_game(s)
