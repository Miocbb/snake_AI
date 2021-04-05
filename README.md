# AI for the Classic Snake Game

Use the deep neural network and genetic algorithm to build the artificial
intelligence (AI) to play the classic snake game.

## The Brain of Snake from Deep Neural Network

The snake looks at 8 directions at the position of its head. These directions
are up, down, left, right, up-right, up-left, down-left and down-right.
In each direction, it measures the distance between its head and the wall,
the food and the body. So there ``8 * 3 = 24`` pieces of information
in total. These 24 pieces of information are the input features used
in the deep neural network to train the AI.


## The Evolution of Snakes by Genetic Algorithm

To use genetic algorithm to train the snakes, or optimize the "brain" of the
snakes, the way of evaluating the performance of snakes are important.

The evaluating of performance for snakes mainly involves the following aspects:
the final length of the snake, the number of steps it moves and the number of
turns it takes.

To make the comparison fair for different snakes (different in brains),
the snakes should be ran under the same environments. In another word,
the generation of food for all the snakes should be the same. Since the food
generation is random, setting the same environment can be easily achieved
by seeding the random number generator with fixed seed.

To use genetic algorithm to train the snakes, there are two layers of loops,
one is the loop over generations of snakes, and the other is the loop over
the set of environments. To evolve the group of snakes, there are two
strategies. One is setting the outer loop for generations and the inner loop
for environments. This lets each generation of snakes explore multiple
environments to learn how to survive. The other is setting the outer loop
for the environments and the inner loop for generations. This lets snakes
learn the certain environment generation by generation.

## Demo
Following are two trained snakes searching for food in a 10x10 game board.

<img src= "./image/best2.gif" width="300"/> <img src= "./image/best3.gif" width="300"/>


## Acknowledgements

Inspired by the following repositories:

- https://github.com/greerviau/SnakeAI

- https://github.com/aliakbar09a/AI_plays_snake
