# FlappyBirdAI

This project serves to showcase the creation of a Flappy Bird clone with the ability to use AI. The majority of the game was made using Pygame and NEAT (NeuroEvolution of Augmenting Topologies).

## How it Works
- The config-feedforward.txt file will be used to provide changes to the population amount of birds on the screen at a time and how long the loop/test should run for
- When "Train the AI" is selected, the game will proceed to run on its own using the setting set in the txt file mentioned before
- This will continue to run until the fitness threshold is complete and the player chooses to exit
- The best AI model will be saved as a Pickle file after the bird reaches 50 points during the training

## Notes
- NEAT values at the top of the config-feedforward.txt file can be changed such as population count or threshold limit if needed
- The points necessary for a Pickle file to be created can also be changed in line 432 (if statement) as needed
- Custom bird images can be added aswell if coding it in (will try to make it easier in the future)

## Tools Used
- 	[Pygame](https://github.com/pygame/pygame) 🐍🎮
- 	[NEAT](https://github.com/CodeReclaimers/neat-python)
