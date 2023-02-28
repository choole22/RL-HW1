# Goal vs Hole-v1, Chris Hoole
### Note: a significant amount of the program was copied from Dr. B and the resources provided.

The program first trains the RL model by randomly traversing the grid for 1000 steps, recording the reward collected
at each given step with the addition of a discounted max-reward at the next step, all entered into a 'state' x 'action'
q-table. I had the best luck with a pure exploration method compared to having a decaying epsilon for some reason. 
After the q-table was filled, it still didn't converge within the amount of steps as it was slightly too sparse. This 
caused issues for testing the model as it tended to diverge, bouncing between the same several states not reaching a
terminal state of any kind. It should also be noted that I attempted to give a minor reward for the non-terminal states
to try and get the table to converge in the value of +1.

Running the program is simple, I just ran it in the PyCharm IDE though any python IDE will work. The program 
automatically runs the train() function, which trains the model, prints the q-table before and after the training, and 
creates a .gif to better visualize the training as it is happening. The test() then runs, using the same q-table with a
0.1 exploration rate to see how well the model was trained and similarly creates a .gif of the process.