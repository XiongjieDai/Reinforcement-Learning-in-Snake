# Reinforcement-Learning-in-Snake

Snake is a famous video game that originated in the 1976 arcade game Blockade. The player uses up, down, left and right to control the snake which grows in length (when it eats the food pellet), with the snake body and walls around the environment being the primary obstacle. In this project, we trained an AI agent using reinforcement learning to play a simple version of the game Snake. We implemented a TD version of the Q-learning algorithm.

The external libraries required for this project are `numpy` and `pygame`. To play the game yourself and get acquainted with it, you can run

`python mp6.py --human` 

### Trained Snake agent:
![alt text](https://github.com/XiongjieDai/Reinforcement-Learning-in-Snake/blob/main/snake.gif)

## Q-Learning Agent

We created a snake agent that learns how to get as many food pellets as possible without dying, which corresponds to maximizing the reward of the agent. In order to do this, we used the Q-learning algorithm. Our task was to implement the TD Q-learning algorithm and train it on the Markov Decision Process (MDP).

![alt text](https://github.com/XiongjieDai/Reinforcement-Learning-in-Snake/blob/main/RL_loop.png)

In Q-learning, instead of explicitly learning a representation for transition probabilities between states, we let the agent observe its environment, choose an action, and obtain some reward. In theory, after enough iterations, the agent will implicitly learn the value for being in a state and taking an action. We refer to this quantity as the **Q-value** for the state-action pair.

Explictly, our agent interacts with it’s environment in the following feedback loop:

1. At step $t$, the agent is in current state $s_t$ and chooses an “optimal” action $a_t$ using the learned values of $Q(s_t,a)$. This acton is then executed in the environment.
2. From the result of the action on the environment, the agent obtains a reward $r_t$.
3. The agent then “discretizes” this new environment by generating a state $s_{t+1}$ based off of the new, post-action environment.
4. With $s_t$, $a_t$, $r_t$, and $s_{t+1}$, the agent can update its Q-value estimate for the state-action pair: $Q(s_t,a_t)$.
5. The agent is now in state $s_{t+1}$, and the process repeats.
   
Often, the notations for the current state $s_t$ and next state $s_{t+1}$ are written as $s$ and $s'$, respectively. Same for the current action $a$ and next action $a'$.

### The Q-Update

The Q update formula is:

$$Q^{\text{new}}(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \cdot \left( r_t + \gamma \cdot \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right)$$

where $\gamma$ is the Temporal-Difference (TD) hyperparameter discounting future rewards, and

$$\alpha = \frac{C}{C + N(s, a)}$$


is the learning rate controlling how much our Q estimate should change with each update. Unpacking this equation: $C$ is a hyperparameter, and $N(s,a)$ is the number of times the agent has been in state and taken action $a$. As you can see, the learning rate decays as we visit a state-action pair more often.

### Choosing the Optimal Action

With its current estimate of the Q-states, the agent must choose an “optimal” action to take. However, reinforcement learning is a balancing act between exploration (visiting new states to learn their Q-values) and greed (choosing the action with the highest Q-value). Thus, during training, we use an exploration policy defined below:

$$a^* = {\mathrm{argmax}_a}\ f(Q(s, a), N(s, a))$$

$$f(Q(s, a), N(s, a)) = 
\begin{cases} 
1 & \text{if } N(s, a) < Ne \\
Q(s, a) & \text{else}
\end{cases}$$



where $Ne$ is a hyperparameter. Intuitively, if an action hasn’t been explored enough times (when $N(s,a) < Ne$), the exploration policy chooses that action regardless of its Q-value. If there are no such actions, the policy chooses the action with the highest Q value. This policy forces the agent to visit each state and action at least $Ne$ times.

### Usage

mp6.py [-h] [--human] [--model_name MODEL_NAME] [--train_episodes TRAIN_EPS] [--test_episodes TEST_EPS] [--show_episodes SHOW_EPS] [--window WINDOW] [--Ne NE] [--C C]
              [--gamma GAMMA] [--snake_head_x SNAKE_HEAD_X] [--snake_head_y SNAKE_HEAD_Y] [--food_x FOOD_X] [--food_y FOOD_Y]
              
Playing around with the hyperparameters to see how well you can train your agent!
