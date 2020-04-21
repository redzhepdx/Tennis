[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition
Udacity Deep Reinforcement Learning Project 3 - Multi Agent Self Learning

### Details

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Setup
The project uses Jupyter Notebook. This command needs to be run to install the needed packages:
```
!pip -q install ./python
```

Running all the cells in the notebook will install it automatically.


### Project Structure and Instructions
- `agents/` -> Contains the implementations of DDPG and MultiDDPG Agents
- `models/` -> Contains the implementations of Critic and Actor Neural Networks. [Pytorch]
- `utils/` -> Contains memory modules and noise generator class implementations.
- `tennis.ipynb` -> Execution of the algorithm. Training Agents and Unity Visualizations
- `report.pdf` -> description of the methods and application
- `*.pth files` -> pre-trained models of each agent [agent_1_actor_checkpoint.pth, agent_2_critic_checkpoint.pth etc.]