# Taxi Q-Learning: Reinforcement Learning with OpenAI Gym

This project implements a Q-learning algorithm to solve the classic **Taxi Problem** from OpenAI Gym (now Gymnasium). The agent learns to navigate a grid world, pick up passengers, and drop them off at their destination efficiently.

## 📋 Problem Description

The Taxi Problem is a reinforcement learning environment where:

- A taxi must navigate a 5x5 grid world
- There are 4 designated pickup/drop-off locations: R(ed), G(reen), Y(ellow), and B(lue)
- The taxi must pick up a passenger from one location and deliver them to another
- The episode ends when the passenger is successfully dropped off

### Map Layout

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

### State Space

- **500 discrete states** (25 taxi positions × 5 passenger locations × 4 destinations)
- **404 reachable states** during normal episodes
- State representation: `(taxi_row, taxi_col, passenger_location, destination)`

### Action Space

6 discrete deterministic actions:

- 0: Move south
- 1: Move north
- 2: Move east
- 3: Move west
- 4: Pickup passenger
- 5: Drop off passenger

### Reward Structure

- **-1** per step (encourages efficiency)
- **+20** for successful passenger delivery
- **-10** for illegal pickup/drop-off actions

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd taxi-marl
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

Or install the specific packages needed:

```bash
pip install gymnasium==1.2.2 numpy==2.3.4 matplotlib==3.10.7
pip install "gymnasium[toy-text]"
```

### Running the Notebook

1. Launch Jupyter Notebook:

```bash
jupyter notebook taxi_q_learning.ipynb
```

2. Run the cells sequentially to:
   - Initialize the environment
   - Observe a random agent (baseline)
   - Train the Q-learning agent
   - Test the trained agent's performance
   - Generate animated GIFs of episodes

## 📊 Features

### 1. **Environment Visualization**

- Real-time rendering of the taxi environment
- Animated GIF generation for episode playback
- Frame-by-frame visualization with state/action/reward information

### 2. **Random Agent Baseline**

- Demonstrates untrained agent behavior
- Shows the difficulty of the problem with random actions
- Provides performance baseline for comparison

### 3. **Q-Learning Implementation**

- Tabular Q-learning with configurable hyperparameters
- Training over 10,000 episodes
- Convergence visualization with reward and epoch plots

### 4. **Performance Testing**

- Trained policy evaluation
- Success rate and efficiency metrics
- Visual comparison between trained and untrained agents

## 🔧 Hyperparameters

The Q-learning algorithm uses the following default hyperparameters:

```python
alpha = 0.1      # Learning rate
gamma = 1.0      # Discount factor
epsilon = 0.1    # Exploration rate
num_episodes = 10000  # Training episodes
```

## 📈 Results

After training:

- The agent learns an optimal policy for passenger pickup and delivery
- Cumulative rewards converge to positive values
- Number of epochs per episode decreases significantly
- Illegal actions (failed pickups/drop-offs) are minimized

Training visualizations show:

1. **Cumulative reward per episode** - demonstrates learning progress
2. **Epochs per episode** - shows increasing efficiency over time

## 🎯 Usage Examples

### Train a New Agent

```python
# Initialize environment
env = gym.make("Taxi-v3", render_mode="rgb_array").env

# Train Q-learning agent
q_table = train_agent(env, num_episodes=10000)
```

### Test Trained Agent

```python
# Evaluate trained policy
test_policy(env, q_table, num_episodes=100)
```

### Generate Episode Animation

```python
# Store episode as GIF
store_episode_as_gif(experience_buffer, filename='taxi_episode.gif')
```

## 📚 Implementation Details

### Q-Learning Algorithm

The implementation follows the classic Q-learning update rule:

Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Where:

- s: current state
- a: action taken
- r: reward received
- s': next state
- α: learning rate
- γ: discount factor

### Exploration Strategies

The notebook includes multiple exploration/exploitation strategies:

- **Basic ε-greedy**: Random action with probability ε
- **Action mask exploration**: Only valid actions during exploration
- **Tie-breaking**: Random selection among equally good actions

## 📝 References

- OpenAI Gym Taxi Environment: [GitHub Source](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)
- Q-Learning Tutorial: [LearningDataSci](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
- Original Paper: "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition" by Tom Dietterich

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Improve documentation
- Add new RL algorithms

## 📄 License

This project is open source and available for educational purposes.

## 👥 Authors

- Based on OpenAI Gym's Taxi environment
- Q-Learning implementation adapted from various RL tutorials

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium team for the environment
- Reinforcement Learning community for educational resources
