# Q-Learning Implementations

This folder contains various implementations of the Q-Learning algorithm applied to the FrozenLake environment. Q-Learning is a model-free reinforcement learning algorithm that learns the optimal action-value function through temporal difference learning.

## 📁 Contents

### Core Implementations

1. **`q_learning_agent.py`** - Standard Q-Learning Agent
   - Full-featured Q-Learning implementation with comprehensive visualization
   - Animated training progress showing Q-value evolution
   - Interactive episode execution with learned policy
   - Automatic hyperparameter adaptation based on environment size
   - Results generation (training plots, GIFs, summary reports)

2. **`q_learning_agent_dynamic.py`** - Dynamic Environment Q-Learning
   - Adapts to constantly changing environments
   - Goals and holes move between episodes
   - Tests agent's ability to handle non-stationary environments
   - Enhanced exploration parameters for dynamic scenarios
   - Adaptation performance metrics

3. **`q_learning_avec_shaping.py`** - Q-Learning with Reward Shaping
   - Implements potential-based reward shaping
   - Uses Manhattan distance to goal as shaping function
   - Accelerates learning by providing intermediate rewards
   - Maintains optimal policy guarantees

### Results

- **`q_learning_results/`** - Training outputs from standard Q-Learning
  - `training_summary.txt` - Detailed training statistics and learned policy
  - `cumulative_rewards.png` - Training performance visualization
  - Animation GIFs of agent behavior

## 🎯 What is Q-Learning?

Q-Learning is a **model-free** reinforcement learning algorithm that learns an optimal action-value function Q(s,a) representing the expected cumulative reward for taking action `a` in state `s`.

### Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
```

Where:
- `α` (alpha): Learning rate (0-1)
- `γ` (gamma): Discount factor (0-1)
- `r`: Immediate reward
- `s'`: Next state
- `a'`: Next action

### Key Features

- **Off-policy learning**: Learns optimal policy while following ε-greedy exploration
- **Temporal difference**: Updates estimates based on other estimates
- **Convergence guarantee**: Provably converges to optimal Q* under certain conditions

## 🚀 Usage

### Basic Training

```python
from q_learning_agent import QLearningAgent

# Create agent
agent = QLearningAgent(
    alpha=0.5,           # Learning rate
    gamma=0.9,           # Discount factor
    epsilon=0.9,         # Initial exploration rate
    epsilon_decay=0.995, # Exploration decay
    epsilon_min=0.01     # Minimum exploration
)

# Train the agent
Q = agent.train(num_episodes=1000, verbose=True)

# Test learned policy
agent.run_learned_episode()

# Generate comprehensive results
agent.generate_results_folder()
```

### Custom Environment

```python
# Create agent with custom environment parameters
env_params = {
    'nrow': 6,
    'ncol': 6,
    'holes': [(2, 2), (3, 4)],
    'goal': (5, 5),
    'start_state': (0, 0)
}

agent = QLearningAgent(env_params=env_params)
```

### Interactive Environment Creation

```python
# Let the user design the environment interactively
agent = QLearningAgent(interactive_env=True)
```

### Dynamic Environment Training

```python
from q_learning_agent_dynamic import DynamicQLearningAgent

# Create dynamic agent where environment changes between episodes
agent = DynamicQLearningAgent(
    move_probability=0.8,      # 80% chance environment changes
    move_frequency='episode'   # Change every episode
)

agent.train(num_episodes=5000)
agent.test_adaptation(test_episodes=50)
```

## 🎨 Visualizations

All implementations provide rich visualizations:

1. **Training Progress Animation**
   - Value function evolution (max Q-values)
   - Policy evolution over time
   - Q-values for best actions
   - Learning curves with success rates

2. **Episode Execution**
   - Step-by-step agent movement
   - Q-value bar charts for action selection
   - Path visualization
   - Success/failure indicators

3. **Result GIFs**
   - Training process animation
   - Final learned policy demonstration
   - Multiple episode comparisons

## 📊 Performance Metrics

The implementations track and report:

- **Success Rate**: Percentage of episodes reaching the goal
- **Average Reward**: Mean cumulative reward per episode
- **Average Steps**: Mean steps to goal/termination
- **Epsilon Decay**: Exploration rate over time
- **Q-value Convergence**: Stability of learned values

### Example Results (5x5 FrozenLake, 400 episodes)

- Success Rate: 100% (last 100 episodes)
- Average Reward: 1.0
- Average Steps: 23.9
- Learning Rate: 0.5
- Discount Factor: 0.9

## 🔧 Hyperparameter Tuning

### Adaptive Parameters

The standard implementation automatically adjusts hyperparameters based on:
- Environment size (grid dimensions)
- Number of holes (complexity)
- Grid shape (square vs rectangular)

### Recommendations by Environment Size

| Environment | Episodes | Learning Rate (α) | Epsilon Decay | Min Epsilon |
|-------------|----------|-------------------|---------------|-------------|
| Small (≤3×3) | 300-800 | 0.3-0.5 | 0.995 | 0.01 |
| Medium (4×4-5×5) | 500-2000 | 0.3-0.5 | 0.998 | 0.02 |
| Large (6×6-7×7) | 1500-4000 | 0.3 | 0.998 | 0.02 |
| Very Large (≥8×8) | 3000-8000 | 0.3 | 0.9995 | 0.05 |

### Training Intensity Options

Each implementation provides flexible training options:
- **Quick**: ~1/3 recommended episodes (fastest results)
- **Standard**: ~60% recommended episodes (balanced)
- **Recommended**: Full recommended episodes (best results) ⭐
- **Thorough**: 1.5× recommended episodes (maximum quality)
- **Extensive**: 2× recommended episodes (research-grade)

## 🆚 Comparison: Standard vs Dynamic vs Shaped

| Feature | Standard | Dynamic | Reward Shaping |
|---------|----------|---------|----------------|
| Environment | Static | Changes each episode | Static |
| Learning Speed | Moderate | Slower | Faster |
| Exploration | ε-greedy | Enhanced ε-greedy | ε-greedy |
| Use Case | Standard RL | Non-stationary problems | Sparse rewards |
| Convergence | Guaranteed | Approximate | Guaranteed |
| Complexity | Low | High | Moderate |

## 📝 Output Files

### Generated by `generate_results_folder()`

- `cumulative_rewards.png` - Training performance chart
- `training_animation.gif` - Q-value and policy evolution
- `final_path.gif` - Agent executing learned policy
- `training_summary.txt` - Complete statistics and hyperparameters

### Dynamic Agent Additional Outputs

- `q_values_table.csv` - Full Q-table with best actions
- `agent_pathfinding_animation.gif` - Multi-episode behavior
- Adaptation metrics and environment change logs

## 🧪 Experimental Features

### Reward Shaping (q_learning_avec_shaping.py)

Uses Manhattan distance to goal as potential function:
```python
φ(s) = |row - goal_row| + |col - goal_col|
shaped_reward = r + γφ(s') - φ(s)
```

Benefits:
- Faster convergence in sparse reward environments
- Maintains optimal policy (potential-based shaping)
- Provides learning signal even when goal is far

## 🔍 Key Differences from Other Algorithms

### vs Value Iteration
- **Q-Learning**: Model-free, learns from experience
- **Value Iteration**: Model-based, requires transition dynamics

### vs Policy Iteration  
- **Q-Learning**: Learns Q-values, derives policy
- **Policy Iteration**: Directly improves policy

### vs DQN (Deep Q-Network)
- **Q-Learning**: Tabular, discrete states
- **DQN**: Neural network, handles large/continuous spaces

## 🎓 Learning Concepts Demonstrated

1. **Temporal Difference Learning**: Updates based on bootstrapping
2. **Off-Policy Learning**: Learning optimal policy while exploring
3. **ε-greedy Exploration**: Balance exploration vs exploitation
4. **Convergence**: Q-values converge to optimal Q* 
5. **Generalization**: Handles various environment configurations

## 🐛 Common Issues & Solutions

### Slow Convergence
- Increase learning rate (α)
- Increase number of episodes
- Adjust epsilon decay for more exploration

### Poor Performance
- Check environment complexity vs training episodes
- Verify reward structure
- Ensure sufficient exploration (epsilon not too low)

### Stuck in Local Optimum
- Increase minimum epsilon
- Use reward shaping
- Try dynamic environment training for robustness

## 📚 References

- Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards" (PhD thesis)
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction" 
- Ng et al. (1999). "Policy Invariance Under Reward Transformations"

## 🤝 Integration with Other Agents

This Q-Learning implementation is part of a larger collection of RL algorithms. See also:
- `../DQN_DeepMind/` - Deep Q-Network with neural networks
- `../PolicyIteration/` - Model-based policy iteration
- `../ValueIteration/` - Model-based value iteration
- `../NeuralQ-Learning/` - Neural network Q-function approximation

---

**Note**: All implementations use the shared `frozenlake_env.py` environment wrapper for consistency across the project.
