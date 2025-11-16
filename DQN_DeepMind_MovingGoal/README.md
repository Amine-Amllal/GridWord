# DQN Agent for FrozenLake with Moving Goal 🎯

An adaptation of the DeepMind DQN implementation to handle **dynamically changing goal positions** in the FrozenLake environment. This creates a more challenging and generalizable learning task where the agent must learn to navigate to *any* goal position, not just memorize a path to a fixed target.

## 🌟 Overview

This project extends the original DQN implementation from `DQN_DeepMind` to support environments where the goal position changes every episode. The agent learns a **general navigation policy** rather than memorizing a specific path.

### Key Difference from Original DQN

| Aspect | Fixed Goal DQN | Moving Goal DQN (This Project) |
|--------|----------------|-------------------------------|
| **State Representation** | Agent position only (one-hot) | Agent position + Goal position (both one-hot) |
| **Input Size** | `grid_size` (e.g., 25 for 5x5) | `2 × grid_size` (e.g., 50 for 5x5) |
| **Learning Task** | Learn path to specific goal | Learn general navigation to any goal |
| **Goal Behavior** | Fixed at (4,4) or specified position | Changes randomly each episode |
| **Generalization** | Only to same goal position | Can reach any valid goal position |
| **Policy Type** | Goal-specific policy | Goal-conditioned policy |

## 🏗️ Architecture

### State Encoding

The state is encoded as a concatenation of two one-hot vectors:

```
State = [Agent Position (one-hot) | Goal Position (one-hot)]
```

**Example for 5×5 grid:**
- Agent at (2, 3) → one-hot vector of length 25
- Goal at (4, 4) → one-hot vector of length 25
- **Final state:** concatenated vector of length **50**

This encoding explicitly tells the network:
1. Where the agent currently is
2. Where the goal currently is
3. Allows learning: "How do I get from position A to position B?"

### Network Architecture

The neural network architecture is inherited from the original DQN:

```
Input Layer (2 × grid_size)
    ↓
Hidden Layer 1 (256 or auto-adjusted)
    ↓ ReLU
Hidden Layer 2 (256 or auto-adjusted)
    ↓ ReLU
Output Layer (4 actions)
    ↓ Linear
Q-values for each action
```

**Auto-adjustment:** Smaller networks are used for smaller grids for faster convergence.

## 🚀 Quick Start

### Installation

1. Make sure you have the base `DQN_DeepMind` implementation in the parent directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from DQN_DeepMind_MovingGoal import DQNAgentMovingGoal

# Create agent with default 5×5 environment
agent = DQNAgentMovingGoal()

# Train the agent (goal changes each episode)
agent.train(num_episodes=1000, verbose=True)

# Evaluate performance
metrics = agent.evaluate(num_episodes=100)

# Generate comprehensive results
agent.generate_results_folder()
```

### Custom Environment

```python
# Define custom environment
env_params = {
    'nrow': 6,
    'ncol': 6,
    'holes': [(1, 1), (2, 3), (3, 2)],
    'start_state': (0, 0),
    'goal_positions': [(5, 5), (5, 0), (0, 5)],  # Goals will cycle through these
    'random_goal': True  # or False for cycling
}

agent = DQNAgentMovingGoal(env_params=env_params)
agent.train(num_episodes=2000)
```

### Run Demo Script

```bash
python dqn_agent_moving_goal.py
```

This will guide you through:
1. Selecting an environment (default, small, large, or custom)
2. Configuring training parameters
3. Training the agent
4. Evaluating performance
5. Generating results folder with visualizations

## 📊 What's Different in Training?

### Training Process

1. **Episode Reset:** Goal position changes (randomly or cyclically)
2. **State Encoding:** Both agent position AND current goal are encoded
3. **Action Selection:** Network receives goal-aware state
4. **Learning:** Agent learns to reach *any* goal, not just one
5. **Generalization:** Can navigate to unseen goal positions

### Reward Shaping

Same as original DQN with distance-based guidance:
- **+10.0** for reaching goal
- **-5.0** for falling in hole
- **-0.01** per step (encourages efficiency)
- **+0.1** for moving closer to goal (distance-based)

## 🎯 Results

### What You Get

After training, `generate_results_folder()` creates:

```
dqn_moving_goal_results_YYYYMMDD_HHMMSS/
├── training_progress.png          # 4-panel training visualization
│   ├── Episode rewards with moving average
│   ├── Training loss over time
│   ├── Success rate progression
│   └── Goal diversity (unique goals encountered)
│
├── learned_policy_goal_X_Y.png    # Policy for specific goals
│   (Multiple files for different goal positions)
│
├── training_summary.txt           # Detailed statistics
│   ├── Environment configuration
│   ├── Network architecture
│   ├── Hyperparameters
│   ├── Training statistics
│   ├── Per-goal performance
│   └── Goal coverage metrics
│
└── agent_navigation_moving_goal.gif  # Animated navigation demo
    (Shows agent reaching different goals)
```

### Example Training Results

**Environment:** 5×5 FrozenLake with 2 holes, random goal each episode

```
Training Statistics:
  Total Episodes: 1000
  Total Steps: 32,458
  Average Steps per Episode: 32.46
  Unique Goals Encountered: 21 (out of 23 possible)

Performance Metrics:
  Overall Success Rate: 89.3%
  Last 100 Episodes Success Rate: 94.0%
  Improvement: +24.5% (from first 100 to last 100)

Goal Coverage:
  Goals trained on: 21 unique positions
  Total possible goals: 23
  Coverage: 91.3%
```

## 🔬 Key Components

### 1. FrozenLakeMovingGoalEnv

Extended environment with moving goal capability:

```python
from DQN_DeepMind_MovingGoal import make_frozen_lake_moving_goal

# Random goal each episode
env = make_frozen_lake_moving_goal(
    nrow=5, ncol=5,
    holes=[(1, 1), (2, 3)],
    random_goal=True
)

# Cycling through specific goals
env = make_frozen_lake_moving_goal(
    nrow=4, ncol=4,
    goal_positions=[(3, 3), (3, 0), (0, 3)],
    random_goal=False  # Cycles: (3,3) → (3,0) → (0,3) → (3,3) ...
)
```

**New Methods:**
- `get_goal_distance()`: Manhattan distance to current goal
- `get_state_with_goal()`: Returns dict with agent position and goal
- Goal position included in `info` dict from `reset()`

### 2. DQNAgentMovingGoal

Adapted DQN agent with goal-conditioned learning:

```python
agent = DQNAgentMovingGoal(
    alpha=0.00025,           # Learning rate
    gamma=0.99,              # Discount factor
    epsilon=1.0,             # Initial exploration
    epsilon_decay=0.995,     # Decay rate
    epsilon_min=0.01,        # Min exploration
    hidden_layers=[256, 256], # Network size
    batch_size=32,           # Training batch size
    memory_size=10000,       # Replay buffer size
    target_update_freq=1000  # Target network update frequency
)
```

**Key Methods:**
- `encode_state(agent_pos, goal_pos)`: Encodes state with goal information
- `choose_action(agent_pos, goal_pos)`: Goal-conditioned action selection
- `train()`: Training with moving goals
- `evaluate()`: Includes per-goal performance metrics
- `visualize_policy(test_goal)`: Visualize policy for specific goal

## 🎓 Learning Insights

### Why Moving Goals are Harder

1. **Larger State Space:** Input is 2× larger (agent + goal positions)
2. **More Complex Policy:** Must learn general navigation, not specific path
3. **Diverse Training:** Each episode presents different challenge
4. **Generalization Required:** Must work for unseen goal positions

### Why This Approach Works

1. **Explicit Goal Information:** Network knows where it needs to go
2. **One-hot Encoding:** Clear position representation
3. **Experience Replay:** Mixes experiences with different goals
4. **Shared Weights:** Same network learns navigation for all goals

### Training Tips

**For Small Environments (3×3, 4×4):**
- Use 500-1000 episodes
- Smaller networks work well (auto-adjusted)
- Faster convergence due to limited state space

**For Medium Environments (5×5, 6×6):**
- Use 1000-2000 episodes
- Default network size is good
- Balance exploration and exploitation

**For Large Environments (7×7, 8×8):**
- Use 2000-3000 episodes
- Consider larger networks [256, 256]
- Larger replay memory (50,000+)
- More diverse goal positions for better coverage

## 🔄 Comparison with Fixed Goal

### Advantages of Moving Goal

✅ **Generalization:** Can reach any goal, not just trained position  
✅ **Robustness:** More diverse training, less overfitting  
✅ **Flexibility:** Same agent works for multiple objectives  
✅ **Real-world Relevance:** Many tasks have changing goals  

### Challenges

⚠️ **Complexity:** Larger input space, more parameters  
⚠️ **Training Time:** May need more episodes to converge  
⚠️ **Hyperparameters:** Requires careful tuning for stability  

### When to Use Each

**Use Fixed Goal DQN when:**
- Goal never changes
- Need fastest training
- Simple, static environment
- Learning specific path is enough

**Use Moving Goal DQN when:**
- Goals change between episodes
- Need generalization to new goals
- Training goal-conditioned policies
- Preparing for multi-task scenarios

## 🛠️ Advanced Usage

### Testing Specific Goals

```python
# Train agent
agent.train(num_episodes=1000)

# Test on specific goal
test_goal = (4, 3)
agent.visualize_policy(test_goal=test_goal, save_path='policy_4_3.png')
```

### Per-Goal Performance Analysis

```python
metrics = agent.evaluate(num_episodes=200)

# Access per-goal statistics
for goal, stats in metrics['goal_performance'].items():
    success_rate = (stats['successes'] / stats['attempts']) * 100
    avg_steps = stats['total_steps'] / stats['attempts']
    print(f"Goal {goal}: {success_rate:.1f}% success, avg {avg_steps:.1f} steps")
```

### Custom Goal Selection

```python
# Only use corner positions as goals
corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
env_params = {
    'nrow': 5,
    'ncol': 5,
    'goal_positions': corners,
    'random_goal': True
}
agent = DQNAgentMovingGoal(env_params=env_params)
```

## 📚 Code Structure

```
DQN_DeepMind_MovingGoal/
├── __init__.py                      # Module exports
├── frozenlake_moving_goal_env.py    # Moving goal environment
├── dqn_agent_moving_goal.py         # Adapted DQN agent
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── dqn_moving_goal_results_*/       # Generated results folders
```

### Dependencies

This implementation **reuses** from the original DQN:
- `DeepMindDQN` neural network class (from `DQN_DeepMind/dqn_agent.py`)
- Base `FrozenLakeEnv` (from `frozenlake_env.py`)

This ensures consistency and leverages the proven neural network architecture.

## 🧪 Experimentation Ideas

1. **Goal Curricula:** Start with nearby goals, gradually increase distance
2. **Sparse Goals:** Limit which positions can be goals
3. **Dynamic Holes:** Change holes along with goals
4. **Multi-Goal:** Reach multiple goals in single episode
5. **Continuous Goals:** Use goal coordinates directly (not one-hot)

## 📖 References

This implementation builds upon:
1. Original DQN: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. Goal-Conditioned RL: Schaul et al. (2015) - "Universal Value Function Approximators"
3. `DQN_DeepMind` implementation in parent directory

## 🤝 Contributing

Feel free to:
- Experiment with different network architectures
- Try different state encodings
- Test on larger environments
- Implement curriculum learning
- Add visualization improvements

## 📝 License

See LICENSE file in the repository root.

---

## 🎯 Quick Reference

### Minimal Example

```python
from DQN_DeepMind_MovingGoal import DQNAgentMovingGoal

# Create, train, evaluate
agent = DQNAgentMovingGoal()
agent.train(num_episodes=1000)
agent.evaluate(num_episodes=100)
agent.generate_results_folder()
```

### Key Classes

- `FrozenLakeMovingGoalEnv`: Environment with moving goals
- `DQNAgentMovingGoal`: Goal-conditioned DQN agent
- `make_frozen_lake_moving_goal()`: Factory function for environment

### Main Modifications from Original

1. **State encoding:** `encode_state(agent_pos, goal_pos)` - concatenates two one-hot vectors
2. **Input size:** Doubled to include goal position
3. **Action selection:** `choose_action(agent_pos, goal_pos)` - goal-aware
4. **Training loop:** Handles goal changes per episode
5. **Evaluation:** Tracks per-goal performance

---

**🎉 Happy Training! May your agent reach all goals successfully! 🎯**
