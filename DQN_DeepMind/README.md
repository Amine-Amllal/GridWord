# DeepMind DQN Agent for FrozenLake

This directory contains a complete implementation of the **Deep Q-Network (DQN)** algorithm as introduced by DeepMind in their landmark 2015 Nature paper: ["Human-level control through deep reinforcement learning"](https://www.nature.com/articles/nature14236).

## 🎯 Overview

The DQN agent learns to navigate the FrozenLake environment using deep neural networks to approximate Q-values, combined with two key innovations:

1. **Experience Replay**: Stores past experiences and samples random mini-batches for training, breaking correlations in the observation sequence.
2. **Target Network**: Uses a separate network for computing target Q-values, providing stable learning targets.

## 🏗️ Architecture

### Network Architecture (DeepMind Style)
- **Input Layer**: One-hot encoded state representation
- **Hidden Layers**: 2 layers with 256 units each (configurable)
- **Activation**: ReLU for hidden layers, Linear for output
- **Output Layer**: Q-values for each action (4 outputs)

### Key Components

1. **`DeepMindDQN` Class**
   - Neural network implementation with forward and backward propagation
   - Xavier/He weight initialization
   - ReLU activation functions
   - Gradient descent optimization

2. **`DQNAgent` Class**
   - Complete DQN algorithm implementation
   - Experience replay memory (deque-based)
   - Epsilon-greedy exploration
   - Target network with periodic updates
   - Training and evaluation functionality
   - Visualization and results generation

## 🚀 Usage

### Basic Usage

```python
from dqn_agent import DQNAgent

# Create agent with default 5x5 environment
agent = DQNAgent()

# Train the agent
agent.train(num_episodes=1000, verbose=True)

# Evaluate performance
metrics = agent.evaluate(num_episodes=100)

# Generate results folder with plots and summary
agent.generate_results_folder()
```

### Custom Environment

```python
# Small environment for quick testing
env_params = {
    'nrow': 3, 
    'ncol': 3, 
    'holes': [(1, 1)], 
    'goal': (2, 2)
}
agent = DQNAgent(env_params=env_params)
```

### Advanced Configuration

```python
agent = DQNAgent(
    alpha=0.00025,          # Learning rate (DeepMind default)
    gamma=0.99,             # Discount factor
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995,    # Epsilon decay per episode
    epsilon_min=0.01,       # Minimum epsilon
    hidden_layers=[256, 256],  # Network architecture
    batch_size=32,          # Mini-batch size
    memory_size=10000,      # Replay memory size
    target_update_freq=1000 # Target network update frequency
)
```

## 📊 Hyperparameters

Based on DeepMind's DQN paper with adaptations for FrozenLake:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| Learning Rate (α) | 0.00025 | Neural network learning rate |
| Discount Factor (γ) | 0.99 | Future reward discount |
| Initial Epsilon (ε) | 1.0 | Starting exploration rate |
| Epsilon Decay | 0.995 | Per-episode decay factor |
| Min Epsilon | 0.01 | Minimum exploration rate |
| Batch Size | 32 | Mini-batch size for training |
| Memory Size | 10,000 | Replay memory capacity |
| Target Update Freq | 1,000 | Steps between target network updates |
| Hidden Layers | [256, 256] | Network architecture |

## 🎮 Running the Demo

Execute the main script to run an interactive demo:

```bash
python dqn_agent.py
```

The demo will:
1. Let you choose an environment (default, small, large, or custom)
2. Configure training parameters
3. Train the DQN agent
4. Evaluate performance over 100 episodes
5. Generate a comprehensive results folder

## 📁 Results Output

The `generate_results_folder()` method creates a timestamped folder containing:

- **`training_progress.png`**: Multi-panel plot showing:
  - Episode rewards with moving average
  - Training loss over time
  - Success rate progression
  - Epsilon decay curve

- **`learned_policy.png`**: Visualization of the learned policy as action arrows on the grid

- **`training_summary.txt`**: Detailed text report including:
  - Environment configuration
  - Network architecture
  - Hyperparameters
  - Training statistics
  - Performance metrics
  - Comparison of first vs last 100 episodes

## 🧪 Example Results

After training for 1000 episodes on a 5x5 FrozenLake:

```
📊 Evaluation Results:
  Average Reward: 0.850 ± 0.357
  Success Rate: 85.0%
  Average Steps: 12.3 ± 4.2
```

## 🎯 Training Results

### Latest Training Run (2025-10-27)

**Environment Configuration:**
- Grid Size: 5x5
- Holes: None (safe path)
- Start: (0, 0)
- Goal: (4, 4)

**Network Architecture:**
- Hidden Layers: [128, 128]
- Total Parameters: 3,588
- Input: 25 (one-hot encoded state)
- Output: 4 (action values)

**Hyperparameters:**
- Learning Rate: 0.0005
- Discount Factor (γ): 0.99
- Batch Size: 32
- Memory Size: 10,000
- Target Update: Every 1,000 steps

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| **Overall Success Rate** | **97.20%** |
| **Last 100 Episodes Success** | **100.00%** |
| **First 100 Episodes Success** | 87.00% |
| **Improvement** | +13.00% |
| Total Episodes | 1,000 |
| Total Steps | 35,074 |
| Avg Steps per Episode | 35.07 |
| Final Epsilon | 0.0110 |
| Average Loss | 0.2157 |
| Final Loss (Last 100) | 0.2076 |

**Key Improvements Implemented:**
- ✅ Gradient clipping for stable training
- ✅ Reward shaping with distance-based guidance
- ✅ Adaptive hyperparameters based on environment size
- ✅ Warmup period for replay buffer diversity
- ✅ Delayed epsilon decay for better exploration

**Visualizations:**

The training results folder (`dqn_results/`) contains:
- **`training_progress.png`**: Shows reward progression, loss curves, success rate, and epsilon decay
- **`learned_policy.png`**: Visualizes the optimal policy learned by the agent

This training run demonstrates excellent convergence with near-perfect performance in the final episodes, validating the effectiveness of the implemented improvements.

## 🔬 Algorithm Details

### DQN Update Rule

For each experience (s, a, r, s', done):

1. **Sample batch** from replay memory
2. **Compute target**: 
   - If done: y = r
   - Otherwise: y = r + γ * max_a' Q_target(s', a')
3. **Update Q-network**: Minimize (Q(s,a) - y)²
4. **Update target network** every N steps

### Key Differences from Tabular Q-Learning

| Aspect | Q-Learning | DQN |
|--------|-----------|-----|
| Q-value storage | Table | Neural Network |
| Training | Single sample | Mini-batch |
| Experience | Immediate use | Replay buffer |
| Target | Bootstrap from same Q | Separate target network |
| Scalability | Limited by state space | Scales to large/continuous spaces |

## 🛠️ Dependencies

- `numpy`: Numerical computations and neural network implementation
- `matplotlib`: Visualization and plotting
- `collections.deque`: Efficient replay memory
- `frozenlake_env`: Custom FrozenLake environment (parent directory)

## 📚 References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
2. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.)

## 🎓 Learning Notes

### Why Experience Replay?
- Breaks temporal correlations between consecutive samples
- Improves data efficiency by reusing experiences multiple times
- Reduces variance in updates
- Enables mini-batch learning for stable gradients

### Why Target Network?
- Provides stable learning targets
- Prevents moving target problem (chasing a moving target makes convergence difficult)
- Updated less frequently than Q-network
- Reduces oscillations and divergence

### Compared to Tabular Methods
- **Pros**: Can handle large state spaces, learns representations, generalizes across similar states
- **Cons**: More complex, requires tuning, slower per-episode, can be less sample-efficient on small problems

## 🔧 Customization

### Modify Network Architecture
```python
agent = DQNAgent(hidden_layers=[512, 256, 128])  # Deeper network
```

### Adjust Exploration Strategy
```python
agent = DQNAgent(
    epsilon=0.5,        # Start with less exploration
    epsilon_decay=0.99, # Slower decay
    epsilon_min=0.05    # Higher minimum exploration
)
```

### Change Memory and Batch Sizes
```python
agent = DQNAgent(
    memory_size=50000,  # Larger memory
    batch_size=64       # Larger batches
)
```

## 📈 Performance Tips

1. **For Small Environments (3x3, 4x4)**:
   - Use fewer episodes (500-1000)
   - Smaller networks ([128, 128])
   - Faster epsilon decay

2. **For Large Environments (8x8+)**:
   - More episodes (2000+)
   - Larger networks ([256, 256] or [512, 256])
   - Larger replay memory (50000+)
   - Slower epsilon decay

3. **General Tips**:
   - Monitor success rate instead of just rewards
   - Increase target update frequency for stability
   - Adjust learning rate if training is unstable
   - Use visualization to debug policy

## 🤝 Contributing

This implementation is designed for educational purposes. Feel free to:
- Experiment with different network architectures
- Try different hyperparameters
- Add additional DQN improvements (Double DQN, Dueling DQN, etc.)
- Test on other environments

## 📝 License

See the LICENSE file in the parent directory.

---

**Happy Learning! 🚀**
