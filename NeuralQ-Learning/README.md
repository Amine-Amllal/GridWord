# NeuralQ-Learning - Deep Q-Learning Implementation

This folder contains a deep Q-learning implementation that uses a neural network to approximate Q-values, enabling better generalization and scalability compared to tabular Q-learning methods.

## 📁 Contents

### Main Files

- **`neural_network_q_value_agent.py`** - Core implementation of the Neural Network Q-Learning agent
  - Custom neural network implementation using NumPy
  - Experience replay for stable learning
  - Epsilon-greedy exploration strategy
  - Policy visualization and training metrics

### Generated Files

- **`neural_q_learning_policy.png`** - Heatmap visualization showing:
  - Learned policy with arrows indicating optimal actions
  - Q-value intensity (color-coded)
  - Grid layout with start, goal, and obstacles

- **`neural_q_learning_training_progress.png`** - Training metrics including:
  - Episode rewards over time
  - Success rate progression
  - Loss convergence
  - Average Q-values evolution

## 🧠 Neural Network Architecture

The agent uses a feedforward neural network with:

- **Input Layer**: State encoding (position + grid features)
- **Hidden Layers**: Configurable depth and width (default: [64, 64])
- **Output Layer**: Q-values for each action (LEFT, DOWN, RIGHT, UP)
- **Activation**: ReLU for hidden layers, linear for output
- **Initialization**: He initialization for better gradient flow

## 🚀 Key Features

### 1. Experience Replay
- Memory buffer storing past experiences (state, action, reward, next_state, done)
- Random sampling of mini-batches for training
- Breaks correlation between consecutive samples
- Improves learning stability and efficiency

### 2. Neural Network Q-Value Approximation
- Generalizes across similar states
- Handles larger state spaces better than tabular methods
- Smooth Q-value function approximation
- Custom backpropagation implementation

### 3. Adaptive Learning
- Epsilon-greedy exploration with decay
- Configurable learning rate
- Batch gradient descent
- Loss tracking for monitoring convergence

### 4. Comprehensive Visualization
- Policy heatmaps with action arrows
- Training progress charts (rewards, success rate, loss)
- Q-value distribution analysis
- Episode-by-episode performance tracking

## 🎮 Usage

### Basic Usage

```bash
# Run from the project root directory
python NeuralQ-Learning/neural_network_q_value_agent.py
```

### Customization

Edit the `main()` function in `neural_network_q_value_agent.py`:

```python
# Environment configuration
env_params = {
    'nrow': 5,              # Grid height
    'ncol': 5,              # Grid width
    'holes': [(1, 1), (2, 3), (3, 2)],  # Obstacle positions
    'goal': (4, 4),         # Goal position
    'start_state': (0, 0)   # Starting position
}

# Agent hyperparameters
agent = NeuralNetworkQValueAgent(
    alpha=0.001,            # Learning rate
    gamma=0.95,             # Discount factor
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995,    # Exploration decay rate
    epsilon_min=0.01,       # Minimum exploration rate
    hidden_sizes=[64, 64],  # Neural network architecture
    batch_size=32,          # Training batch size
    memory_size=10000       # Experience replay buffer size
)

# Training configuration
agent.train(
    num_episodes=2000,      # Number of training episodes
    max_steps_per_episode=100,  # Max steps per episode
    train_frequency=1,      # How often to train (every N steps)
    verbose=True,           # Print progress
    save_results=True       # Save visualizations
)
```

## 📊 Expected Output

After running the agent, you'll see:

1. **Training Progress** - Real-time updates showing:
   - Current episode number
   - Average reward (last 100 episodes)
   - Success rate
   - Current epsilon value
   - Neural network loss

2. **Test Results** - Evaluation metrics:
   - Success rate over test episodes
   - Average steps to goal
   - Performance comparison with training

3. **Visualizations**:
   - Policy heatmap (`neural_q_learning_policy.png`)
   - Training curves (`neural_q_learning_training_progress.png`)

## 🔬 How It Works

### State Encoding
The agent encodes the grid state into a feature vector:
- Current position (row, col)
- Goal position
- Obstacle locations
- Grid dimensions

### Q-Value Approximation
Instead of maintaining a Q-table, the neural network learns to approximate:
```
Q(s, a) ≈ NN(s)[a]
```
where NN(s) outputs Q-values for all actions given state s.

### Training Loop
1. **Observe** current state
2. **Select** action using epsilon-greedy policy
3. **Execute** action and observe reward and next state
4. **Store** experience in replay memory
5. **Sample** random mini-batch from memory
6. **Update** network weights using gradient descent
7. **Repeat** until convergence

### Bellman Update
The network is trained to minimize the loss:
```
Loss = (Q(s,a) - (r + γ * max_a' Q(s',a')))²
```

## ⚡ Performance

### Advantages over Tabular Q-Learning:
- ✅ Better generalization to unseen states
- ✅ Handles larger state spaces efficiently
- ✅ Smoother learning curves with experience replay
- ✅ Can learn complex patterns and relationships

### Considerations:
- ⚠️ Requires more computational resources
- ⚠️ Hyperparameter tuning is crucial
- ⚠️ May require more training episodes for small grids

## 🛠️ Dependencies

Required packages (install via root `requirements.txt`):
- numpy
- matplotlib

## 📈 Monitoring Training

Watch for these indicators of successful training:

1. **Increasing average rewards** - Should trend upward
2. **Increasing success rate** - Should approach 100% for solvable grids
3. **Decreasing loss** - Network prediction error should decrease
4. **Stable Q-values** - Should converge to consistent values

## 💡 Tips for Best Results

1. **Start with higher epsilon** - Ensures thorough exploration early
2. **Use appropriate decay** - Balance exploration and exploitation
3. **Tune learning rate** - Too high = unstable, too low = slow
4. **Monitor loss** - Should decrease steadily; if not, adjust hyperparameters
5. **Increase memory size** - For complex environments, larger replay buffers help
6. **Adjust network size** - Larger grids may benefit from deeper networks

## 🔄 Comparison with Other Agents

| Feature | Neural Q-Learning | Tabular Q-Learning | DQN DeepMind |
|---------|------------------|-------------------|--------------|
| State Space | Medium to Large | Small to Medium | Large |
| Generalization | ✅ Excellent | ❌ None | ✅ Excellent |
| Sample Efficiency | 🟡 Moderate | ✅ High | 🟡 Moderate |
| Complexity | 🟡 Moderate | ✅ Simple | 🔴 High |
| Memory Required | 🟡 Moderate | ✅ Low | 🔴 High |

## 📚 Further Reading

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Deep Reinforcement Learning](http://www.deeplearningbook.org/) - Comprehensive textbook
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)

---

**Last Updated**: November 16, 2025
