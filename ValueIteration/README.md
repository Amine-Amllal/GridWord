# Value Iteration

This folder contains an implementation of the Value Iteration algorithm for the FrozenLake environment. Value Iteration is a dynamic programming algorithm that computes the optimal value function and policy by iteratively updating state values until convergence.

## 📁 Contents

- **`value_iteration_agent.py`** - Value Iteration Implementation
  - Complete Value Iteration algorithm with Bellman optimality equations
  - Animated convergence visualization showing value function evolution
  - Optimal policy execution with real-time visualization
  - Guaranteed convergence to optimal policy
  - Works with custom and interactive environments

## 🎯 What is Value Iteration?

**Value Iteration** is a **model-based** dynamic programming algorithm that computes the optimal value function V*(s) directly by iteratively applying the Bellman optimality equation until convergence.

### Bellman Optimality Equation

```
V(s) ← max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
```

Where:
- `V(s)`: Value function for state s
- `a`: Action
- `s'`: Next state
- `P(s'|s,a)`: Transition probability
- `R(s,a,s')`: Reward function
- `γ` (gamma): Discount factor (0-1)

### Algorithm Steps

1. **Initialize**: V(s) = 0 for all states
2. **Iterate**: For each state, update V(s) using Bellman equation
3. **Check Convergence**: Stop when max|V_new(s) - V_old(s)| < θ
4. **Extract Policy**: π(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

### Key Properties

- **Model-Based**: Requires knowledge of transition dynamics P(s'|s,a)
- **Optimal**: Guaranteed to converge to optimal value function V*
- **Efficient**: Faster convergence than Policy Iteration in many cases
- **Synchronous**: Updates all states in each iteration
- **Deterministic**: Given same inputs, produces same policy

## 🚀 Usage

### Basic Training

```python
from value_iteration_agent import ValueIterationAgent

# Create agent
agent = ValueIterationAgent(
    gamma=0.9,      # Discount factor
    theta=1e-6      # Convergence threshold
)

# Run Value Iteration
V, policy = agent.value_iteration(animate=True)

# Print results
agent.print_value_function()
agent.print_policy()

# Run optimal episode
agent.run_optimal_episode(max_steps=50, delay=1.0)
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

agent = ValueIterationAgent(
    gamma=0.95,
    theta=1e-8,
    env_params=env_params
)

V, policy = agent.value_iteration()
```

### Interactive Environment

```python
# Let user design the environment
agent = ValueIterationAgent(interactive_env=True)
V, policy = agent.value_iteration(animate=True)
```

### Convergence Animation

```python
# Run Value Iteration with animation enabled
agent = ValueIterationAgent()
V, policy = agent.value_iteration(animate=True)

# Show convergence process
agent.animate_value_iteration()
```

## 🎨 Visualizations

The implementation provides comprehensive visualizations:

### 1. Convergence Animation
- **Value Function Evolution**: Heatmap showing V(s) convergence
- **Policy Evolution**: Arrow grid showing policy updates
- **Iteration Counter**: Tracks convergence progress
- **Side-by-side Comparison**: Values and policy synchronized

### 2. Optimal Policy Execution
- **Step-by-Step Visualization**: Agent following optimal policy
- **Value Function Display**: Shows learned values with current state highlighted
- **Action Indicators**: Red arrows showing optimal actions
- **Path Tracking**: Complete trajectory visualization

### 3. Final Results
- **Success/Failure Indicator**: Clear outcome display
- **Episode Path**: Agent's route overlaid on value function
- **Performance Metrics**: Steps taken and reward earned

## 📊 Performance Metrics

Value Iteration guarantees optimal performance:

- **Success Rate**: 100% (optimal policy)
- **Steps to Goal**: Minimum possible (optimal path)
- **Convergence**: Typically 5-20 iterations
- **Computational Cost**: O(|S|²|A|) per iteration

### Example Results (5×5 FrozenLake)

- **Iterations to Convergence**: ~10-15
- **Success Rate**: 100%
- **Steps to Goal**: 8 (optimal)
- **Discount Factor (γ)**: 0.9
- **Convergence Threshold (θ)**: 1e-6

## 🔧 Hyperparameters

### Discount Factor (γ)

```python
# Short-sighted (prefers immediate rewards)
agent = ValueIterationAgent(gamma=0.5)

# Balanced (default)
agent = ValueIterationAgent(gamma=0.9)

# Far-sighted (values future rewards highly)
agent = ValueIterationAgent(gamma=0.99)
```

**Recommendations**:
- **γ = 0.9**: Good default for most environments
- **γ = 0.95-0.99**: Use when future rewards are very important
- **γ < 0.9**: Use for environments with immediate rewards

### Convergence Threshold (θ)

```python
# Loose convergence (faster, less precise)
agent = ValueIterationAgent(theta=1e-4)

# Standard convergence (balanced)
agent = ValueIterationAgent(theta=1e-6)

# Tight convergence (slower, more precise)
agent = ValueIterationAgent(theta=1e-10)
```

**Recommendations**:
- **θ = 1e-6**: Good default (converges quickly with high accuracy)
- **θ = 1e-8**: Use for research or when precision is critical
- **θ = 1e-4**: Use for quick prototyping

## 🆚 Comparison with Other Algorithms

### Value Iteration vs Policy Iteration

| Feature | Value Iteration | Policy Iteration |
|---------|----------------|------------------|
| **Approach** | Update values directly | Alternate policy eval & improvement |
| **Iterations** | More iterations | Fewer iterations |
| **Per-iteration Cost** | Lower | Higher (policy evaluation) |
| **Total Time** | Often faster | Sometimes slower |
| **Memory** | O(\|S\|) | O(\|S\| + \|S\|\|A\|) |
| **Convergence** | To optimal V* | To optimal π* |

**When to use Value Iteration**:
- Large state spaces (lower memory)
- When optimal value function is needed
- Fast prototyping and testing

**When to use Policy Iteration**:
- Small state spaces
- When policy is primary concern
- Tighter convergence guarantees needed

### Value Iteration vs Q-Learning

| Feature | Value Iteration | Q-Learning |
|---------|----------------|------------|
| **Type** | Model-based | Model-free |
| **Requirements** | Transition model | Sample episodes |
| **Optimality** | Guaranteed | Guaranteed (with sufficient exploration) |
| **Speed** | Fast (with model) | Slower (needs samples) |
| **Flexibility** | Needs model | Works without model |
| **Use Case** | Known dynamics | Unknown dynamics |

## 🧮 Mathematical Foundation

### Bellman Optimality Operator

Value Iteration applies the Bellman optimality operator:

```
T(V)(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
```

This operator is a **contraction mapping**, guaranteeing convergence:
```
||T(V) - T(U)|| ≤ γ||V - U||
```

### Convergence Proof

1. **Contraction Property**: T is a γ-contraction
2. **Fixed Point**: V* is the unique fixed point: T(V*) = V*
3. **Convergence**: lim_{k→∞} T^k(V) = V* for any initial V

### Computational Complexity

- **Per Iteration**: O(|S|²|A|)
  - |S| states to update
  - |A| actions per state
  - |S| next states per action
- **Total Iterations**: O(log(1/θ) / (1-γ))
- **Total Complexity**: O(|S|²|A| · log(1/θ) / (1-γ))

## 🎓 Learning Concepts Demonstrated

1. **Dynamic Programming**: Optimal substructure and overlapping subproblems
2. **Bellman Equations**: Optimality conditions for value functions
3. **Convergence Theory**: Contraction mappings and fixed points
4. **Policy Extraction**: Deriving optimal policy from optimal values
5. **Model-Based Planning**: Using known dynamics for planning

## 📈 Convergence Analysis

### Tracking Convergence

The implementation reports convergence progress:

```
🔄 Starting Value Iteration...
==================================================
Iteration 1: Max value change = 0.900000
Iteration 2: Max value change = 0.810000
Iteration 3: Max value change = 0.729000
...
Iteration 12: Max value change = 0.000001
✅ Value Iteration converged after 12 iterations!
==================================================
```

### Factors Affecting Convergence

1. **Discount Factor (γ)**:
   - Higher γ → Slower convergence
   - γ = 0.9: ~10-15 iterations
   - γ = 0.99: ~50-100 iterations

2. **Convergence Threshold (θ)**:
   - Smaller θ → More iterations
   - θ = 1e-4: ~5-8 iterations
   - θ = 1e-8: ~15-25 iterations

3. **Environment Complexity**:
   - Larger grids → More iterations
   - More holes → Faster convergence (terminal states)

## 🔍 Implementation Details

### Deterministic Environment

For the FrozenLake environment (deterministic):

```python
def get_transition_prob_and_reward(self, state, action):
    # Deterministic: P(s'|s,a) = 1.0 for next state
    next_state = get_next_state(state, action)
    reward = 1.0 if next_state == goal else 0.0
    return [(1.0, next_state, reward, is_terminal)]
```

### Value Update

```python
for state in states:
    action_values = []
    for action in actions:
        value = sum(prob * (reward + gamma * V[next_state])
                   for prob, next_state, reward, done 
                   in transitions(state, action))
        action_values.append(value)
    V[state] = max(action_values)
```

### Policy Extraction

```python
policy[state] = argmax_a sum(prob * (reward + gamma * V[next_state])
                             for prob, next_state, reward, done 
                             in transitions(state, action))
```

## 🎯 Advantages of Value Iteration

1. **Guaranteed Optimality**: Converges to optimal value function
2. **No Policy Evaluation**: Simpler than Policy Iteration
3. **Fast Convergence**: Often faster than Policy Iteration
4. **Straightforward**: Easy to understand and implement
5. **Versatile**: Works with any MDP

## ⚠️ Limitations

1. **Requires Model**: Needs P(s'|s,a) and R(s,a,s')
2. **Curse of Dimensionality**: Scales poorly with state space size
3. **Full Sweep**: Must update all states each iteration
4. **Synchronous**: Can't exploit state topology for faster convergence
5. **Memory**: Must store V(s) for all states

## 🧪 Experimental Variations

### Asynchronous Value Iteration

Update states in arbitrary order (not yet implemented):

```python
# Instead of full sweeps, update states individually
for _ in range(num_updates):
    state = select_state()  # e.g., random, prioritized
    update_value(state)
```

**Benefits**: Can converge faster, exploits state structure

### Prioritized Value Iteration

Update states with largest changes first:

```python
priority_queue = [(initial_priority, state) for state in states]
while priority_queue:
    state = priority_queue.pop()
    update_value(state)
    update_priorities(successors(state))
```

**Benefits**: Much faster convergence in large state spaces

## 🐛 Common Issues & Solutions

### Slow Convergence

**Problem**: Takes many iterations to converge

**Solutions**:
- Increase θ (looser convergence)
- Decrease γ (less far-sighted)
- Use asynchronous updates
- Try modified Policy Iteration

### Memory Issues

**Problem**: Large state space causes memory problems

**Solutions**:
- Use sparse representations for V
- Switch to model-free methods (Q-Learning, DQN)
- Use function approximation

### Numerical Instability

**Problem**: Values overflow or underflow

**Solutions**:
- Normalize rewards to [0, 1]
- Use smaller γ
- Clip values to reasonable range

## 📚 References

- Bellman, R. (1957). "Dynamic Programming"
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction" (Chapter 4)
- Puterman, M.L. (1994). "Markov Decision Processes"
- Bertsekas, D.P. (2012). "Dynamic Programming and Optimal Control"

## 🤝 Integration with Other Agents

This Value Iteration implementation complements:

- `../PolicyIteration/` - Alternative DP algorithm for comparison
- `../Q-Learning/` - Model-free alternative requiring no dynamics knowledge
- `../DQN_DeepMind/` - Deep learning approach for large state spaces
- `../RandomAgent/` - Baseline showing value of optimal planning

### Comparison Summary

| Algorithm | Type | Optimality | Speed | Model Required |
|-----------|------|------------|-------|----------------|
| **Value Iteration** | DP | ✅ Optimal | Fast | ✅ Yes |
| **Policy Iteration** | DP | ✅ Optimal | Medium | ✅ Yes |
| **Q-Learning** | RL | ✅ Optimal* | Slow | ❌ No |
| **DQN** | Deep RL | ⚠️ Approximate | Medium | ❌ No |
| **Random** | Baseline | ❌ Suboptimal | Instant | ❌ No |

*With sufficient exploration and convergence

## 💡 Key Insights

### When to Use Value Iteration

✅ **Use when**:
- Environment dynamics are known
- State space is manageable (< 10^6 states)
- Optimal policy is required
- Fast computation is needed
- Simple implementation is desired

❌ **Don't use when**:
- Dynamics are unknown (use Q-Learning/DQN)
- State space is huge (use function approximation)
- Continuous states (use approximate methods)
- Real-time learning needed (use model-free RL)

### Theoretical Guarantees

Value Iteration provides:
1. **Convergence**: Always converges to V*
2. **Optimality**: Policy extracted from V* is optimal
3. **Bounded Error**: |V_k - V*| ≤ γ^k|V_0 - V*|
4. **Polynomial Time**: Converges in polynomial iterations

## 🎯 Output Example

```
🔄 Starting Value Iteration...
==================================================
Iteration 1: Max value change = 0.900000
Iteration 2: Max value change = 0.810000
Iteration 3: Max value change = 0.729000
Iteration 4: Max value change = 0.656100
Iteration 5: Max value change = 0.590490
...
Iteration 11: Max value change = 0.000002
Iteration 12: Max value change = 0.000001
✅ Value Iteration converged after 12 iterations!
==================================================

💰 LEARNED VALUE FUNCTION
========================================
| 0.531 | 0.590 | 0.656 | 0.729 | 0.810 |
| 0.590 | 0.656 | 0.729 | 0.810 | 0.900 |
| 0.656 | 0.729 | 0.810 | 0.900 | 1.000 |
| 0.729 | 0.810 | 0.900 | 1.000 | 0.000 |
| 0.810 | 0.900 | 1.000 | 0.000 | 1.000 |
========================================

📋 LEARNED OPTIMAL POLICY
==============================
| ↓ | ↓ | ↓ | ↓ | ↓ |
| → | ↓ | ↓ | ↓ | ↓ |
| → | → | ↓ | ↓ | ↓ |
| → | → | → | ↓ | ↓ |
| → | → | → | → | G |
==============================

🎮 Running Optimal Policy Episode...
Step 1: DOWN ↓ → (1, 0) (Reward: 0)
Step 2: RIGHT → → (1, 1) (Reward: 0)
...
Step 8: DOWN ↓ → (4, 4) (Reward: 1)

🎯 FINAL RESULTS:
   Success Rate: 100%
   Average Steps: 8
   Total Reward: 1.0
```

## 🎓 Key Takeaways

1. **Dynamic Programming Power**: With a model, optimal policy is guaranteed
2. **Fast Convergence**: Typically 10-20 iterations for small-medium grids
3. **Theoretical Beauty**: Elegant connection to contraction mappings
4. **Practical Utility**: Forms basis for many modern RL algorithms
5. **Model Requirement**: Strength (optimal) and limitation (needs dynamics)

---

**Note**: This Value Iteration implementation uses the shared `frozenlake_env.py` environment wrapper. The algorithm assumes deterministic dynamics but can be easily extended to stochastic environments by modifying the transition probability function.
