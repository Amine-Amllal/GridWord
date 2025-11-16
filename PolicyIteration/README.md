# PolicyIteration - Dynamic Programming for Optimal Policy

This folder contains an implementation of the **Policy Iteration** algorithm, a classic dynamic programming method for solving Markov Decision Processes (MDPs). Policy Iteration finds the optimal policy through iterative cycles of policy evaluation and policy improvement.

## 📁 Contents

### Main Files

- **`policy_iteration_agent.py`** - Complete Policy Iteration implementation featuring:
  - Policy evaluation using iterative methods
  - Policy improvement based on value function
  - Animated visualization of convergence process
  - Interactive episode execution with learned policy
  - Comprehensive statistics and metrics

## 🧮 Algorithm Overview

Policy Iteration is a dynamic programming algorithm that alternates between two steps:

### 1. Policy Evaluation
Computes the value function V(s) for the current policy π:
```
V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
```

Iteratively updates values until convergence (when changes are below threshold θ).

### 2. Policy Improvement
Updates the policy to be greedy with respect to the current value function:
```
π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
```

The algorithm **terminates** when the policy no longer changes (policy stable).

## 🚀 Key Features

### 1. **Guaranteed Optimal Solution**
- Finds the exact optimal policy (not an approximation)
- Converges in finite iterations for finite MDPs
- No exploration needed - uses model-based approach

### 2. **Comprehensive Visualization**
- **Real-time convergence animation** showing:
  - Value function evolution during evaluation
  - Policy changes during improvement
  - Difference metrics between iterations
  - Step-by-step convergence process
- **Interactive policy execution** with visual feedback
- **Final results display** with path overlay

### 3. **Detailed Tracking**
- Iteration count
- Evaluation steps per iteration
- Value function history
- Policy change history
- Convergence metrics

### 4. **Educational Features**
- Clear console output for each iteration
- Visual representation of value propagation
- Policy comparison across iterations
- Annotated visualizations

## 🎮 Usage

### Basic Usage

```bash
# Run from the project root directory
python PolicyIteration/policy_iteration_agent.py
```

### Customization

Edit the `main()` function or create your own script:

```python
from PolicyIteration.policy_iteration_agent import PolicyIterationAgent

# Environment configuration
env_params = {
    'nrow': 5,              # Grid height
    'ncol': 5,              # Grid width
    'holes': [(1, 1), (2, 3), (3, 2)],  # Obstacle positions
    'goal': (4, 4),         # Goal position
    'start_state': (0, 0)   # Starting position
}

# Create agent
agent = PolicyIterationAgent(
    gamma=0.9,              # Discount factor (0-1)
    theta=1e-6,             # Convergence threshold for evaluation
    env_params=env_params   # Environment configuration
)

# Run Policy Iteration
V, policy = agent.policy_iteration(animate=True)

# Display results
agent.print_value_function()
agent.print_policy()

# Show convergence animation
agent.animate_policy_iteration()

# Test the learned policy
agent.run_optimal_episode(max_steps=50, delay=0.8)
```

### Interactive Mode

```python
# Create environment interactively (prompts for grid size, goal, obstacles)
agent = PolicyIterationAgent(interactive_env=True)
```

## 📊 Expected Output

### Console Output

The agent provides detailed console feedback:

```
🔄 Starting Policy Iteration...
==================================================

--- Iteration 1 ---
🔍 Policy Evaluation...
   Converged in 12 evaluation steps
🔧 Policy Improvement...
   Policy stable: False

--- Iteration 2 ---
🔍 Policy Evaluation...
   Converged in 8 evaluation steps
🔧 Policy Improvement...
   Policy stable: False

...

✅ Policy Iteration converged after 4 iterations!
==================================================
```

### Visualizations

1. **Convergence Animation** - Shows 4 synchronized panels:
   - **Value Function**: Heatmap with numerical values
   - **Current Policy**: Arrows showing action for each state
   - **Value Change**: Difference from previous evaluation step
   - **Policy Changes**: Highlights states where policy changed

2. **Optimal Episode Execution** - Real-time display:
   - Current state with agent position
   - Optimal action arrow
   - Value function with current state highlighted
   - Episode statistics

3. **Final Results** - Summary display:
   - Complete path taken
   - Final state and success status
   - Total reward and steps
   - Value function overlay

### Learned Policy

```
📋 LEARNED OPTIMAL POLICY
==============================
| ↓ | → | → | → | ↓ |
| ↓ | H | → | ↓ | ↓ |
| → | → | → | H | ↓ |
| ↑ | → | H | ↓ | ↓ |
| ↑ | → | → | → | G |
==============================
Legend: ← LEFT, ↓ DOWN, → RIGHT, ↑ UP, G GOAL, H HOLE
```

### Value Function

```
💰 LEARNED VALUE FUNCTION
========================================
| 0.590 | 0.656 | 0.729 | 0.810 | 0.729 |
| 0.531 | 0.000 | 0.810 | 0.729 | 0.656 |
| 0.590 | 0.656 | 0.729 | 0.000 | 0.590 |
| 0.531 | 0.590 | 0.000 | 0.729 | 0.656 |
| 0.478 | 0.531 | 0.590 | 0.810 | 1.000 |
========================================
```

## 🔬 How It Works

### Initialization
1. Create random policy π₀
2. Initialize value function V(s) = 0 for all states

### Main Loop
```
Repeat:
    1. Policy Evaluation:
       - Iteratively compute V^π(s) for all states
       - Continue until convergence (Δ < θ)
    
    2. Policy Improvement:
       - For each state s:
         - Compute Q(s,a) for all actions a
         - Set π(s) = argmax_a Q(s,a)
       - Check if policy changed
    
    Until: Policy is stable (no changes)
```

### Termination
The algorithm guarantees:
- **Convergence** to optimal policy π*
- **Finite iterations** for finite MDPs
- **Optimal value function** V*

## ⚡ Performance Characteristics

### Advantages
- ✅ **Finds exact optimal policy** (not an approximation)
- ✅ **Guaranteed convergence** in finite iterations
- ✅ **No exploration required** (uses model knowledge)
- ✅ **Faster than Value Iteration** in many cases
- ✅ **Each iteration produces a complete policy**

### Considerations
- ⚠️ **Requires model knowledge** (transition probabilities, rewards)
- ⚠️ **Computationally expensive** for large state spaces
- ⚠️ **Not suitable for** model-free scenarios
- ⚠️ **Memory intensive** for very large MDPs

### Complexity
- **Time per iteration**: O(|S|²|A|) for policy evaluation + O(|S||A|) for improvement
- **Space**: O(|S|) for value function + O(|S|) for policy
- **Iterations**: Typically fewer than Value Iteration

Where:
- |S| = number of states
- |A| = number of actions

## 🎯 When to Use Policy Iteration

**Best for:**
- ✅ Small to medium-sized MDPs
- ✅ When model (transitions/rewards) is known
- ✅ Problems requiring exact optimal solutions
- ✅ Educational purposes and algorithm comparison

**Not recommended for:**
- ❌ Very large state spaces (use approximation methods)
- ❌ Model-free scenarios (use Q-Learning, DQN)
- ❌ Continuous state spaces (use function approximation)
- ❌ Real-time applications (too slow)

## 📊 Comparison with Other Methods

| Feature | Policy Iteration | Value Iteration | Q-Learning |
|---------|-----------------|-----------------|------------|
| **Type** | Model-based DP | Model-based DP | Model-free RL |
| **Optimality** | ✅ Guaranteed | ✅ Guaranteed | 🟡 Converges |
| **Model Required** | ✅ Yes | ✅ Yes | ❌ No |
| **Convergence** | Fast | Medium | Slow |
| **Iterations** | Fewer | More | Many episodes |
| **Per Iteration** | Expensive | Cheaper | Varies |
| **Exploration** | ❌ Not needed | ❌ Not needed | ✅ Required |
| **Scalability** | 🔴 Limited | 🔴 Limited | 🟡 Better |

## 🔧 Hyperparameters

### Gamma (γ) - Discount Factor
- **Range**: 0.0 to 1.0
- **Default**: 0.9
- **Effect**: 
  - Higher (→1.0): Values long-term rewards more
  - Lower (→0.0): Focuses on immediate rewards
- **Recommendation**: 0.9-0.99 for most problems

### Theta (θ) - Convergence Threshold
- **Range**: 1e-8 to 1e-3
- **Default**: 1e-6
- **Effect**:
  - Smaller: More accurate but slower convergence
  - Larger: Faster but less accurate
- **Recommendation**: 1e-6 for good balance

## 📈 Monitoring Convergence

**Signs of good convergence:**
1. **Decreasing evaluation steps** per iteration
2. **Fewer policy changes** in later iterations
3. **Value function stabilization**
4. **Consistent optimal path** in test episodes

**Troubleshooting:**
- **Not converging**: Check if MDP has cycles, adjust θ
- **Too many iterations**: Increase θ slightly
- **Suboptimal policy**: Check environment model correctness

## 💡 Tips for Best Results

1. **Appropriate gamma**: Use 0.9-0.99 for most grid worlds
2. **Sufficient precision**: θ = 1e-6 usually sufficient
3. **Verify model**: Ensure transitions and rewards are correct
4. **Watch animations**: Understand how policy evolves
5. **Compare with Value Iteration**: Use both to verify results

## 🔄 Algorithm Variants

This implementation uses:
- **Iterative Policy Evaluation** (not linear system solving)
- **Deterministic policy** (single action per state)
- **Synchronous updates** (all states updated together)

Alternative approaches:
- **Modified Policy Iteration**: Truncate evaluation early
- **Asynchronous updates**: Update states in different order
- **Stochastic policies**: For exploration or continuous actions

## 📚 Further Reading

- **Sutton & Barto** - Reinforcement Learning: An Introduction (Chapter 4)
  - Comprehensive coverage of Policy Iteration
  - Theoretical foundations and convergence proofs
  
- **Bellman, R.** (1957) - Dynamic Programming
  - Original formulation of the method
  
- **Puterman, M.** - Markov Decision Processes
  - Advanced theoretical treatment

## 🧪 Example Use Cases

1. **Grid World Navigation**: Finding shortest paths (this implementation)
2. **Inventory Management**: Optimal stocking policies
3. **Robot Control**: Optimal action sequences
4. **Game AI**: Perfect play in fully observable games
5. **Resource Allocation**: Optimal scheduling

---

**Last Updated**: November 16, 2025
