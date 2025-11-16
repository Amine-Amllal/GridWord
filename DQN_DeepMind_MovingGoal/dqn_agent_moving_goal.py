import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import os
import sys
from datetime import datetime
from collections import deque, defaultdict
import random

# Add parent directory to path to import base DQN and environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DQN_DeepMind.dqn_agent import DeepMindDQN
from frozenlake_env import FrozenLakeEnv


class PrioritizedReplayMemory:
    """
    Optimized Prioritized Experience Replay Memory.
    
    Features:
    - Prioritized sampling based on TD-error
    - Success-weighted sampling (successful trajectories sampled more)
    - Goal-balanced sampling (ensures diverse goal coverage)
    - Efficient numpy-based storage
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            capacity: Maximum memory size
            alpha: Prioritization exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling weight (anneals to 1)
            beta_increment: Beta annealing rate
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.position = 0
        self.size = 0
        
        # Storage arrays for efficient access
        self.agent_positions = np.zeros((capacity, 2), dtype=np.int8)
        self.goal_positions = np.zeros((capacity, 2), dtype=np.int8)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_agent_positions = np.zeros((capacity, 2), dtype=np.int8)
        self.next_goal_positions = np.zeros((capacity, 2), dtype=np.int8)
        self.dones = np.zeros(capacity, dtype=bool)
        
        # Priority arrays
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # Success tracking for weighted sampling
        self.is_success = np.zeros(capacity, dtype=bool)
        
        # Goal tracking for balanced sampling
        self.goal_indices = defaultdict(list)
        
    def add(self, agent_pos, goal_pos, action, reward, next_agent_pos, next_goal_pos, done):
        """Add experience to memory with max priority."""
        idx = self.position
        
        # Store experience
        self.agent_positions[idx] = agent_pos
        self.goal_positions[idx] = goal_pos
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_agent_positions[idx] = next_agent_pos
        self.next_goal_positions[idx] = next_goal_pos
        self.dones[idx] = done
        
        # Assign max priority to new experiences
        self.priorities[idx] = self.max_priority
        
        # Track success
        self.is_success[idx] = reward > 0
        
        # Track goal for balanced sampling
        goal_tuple = tuple(goal_pos)
        if idx in self.goal_indices[goal_tuple]:
            self.goal_indices[goal_tuple].remove(idx)
        self.goal_indices[goal_tuple].append(idx)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, use_prioritization=True, success_boost=2.0):
        """
        Sample batch with optional prioritization and success boosting.
        
        Args:
            batch_size: Number of samples
            use_prioritization: Use TD-error priorities
            success_boost: Multiplier for successful experience probabilities
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        # Calculate sampling probabilities
        if use_prioritization:
            priorities = self.priorities[:self.size] ** self.alpha
        else:
            priorities = np.ones(self.size)
        
        # Boost successful experiences
        success_mask = self.is_success[:self.size]
        priorities = np.where(success_mask, priorities * success_boost, priorities)
        
        # Normalize to probabilities
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Retrieve experiences
        experiences = (
            self.agent_positions[indices],
            self.goal_positions[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_agent_positions[indices],
            self.next_goal_positions[indices],
            self.dones[indices]
        )
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors."""
        priorities = np.abs(td_errors) + 1e-6  # Small constant to avoid zero priority
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def sample_goal_balanced(self, batch_size, min_goals=3):
        """
        Sample batch ensuring diverse goal coverage.
        
        Args:
            batch_size: Number of samples
            min_goals: Minimum number of different goals in batch
        """
        if len(self.goal_indices) < min_goals:
            # Not enough goals yet, use regular sampling
            return self.sample(batch_size)
        
        # Sample from different goals
        goals = list(self.goal_indices.keys())
        samples_per_goal = max(1, batch_size // len(goals))
        
        indices = []
        for goal in goals:
            goal_samples = self.goal_indices[goal]
            if len(goal_samples) > 0:
                # Sample with priority weighting from this goal
                n_samples = min(samples_per_goal, len(goal_samples))
                goal_priorities = self.priorities[goal_samples] ** self.alpha
                
                # Boost successful experiences
                success_mask = self.is_success[goal_samples]
                goal_priorities = np.where(success_mask, goal_priorities * 2.0, goal_priorities)
                
                goal_probs = goal_priorities / goal_priorities.sum()
                sampled = np.random.choice(goal_samples, size=n_samples, replace=False, p=goal_probs)
                indices.extend(sampled)
        
        # If we need more samples, fill with regular sampling
        if len(indices) < batch_size:
            remaining = batch_size - len(indices)
            all_indices = set(range(self.size))
            available = list(all_indices - set(indices))
            if len(available) > 0:
                priorities = self.priorities[available] ** self.alpha
                probs = priorities / priorities.sum()
                extra = np.random.choice(available, size=min(remaining, len(available)), 
                                        replace=False, p=probs)
                indices.extend(extra)
        
        indices = np.array(indices[:batch_size])
        
        # Calculate importance weights
        probabilities = (self.priorities[indices] ** self.alpha) / (self.priorities[:self.size] ** self.alpha).sum()
        weights = (self.size * probabilities) ** (-self.beta)
        weights /= weights.max()
        
        # Retrieve experiences
        experiences = (
            self.agent_positions[indices],
            self.goal_positions[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_agent_positions[indices],
            self.next_goal_positions[indices],
            self.dones[indices]
        )
        
        return experiences, indices, weights
    
    def __len__(self):
        return self.size


class MovingGoalFrozenLake(FrozenLakeEnv):
    """Extended FrozenLake with moving goal capability."""
    
    def __init__(self, nrow=5, ncol=5, holes=None, start_state=(0, 0), 
                 goal_positions=None, random_goal=True):
        self.random_goal = random_goal
        self.goal_cycle_index = 0
        
        # Determine valid goal positions
        if goal_positions is None:
            goal_positions = []
            for i in range(nrow):
                for j in range(ncol):
                    pos = (i, j)
                    if pos != start_state and (holes is None or pos not in holes):
                        goal_positions.append(pos)
        
        self.goal_positions = goal_positions
        if len(self.goal_positions) == 0:
            raise ValueError("Must have at least one valid goal position!")
        
        # Shuffle goal positions for better randomization
        np.random.shuffle(self.goal_positions)
        
        initial_goal = self.goal_positions[0]
        super().__init__(nrow=nrow, ncol=ncol, holes=holes, 
                        goal=initial_goal, start_state=start_state)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        # Select new goal
        if self.random_goal:
            goal_idx = np.random.randint(0, len(self.goal_positions))
            new_goal = self.goal_positions[goal_idx]
        else:
            new_goal = self.goal_positions[self.goal_cycle_index]
            self.goal_cycle_index = (self.goal_cycle_index + 1) % len(self.goal_positions)
        
        # Update goal in grid
        old_goal = self.goal
        if old_goal != self.start_state and old_goal not in self.holes:
            self.desc[old_goal[0], old_goal[1]] = 'F'
        if old_goal in self.terminal_states:
            self.terminal_states.remove(old_goal)
        
        self.goal = new_goal
        self.desc[new_goal[0], new_goal[1]] = 'G'
        if new_goal not in self.terminal_states:
            self.terminal_states.append(new_goal)
        
        self.state = self.start_state
        self.last_action = None
        
        return self.state, {'goal': self.goal}


class DQNAgentMovingGoal:
    """
    Deep Q-Network Agent adapted for FrozenLake with Moving Goal.
    
    Key adaptations from the original DQN:
    - State encoding includes both agent position AND goal position
    - Input size = 2 * grid_size (agent position + goal position, both one-hot)
    - Agent learns general navigation policy rather than path to fixed goal
    - Training handles goal changes across episodes automatically
    
    This agent reuses the DeepMindDQN neural network class from the original
    implementation, only modifying the state encoding and training loop.
    """
    
    def __init__(self, alpha=0.00025, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, hidden_layers=[256, 256], env_params=None,
                 batch_size=32, memory_size=10000, target_update_freq=1000):
        """
        Initialize DQN Agent for Moving Goal environment.
        
        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum epsilon
            hidden_layers (list): Network architecture
            env_params (dict): Environment parameters for moving goal setup
            batch_size (int): Minibatch size for training
            memory_size (int): Replay memory size
            target_update_freq (int): Steps between target network updates
        """
        # Create moving goal environment
        if env_params:
            self.env = MovingGoalFrozenLake(**env_params)
        else:
            self.env = MovingGoalFrozenLake()
        
        # Adapt hyperparameters based on environment size
        env_size = self.env.nrow * self.env.ncol
        
        # Use smaller network for smaller environments
        if env_size <= 16 and hidden_layers == [256, 256]:
            hidden_layers = [128, 64]
            print(f"📐 Auto-adjusted network size to {hidden_layers} for {self.env.nrow}x{self.env.ncol} environment")
        elif env_size <= 36 and hidden_layers == [256, 256]:
            hidden_layers = [128, 128]
            print(f"📐 Auto-adjusted network size to {hidden_layers} for {self.env.nrow}x{self.env.ncol} environment")
        
        # Increase learning rate for moving goal (larger state space)
        if env_size <= 16 and alpha == 0.00025:
            alpha = 0.002  # Increased from 0.001
            print(f"⚡ Auto-adjusted learning rate to {alpha} for faster convergence")
        elif env_size <= 36 and alpha == 0.00025:
            alpha = 0.001  # Increased from 0.0005
            print(f"⚡ Auto-adjusted learning rate to {alpha} for faster convergence")
        
        # Slower epsilon decay for moving goal (more exploration needed)
        if epsilon_decay == 0.995:
            epsilon_decay = 0.997  # Slower decay
            print(f"🔄 Auto-adjusted epsilon decay to {epsilon_decay} for better exploration")
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        
        # Action mappings
        self.action_mapping = {0: "LEFT ←", 1: "DOWN ↓", 2: "RIGHT →", 3: "UP ↑"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        # State encoding: agent position (one-hot) + goal position (one-hot)
        # This allows the network to learn general navigation to any goal
        self.grid_size = self.env.nrow * self.env.ncol
        self.input_size = 2 * self.grid_size  # Agent pos + Goal pos
        self.output_size = 4  # 4 possible actions
        
        # Initialize Q-network and Target network
        self.q_network = DeepMindDQN(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=hidden_layers,
            learning_rate=alpha
        )
        
        self.target_network = DeepMindDQN(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=hidden_layers,
            learning_rate=alpha
        )
        
        # Initialize target network with same weights as Q-network
        self.target_network.copy_weights_from(self.q_network)
        
        # Optimized Prioritized Experience Replay Memory
        self.memory = PrioritizedReplayMemory(
            capacity=memory_size,
            alpha=0.6,  # Prioritization strength
            beta=0.4,   # Importance sampling (anneals to 1)
            beta_increment=0.001
        )
        
        # Training tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_goals = []  # Track which goals were reached
        self.loss_history = []
        self.epsilon_history = []
        self.steps_done = 0
        self.use_goal_balanced_sampling = True  # Enable goal-balanced sampling
        
        print("🤖 DQN Agent for Moving Goal Initialized")
        print("=" * 70)
        print(f"Environment: {self.env.nrow}x{self.env.ncol} FrozenLake with Moving Goal")
        print(f"Network Input: {self.input_size} (agent pos + goal pos, both one-hot)")
        print(f"Network: {self.input_size} → {' → '.join(map(str, hidden_layers))} → {self.output_size}")
        print(f"Learning Rate: {alpha}")
        print(f"Discount Factor (γ): {gamma}")
        print(f"Batch Size: {batch_size}")
        print(f"Memory Size: {memory_size}")
        print(f"Target Update Frequency: {target_update_freq} steps")
        print("\n🎯 Goal Configuration:")
        print(f"  ✓ Total possible goal positions: {len(self.env.goal_positions)}")
        print(f"  ✓ Goal selection: Random from entire grid")
        
        # Show sample of goal positions to confirm diversity
        sample_goals = self.env.goal_positions[:min(5, len(self.env.goal_positions))]
        print(f"  ✓ Sample goals: {sample_goals}")
        
        # Show goal distribution across rows
        goal_rows = [g[0] for g in self.env.goal_positions]
        unique_rows = len(set(goal_rows))
        print(f"  ✓ Goals distributed across {unique_rows}/{self.env.nrow} rows")
        
        print("\n🚀 Optimized Replay Memory Features:")
        print(f"  ✓ Prioritized Experience Replay (α={self.memory.alpha})")
        print(f"  ✓ Importance Sampling (β={self.memory.beta}→1.0)")
        print(f"  ✓ Success-weighted Sampling (2× boost)")
        print(f"  ✓ Goal-balanced Sampling (diverse coverage)")
        print(f"  ✓ Efficient numpy-based storage")
        print("=" * 70)
    
    def encode_state(self, agent_pos, goal_pos):
        """
        Encode state as concatenation of two one-hot vectors:
        [agent_position_one_hot | goal_position_one_hot]
        
        This encoding allows the network to learn:
        - Where the agent currently is
        - Where the goal currently is
        - General navigation strategy to reach any goal
        
        Args:
            agent_pos (tuple): (row, col) agent position
            goal_pos (tuple): (row, col) goal position
            
        Returns:
            np.array: Encoded state of shape (1, 2*grid_size)
        """
        # One-hot encode agent position
        agent_row, agent_col = agent_pos
        agent_idx = agent_row * self.env.ncol + agent_col
        agent_encoded = np.zeros(self.grid_size)
        agent_encoded[agent_idx] = 1
        
        # One-hot encode goal position
        goal_row, goal_col = goal_pos
        goal_idx = goal_row * self.env.ncol + goal_col
        goal_encoded = np.zeros(self.grid_size)
        goal_encoded[goal_idx] = 1
        
        # Concatenate: [agent_pos | goal_pos]
        state_encoded = np.concatenate([agent_encoded, goal_encoded])
        
        return state_encoded.reshape(1, -1)
    
    def choose_action(self, agent_pos, goal_pos, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            agent_pos (tuple): Current agent position
            goal_pos (tuple): Current goal position
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice([0, 1, 2, 3])
        else:
            # Exploitation: greedy action from Q-network
            encoded_state = self.encode_state(agent_pos, goal_pos)
            q_values = self.q_network.forward(encoded_state)
            return np.argmax(q_values[0])
    
    def store_experience(self, agent_pos, goal_pos, action, reward, 
                        next_agent_pos, next_goal_pos, done):
        """
        Store experience in optimized replay memory.
        
        Args:
            agent_pos: Current agent position
            goal_pos: Current goal position
            action: Action taken
            reward: Reward received
            next_agent_pos: Next agent position
            next_goal_pos: Next goal position (same as goal_pos in current implementation)
            done: Whether episode ended
        """
        self.memory.add(agent_pos, goal_pos, action, reward, 
                       next_agent_pos, next_goal_pos, done)
    
    def sample_batch(self):
        """
        Sample batch from optimized replay memory with prioritization.
        
        Returns:
            tuple: (Batch of experiences, indices, importance weights)
        """
        if len(self.memory) < self.batch_size:
            return None, None, None
        
        # Use goal-balanced sampling for better coverage
        if self.use_goal_balanced_sampling:
            experiences, indices, weights = self.memory.sample_goal_balanced(self.batch_size)
        else:
            experiences, indices, weights = self.memory.sample(self.batch_size)
        
        # Unpack experiences
        agent_pos, goal_pos, actions, rewards, next_agent_pos, next_goal_pos, dones = experiences
        
        # Encode states
        states = np.vstack([self.encode_state(ap, gp) for ap, gp in zip(agent_pos, goal_pos)])
        next_states = np.vstack([self.encode_state(nap, ngp) for nap, ngp in zip(next_agent_pos, next_goal_pos)])
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def train_on_batch(self):
        """
        Train the Q-network on a batch from prioritized replay memory.
        
        Returns:
            float: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch with priorities
        batch_data, indices, weights = self.sample_batch()
        
        if batch_data is None or indices is None or weights is None:
            return 0.0
        
        states, actions, rewards, next_states, dones = batch_data
        
        # Compute target Q-values using target network
        current_q_values = self.q_network.forward(states)
        next_q_values = self.target_network.forward(next_states)
        
        # Compute targets with importance sampling weights
        target_q_values = current_q_values.copy()
        td_errors = []
        
        for i in range(len(states)):
            old_val = current_q_values[i, actions[i]]
            
            if dones[i]:
                target_val = rewards[i]
            else:
                target_val = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            # Apply importance sampling weight to the target update
            # This reduces the update magnitude for less important samples
            weighted_target = old_val + weights[i] * (target_val - old_val)
            target_q_values[i, actions[i]] = weighted_target
            
            # Compute TD-error for priority update (unweighted for proper priority calculation)
            td_error = target_val - old_val
            td_errors.append(td_error)
        
        # Update priorities in memory
        self.memory.update_priorities(indices, np.array(td_errors))
        
        # Train network
        loss = self.q_network.train_step(states, target_q_values)
        
        return loss
    
    def update_target_network(self):
        """Update target network with weights from Q-network."""
        self.target_network.copy_weights_from(self.q_network)
    
    def train(self, num_episodes=1000, verbose=True, save_freq=100):
        """
        Train the DQN agent on moving goal environment.
        
        Args:
            num_episodes (int): Number of training episodes
            verbose (bool): Print training progress
            save_freq (int): Frequency of progress updates
        """
        print("\n🎓 Starting DQN Training with Moving Goal...")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Max Steps per Episode: 200")
        warmup_episodes = max(50, self.batch_size)  # Reduced warmup
        print(f"Warmup Episodes: {warmup_episodes}")
        print("=" * 70)
        
        # Warmup phase: fill replay buffer with random exploration
        
        for episode in range(num_episodes):
            # Reset environment - goal will change each episode
            state, info = self.env.reset()
            goal = info['goal']
            
            total_reward = 0
            steps = 0
            episode_loss = []
            
            terminated = False
            truncated = False
            max_steps = 200
            
            # Use pure exploration during warmup
            if episode < warmup_episodes:
                exploration_epsilon = 1.0
            else:
                exploration_epsilon = self.epsilon
            
            while not (terminated or truncated) and steps < max_steps:
                # Choose action with appropriate exploration rate
                original_epsilon = self.epsilon
                self.epsilon = exploration_epsilon
                action = self.choose_action(state, goal, training=True)
                self.epsilon = original_epsilon
                
                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                
                # Enhanced reward shaping for moving goal
                shaped_reward = reward
                if reward > 0:  # Reached goal
                    shaped_reward = 10.0
                elif terminated:  # Fell in hole
                    shaped_reward = -5.0
                else:  # Normal step
                    # Distance-based reward shaping (more aggressive)
                    curr_dist = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
                    next_dist = abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
                    
                    if next_dist < curr_dist:
                        # Reward proportional to progress
                        shaped_reward = 0.2  # Increased from 0.1
                    elif next_dist > curr_dist:
                        # Small penalty for moving away
                        shaped_reward = -0.05
                    else:
                        # Small penalty for no progress
                        shaped_reward = -0.02
                
                # Store experience
                self.store_experience(state, goal, int(action), shaped_reward,
                                    next_state, goal, terminated or truncated)
                
                # Train more frequently after warmup (multiple training steps per env step)
                if episode >= warmup_episodes and len(self.memory) >= self.batch_size:
                    # Multiple training steps for better convergence
                    for _ in range(2):  # Train 2 times per environment step
                        loss = self.train_on_batch()
                        episode_loss.append(loss)
                
                # Update target network periodically
                self.steps_done += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()
                    if verbose and episode % 50 == 0:
                        print(f"   🎯 Target network updated at step {self.steps_done}")
                
                # Update for next iteration
                total_reward += reward
                steps += 1
                state = next_state
            
            # Store episode results
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.episode_goals.append(goal)
            self.epsilon_history.append(self.epsilon)
            
            if len(episode_loss) > 0:
                self.loss_history.append(np.mean(episode_loss))
            else:
                self.loss_history.append(0)
            
            # Decay epsilon only after warmup
            if episode >= warmup_episodes and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Print progress
            if verbose and (episode + 1) % save_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                avg_loss = np.mean(self.loss_history[-100:]) if len(self.loss_history) > 0 else 0
                success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
                
                # Show unique goals encountered
                recent_goals = set(self.episode_goals[-100:])
                
                print(f"Episode {episode + 1:4d} | "
                      f"Reward: {avg_reward:.3f} | "
                      f"Steps: {avg_steps:.1f} | "
                      f"Success: {success_rate:.1f}% | "
                      f"Loss: {avg_loss:.6f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Goals seen: {len(recent_goals)}")
        
        print("\n✅ DQN Training Complete!")
        print("=" * 70)
        
        # Final statistics
        final_avg_reward = np.mean(self.episode_rewards[-100:])
        final_success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
        unique_goals_trained = len(set(self.episode_goals))
        
        print(f"Final Performance (last 100 episodes):")
        print(f"  Average Reward: {final_avg_reward:.3f}")
        print(f"  Success Rate: {final_success_rate:.1f}%")
        print(f"  Final Epsilon: {self.epsilon:.3f}")
        print(f"  Total Steps: {self.steps_done}")
        print(f"  Unique Goals Encountered: {unique_goals_trained}")
        print(f"  Replay Memory Size: {len(self.memory)}")
    
    def evaluate(self, num_episodes=100, render=False, test_all_goals=False):
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            render (bool): Whether to render episodes
            test_all_goals (bool): Test agent on all possible goal positions
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n🔬 Evaluating DQN Agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_steps = []
        eval_successes = []
        goal_performance = {}  # Track performance per goal position
        
        for ep in range(num_episodes):
            state, info = self.env.reset()
            goal = info['goal']
            
            if goal not in goal_performance:
                goal_performance[goal] = {'attempts': 0, 'successes': 0, 'total_steps': 0}
            
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < 200:
                action = self.choose_action(state, goal, training=False)
                state, reward, terminated, truncated, _ = self.env.step(int(action))
                total_reward += reward
                steps += 1
                
                if render and ep < 5:
                    self.env.render(mode='text')
                    time.sleep(0.5)
            
            # Update goal-specific stats
            goal_performance[goal]['attempts'] += 1
            goal_performance[goal]['total_steps'] += steps
            if total_reward > 0:
                goal_performance[goal]['successes'] += 1
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
            eval_successes.append(total_reward > 0)
        
        # Compute metrics
        metrics = {
            'avg_reward': np.mean(eval_rewards),
            'success_rate': np.mean(eval_successes) * 100,
            'avg_steps': np.mean(eval_steps),
            'std_reward': np.std(eval_rewards),
            'std_steps': np.std(eval_steps),
            'goal_performance': goal_performance
        }
        
        print("=" * 70)
        print("📊 Evaluation Results:")
        print(f"  Average Reward: {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Success Rate: {metrics['success_rate']:.1f}%")
        print(f"  Average Steps: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
        print(f"  Unique Goals Tested: {len(goal_performance)}")
        
        # Show per-goal performance
        if len(goal_performance) <= 10:
            print("\n  Per-Goal Performance:")
            for goal, stats in sorted(goal_performance.items()):
                goal_success_rate = (stats['successes'] / stats['attempts']) * 100
                avg_steps = stats['total_steps'] / stats['attempts']
                print(f"    Goal {goal}: {goal_success_rate:.1f}% success "
                      f"({stats['successes']}/{stats['attempts']}), "
                      f"avg steps: {avg_steps:.1f}")
        
        print("=" * 70)
        
        return metrics
    
    def plot_training_progress(self, save_path=None):
        """
        Plot training progress with moving goal specific metrics.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Episode rewards
        ax1 = axes[0, 0]
        episodes = range(len(self.episode_rewards))
        ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Episode Reward')
        
        # Moving average
        if len(self.episode_rewards) >= 50:
            window = 50
            moving_avg = [np.mean(self.episode_rewards[max(0, i-window):i+1])
                         for i in range(len(self.episode_rewards))]
            ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label='Moving Avg (50)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards (Moving Goal)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # Loss
        ax2 = axes[0, 1]
        ax2.plot(range(len(self.loss_history)), self.loss_history, 'g-', alpha=0.5, linewidth=0.5)
        if len(self.loss_history) >= 50:
            window = 50
            loss_avg = [np.mean(self.loss_history[max(0, i-window):i+1])
                       for i in range(len(self.loss_history))]
            ax2.plot(range(len(loss_avg)), loss_avg, 'darkgreen', linewidth=2, label='Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Success rate
        ax3 = axes[1, 0]
        if len(self.episode_rewards) >= 100:
            window = 100
            success_rates = []
            for i in range(window, len(self.episode_rewards) + 1):
                rate = np.mean([r > 0 for r in self.episode_rewards[i-window:i]]) * 100
                success_rates.append(rate)
            ax3.plot(range(window, len(self.episode_rewards) + 1), success_rates, 'purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Success Rate (100-episode window)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-5, 105)
        
        # Goal diversity (unique goals per window)
        ax4 = axes[1, 1]
        if len(self.episode_goals) >= 100:
            window = 100
            goal_diversity = []
            for i in range(window, len(self.episode_goals) + 1):
                unique_goals = len(set(self.episode_goals[i-window:i]))
                goal_diversity.append(unique_goals)
            ax4.plot(range(window, len(self.episode_goals) + 1), goal_diversity, 'orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Unique Goals Encountered')
        ax4.set_title('Goal Diversity (100-episode window)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_policy(self, test_goal=None, save_path=None):
        """
        Visualize the learned policy for a specific goal position.
        
        Args:
            test_goal (tuple): Goal position to visualize policy for.
                              If None, uses a random valid goal.
            save_path (str): Path to save the visualization
        """
        if test_goal is None:
            test_goal = self.env.goal_positions[0]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for i in range(self.env.nrow + 1):
            ax.axhline(y=i - 0.5, color='black', linewidth=2)
        for j in range(self.env.ncol + 1):
            ax.axvline(x=j - 0.5, color='black', linewidth=2)
        
        # Color cells
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                cell_type = self.env.desc[i, j]
                
                if cell_type == 'S':
                    color = 'lightgreen'
                    text = 'START'
                elif (i, j) == test_goal:
                    color = 'gold'
                    text = 'GOAL'
                elif cell_type == 'H':
                    color = 'red'
                    text = 'HOLE'
                else:
                    color = 'lightblue'
                    text = ''
                
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, alpha=0.5)
                ax.add_patch(rect)
                
                if text:
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=10, fontweight='bold')
        
        # Draw policy arrows for this specific goal
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                if (i, j) not in self.env.terminal_states and (i, j) != test_goal:
                    state = (i, j)
                    action = self.choose_action(state, test_goal, training=False)
                    symbol = self.action_symbols[int(action)]
                    ax.text(j, i, symbol, ha='center', va='center',
                           fontsize=24, fontweight='bold', color='darkblue')
        
        ax.set_xlim(-0.5, self.env.ncol - 0.5)
        ax.set_ylim(-0.5, self.env.nrow - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.env.ncol))
        ax.set_yticks(range(self.env.nrow))
        ax.set_title(f'DQN Learned Policy (Moving Goal) - Target: {test_goal}\n'
                    f'{self.env.nrow}x{self.env.ncol} FrozenLake',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Policy visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_results_folder(self, folder_name=None, include_gif=True):
        """
        Generate comprehensive results folder with plots and statistics.
        
        Args:
            folder_name (str): Name of results folder
            include_gif (bool): Whether to generate agent navigation GIF
            
        Returns:
            str: Path to results folder
        """
        if folder_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"dqn_moving_goal_results_{timestamp}"
        
        # Create folder in current directory
        results_path = os.path.join(os.path.dirname(__file__), folder_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        print(f"\n📁 Generating results in: {results_path}")
        print("=" * 70)
        
        # 1. Training progress plot
        print("📈 Creating training progress plot...")
        self.plot_training_progress(save_path=os.path.join(results_path, 'training_progress.png'))
        
        # 2. Policy visualizations for multiple goals (diverse sampling)
        print("🎯 Creating policy visualizations...")
        num_policy_plots = min(6, len(self.env.goal_positions))
        
        # Select diverse goals from different regions of the grid
        if len(self.env.goal_positions) <= num_policy_plots:
            # Use all available goals
            selected_goals = self.env.goal_positions
        else:
            # Sample goals from different regions for better coverage
            # Divide goal positions into quadrants/regions and sample from each
            selected_goals = []
            nrow, ncol = self.env.nrow, self.env.ncol
            
            # Define regions: top-left, top-right, bottom-left, bottom-right, center
            regions = {
                'top_left': [], 'top_right': [], 
                'bottom_left': [], 'bottom_right': [], 
                'center': []
            }
            
            mid_row, mid_col = nrow // 2, ncol // 2
            
            for goal in self.env.goal_positions:
                r, c = goal
                if r < mid_row and c < mid_col:
                    regions['top_left'].append(goal)
                elif r < mid_row and c >= mid_col:
                    regions['top_right'].append(goal)
                elif r >= mid_row and c < mid_col:
                    regions['bottom_left'].append(goal)
                elif r >= mid_row and c >= mid_col:
                    regions['bottom_right'].append(goal)
                
                # Also track center region (more flexible definition)
                if mid_row - 1 <= r <= mid_row + 1 and mid_col - 1 <= c <= mid_col + 1:
                    regions['center'].append(goal)
            
            # Sample from each region
            samples_per_region = max(1, num_policy_plots // len([r for r in regions.values() if r]))
            
            for region_name, region_goals in regions.items():
                if region_goals and len(selected_goals) < num_policy_plots:
                    # Randomly select from this region
                    sample_count = min(samples_per_region, len(region_goals), 
                                     num_policy_plots - len(selected_goals))
                    selected = np.random.choice(len(region_goals), 
                                              size=sample_count, 
                                              replace=False)
                    for idx in selected:
                        selected_goals.append(region_goals[idx])
        
        # Visualize selected goals
        for goal in selected_goals[:num_policy_plots]:
            self.visualize_policy(
                test_goal=goal,
                save_path=os.path.join(results_path, f'learned_policy_goal_{goal[0]}_{goal[1]}.png')
            )
        
        # 3. Training summary
        print("📝 Creating training summary...")
        self._save_training_summary(results_path)
        
        # 4. Agent navigation GIF (optional)
        if include_gif:
            print("🎬 Creating agent navigation GIF...")
            try:
                gif_path = os.path.join(results_path, 'agent_navigation_moving_goal.gif')
                self._generate_navigation_gif(gif_path, num_episodes=5, fps=2)
            except Exception as e:
                print(f"⚠️ Could not generate GIF: {e}")
                print("💡 Make sure pillow and imageio are installed")
        
        print("\n✅ Results generation complete!")
        print("=" * 70)
        print(f"📂 Results saved in: {os.path.abspath(results_path)}")
        print(f"   📊 training_progress.png - Training metrics visualization")
        print(f"   🎯 learned_policy_goal_*.png - Policy visualizations for different goals")
        print(f"   📄 training_summary.txt - Detailed training statistics")
        if include_gif:
            print(f"   🎬 agent_navigation_moving_goal.gif - Agent navigation animation")
        print("=" * 70)
        
        return results_path
    
    def _save_training_summary(self, folder_path):
        """Save detailed training summary to text file."""
        summary_path = os.path.join(folder_path, 'training_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DQN MOVING GOAL TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Environment
            f.write("ENVIRONMENT CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Grid Size: {self.env.nrow}x{self.env.ncol}\n")
            f.write(f"Total States: {self.env.nrow * self.env.ncol}\n")
            f.write(f"Start State: {self.env.start_state}\n")
            f.write(f"Goal Mode: {'Random' if self.env.random_goal else 'Cycling'}\n")
            f.write(f"Possible Goal Positions: {len(self.env.goal_positions)}\n")
            if len(self.env.goal_positions) <= 20:
                f.write(f"Goal Positions: {self.env.goal_positions}\n")
            f.write(f"Holes: {self.env.holes}\n")
            f.write(f"Number of Holes: {len(self.env.holes)}\n\n")
            
            # Network Architecture
            f.write("NETWORK ARCHITECTURE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Algorithm: Deep Q-Network (DQN) for Moving Goal\n")
            f.write(f"Input Size: {self.input_size} (agent pos + goal pos, both one-hot)\n")
            f.write(f"Hidden Layers: {[len(layer['weight'][0]) for layer in self.q_network.layers[:-1]]}\n")
            f.write(f"Output Size: {self.output_size} (Q-values for each action)\n")
            f.write(f"Activation: ReLU (hidden), Linear (output)\n\n")
            
            # Hyperparameters
            f.write("HYPERPARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Learning Rate (α): {self.alpha}\n")
            f.write(f"Discount Factor (γ): {self.gamma}\n")
            f.write(f"Initial Epsilon (ε): 1.0\n")
            f.write(f"Epsilon Decay: {self.epsilon_decay}\n")
            f.write(f"Min Epsilon: {self.epsilon_min}\n")
            f.write(f"Final Epsilon: {self.epsilon:.4f}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Memory Size: {self.memory_size}\n")
            f.write(f"Target Update Frequency: {self.target_update_freq} steps\n\n")
            
            # Training Statistics
            f.write("TRAINING STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
            f.write(f"Total Steps: {self.steps_done}\n")
            f.write(f"Average Steps per Episode: {np.mean(self.episode_steps):.2f}\n")
            f.write(f"Replay Memory Size: {len(self.memory)}\n")
            f.write(f"Unique Goals Encountered: {len(set(self.episode_goals))}\n\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            
            # Overall
            avg_reward = np.mean(self.episode_rewards)
            success_rate = np.mean([r > 0 for r in self.episode_rewards]) * 100
            f.write(f"Overall Average Reward: {avg_reward:.4f}\n")
            f.write(f"Overall Success Rate: {success_rate:.2f}%\n\n")
            
            # First 100 vs Last 100
            if len(self.episode_rewards) >= 100:
                first_100_reward = np.mean(self.episode_rewards[:100])
                first_100_success = np.mean([r > 0 for r in self.episode_rewards[:100]]) * 100
                last_100_reward = np.mean(self.episode_rewards[-100:])
                last_100_success = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
                
                f.write(f"First 100 Episodes:\n")
                f.write(f"  Average Reward: {first_100_reward:.4f}\n")
                f.write(f"  Success Rate: {first_100_success:.2f}%\n\n")
                
                f.write(f"Last 100 Episodes:\n")
                f.write(f"  Average Reward: {last_100_reward:.4f}\n")
                f.write(f"  Success Rate: {last_100_success:.2f}%\n\n")
                
                f.write(f"Improvement: {last_100_success - first_100_success:+.2f}% success rate\n\n")
            
            # Goal coverage
            goal_counts = {}
            for goal in self.episode_goals:
                goal_counts[goal] = goal_counts.get(goal, 0) + 1
            
            f.write(f"Goal Coverage:\n")
            f.write(f"  Unique goals encountered: {len(goal_counts)}\n")
            f.write(f"  Total possible goals: {len(self.env.goal_positions)}\n")
            f.write(f"  Coverage: {len(goal_counts) / len(self.env.goal_positions) * 100:.1f}%\n\n")
            
            if len(goal_counts) <= 20:
                f.write(f"Per-Goal Episode Count:\n")
                for goal, count in sorted(goal_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {goal}: {count} episodes\n")
                f.write("\n")
            
            # Loss Statistics
            if len(self.loss_history) > 0:
                f.write(f"Average Training Loss: {np.mean(self.loss_history):.6f}\n")
                f.write(f"Final Loss (last 100): {np.mean(self.loss_history[-100:]):.6f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("KEY DIFFERENCES FROM FIXED GOAL DQN:\n")
            f.write("-" * 80 + "\n")
            f.write("• State encoding includes both agent position AND goal position\n")
            f.write("• Input size is 2x larger (agent one-hot + goal one-hot)\n")
            f.write("• Agent learns general navigation policy, not path to fixed goal\n")
            f.write("• Goal changes each episode for diverse training\n")
            f.write("• Can generalize to unseen goal positions\n")
            f.write("=" * 80 + "\n")
            
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def _generate_navigation_gif(self, save_path, num_episodes=5, fps=2):
        """Generate an animated GIF showing the agent navigating with different goals."""
        # Collect episodes with different goals
        episodes_data = []
        goals_used = set()
        
        while len(episodes_data) < num_episodes:
            trajectory = []
            state, info = self.env.reset()
            goal = info['goal']
            
            # Try to get diverse goals
            if len(goals_used) < len(self.env.goal_positions) and goal in goals_used:
                continue
            
            goals_used.add(goal)
            final_reward = 0
            
            for step in range(100):
                action = self.choose_action(state, goal, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                final_reward = reward
                
                trajectory.append({
                    'state': state,
                    'goal': goal,
                    'action': int(action),
                    'done': terminated or truncated,
                    'reward': reward
                })
                
                state = next_state
                
                if terminated or truncated:
                    break
            
            episodes_data.append(trajectory)
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 10))
        
        total_frames = sum(len(ep) for ep in episodes_data) + len(episodes_data) * 3
        
        def get_frame_data(frame_num):
            frame_count = 0
            for ep_idx, episode in enumerate(episodes_data):
                ep_frames = len(episode) + 3
                if frame_num < frame_count + ep_frames:
                    step_idx = frame_num - frame_count
                    if step_idx >= len(episode):
                        step_idx = len(episode) - 1
                    return ep_idx, step_idx, (frame_num >= frame_count + len(episode))
                frame_count += ep_frames
            return len(episodes_data) - 1, len(episodes_data[-1]) - 1, True
        
        def animate(frame):
            ax.clear()
            
            ep_idx, step_idx, is_pause = get_frame_data(frame)
            trajectory = episodes_data[ep_idx]
            
            if step_idx >= len(trajectory):
                step_idx = len(trajectory) - 1
            
            step_data = trajectory[step_idx]
            state = step_data['state']
            goal = step_data['goal']
            action = step_data['action'] if not is_pause else None
            
            # Draw grid
            for i in range(self.env.nrow + 1):
                ax.axhline(y=i - 0.5, color='black', linewidth=2.5)
            for j in range(self.env.ncol + 1):
                ax.axvline(x=j - 0.5, color='black', linewidth=2.5)
            
            # Draw cells
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    pos = (i, j)
                    
                    if pos == self.env.start_state:
                        color, label, text_color = 'lightgreen', 'START', 'darkgreen'
                    elif pos == goal:
                        color, label, text_color = 'gold', 'GOAL ★', 'darkred'
                    elif pos in self.env.holes:
                        color, label, text_color = 'red', 'HOLE', 'white'
                    else:
                        color, label, text_color = 'lightblue', '', 'black'
                    
                    alpha = 0.9 if pos == goal else 0.6
                    linewidth = 2 if pos == goal else 1
                    
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=color, alpha=alpha,
                                           edgecolor='gold' if pos == goal else 'none',
                                           linewidth=linewidth, zorder=1)
                    ax.add_patch(rect)
                    
                    if label:
                        fontsize = 11 if pos == goal else 9
                        ax.text(j, i + 0.35, label, ha='center', va='center',
                               fontsize=fontsize, fontweight='bold',
                               color=text_color, alpha=0.8, zorder=2)
            
            # Draw agent
            row, col = state
            circle = patches.Circle((col, row), 0.35, facecolor='blue',
                                   edgecolor='darkblue', linewidth=3, zorder=10)
            ax.add_patch(circle)
            ax.text(col, row, '🤖', ha='center', va='center',
                   fontsize=26, zorder=11)
            
            # Draw action arrow
            if action is not None:
                arrows = {0: -0.6, 1: 0.6, 2: 0.6, 3: -0.6}
                if action in [0, 2]:  # Left, Right
                    dx, dy = arrows[action], 0
                else:  # Up, Down
                    dx, dy = 0, arrows[action]
                
                ax.annotate('', xy=(col + dx * 0.7, row + dy * 0.7),
                           xytext=(col, row),
                           arrowprops=dict(arrowstyle='->', lw=4, color='navy'),
                           zorder=9)
            
            # Configure axes
            ax.set_xlim(-0.5, self.env.ncol - 0.5)
            ax.set_ylim(-0.5, self.env.nrow - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks(range(self.env.ncol))
            ax.set_yticks(range(self.env.nrow))
            
            # Title with goal information
            status_text = ""
            if step_idx == len(trajectory) - 1:
                if step_data['reward'] > 0:
                    status_text = "\n✅ Goal Reached!"
                elif step_data['done']:
                    status_text = "\n❌ Fell in Hole"
            
            distance = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
            ax.set_title(f'DQN Agent - Moving Goal Navigation\n'
                        f'Episode {ep_idx + 1}/{num_episodes} | Step {step_idx + 1}/{len(trajectory)}\n'
                        f'Current Goal: {goal} | Distance: {distance}'
                        f'{status_text}',
                        fontsize=12, fontweight='bold', pad=15)
            
            return []
        
        # Create and save animation
        anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                      interval=1000//fps, blit=True, repeat=True)
        anim.save(save_path, writer='pillow', fps=fps, dpi=100)
        plt.close()
        
        print(f"   ✅ GIF saved: {os.path.basename(save_path)}")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("🤖 DQN AGENT FOR FROZENLAKE WITH MOVING GOAL")
    print("=" * 80)
    print("Deep Q-Network adapted to handle dynamically changing goal positions")
    print("\n✨ Key Features:")
    print("  • State encoding includes both agent position AND goal position")
    print("  • Learns general navigation policy to reach any goal")
    print("  • Goal changes each episode for diverse training")
    print("  • Can generalize to unseen goal positions")
    print("=" * 80)
    
    # Environment selection
    print("\n🎯 Select Environment:")
    print("1. Default 5x5 FrozenLake with Moving Goal")
    print("2. Small 3x3 (Quick Training)")
    print("3. Large 6x6 (Challenge)")
    print("4. Custom Configuration")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        agent = DQNAgentMovingGoal()
    elif choice == "2":
        env_params = {
            'nrow': 3, 'ncol': 3,
            'holes': [(1, 1)],
            'start_state': (0, 0),
            'random_goal': True
        }
        agent = DQNAgentMovingGoal(env_params=env_params)
    elif choice == "3":
        env_params = {
            'nrow': 6, 'ncol': 6,
            'holes': [(1, 1), (2, 3), (3, 2), (4, 4)],
            'start_state': (0, 0),
            'random_goal': True
        }
        agent = DQNAgentMovingGoal(env_params=env_params, memory_size=50000)
    else:
        print("\n📝 Custom Configuration:")
        nrow = int(input("  Number of rows: "))
        ncol = int(input("  Number of columns: "))
        num_holes = int(input("  Number of holes: "))
        holes = []
        for i in range(num_holes):
            hole_input = input(f"  Hole {i+1} position (row,col): ")
            r, c = map(int, hole_input.split(','))
            holes.append((r, c))
        
        env_params = {
            'nrow': nrow, 'ncol': ncol,
            'holes': holes,
            'start_state': (0, 0),
            'random_goal': True
        }
        agent = DQNAgentMovingGoal(env_params=env_params)
    
    # Training configuration
    print(f"\n⚙️ Training Configuration:")
    num_episodes = input(f"Number of episodes [default: 1000]: ").strip()
    num_episodes = int(num_episodes) if num_episodes else 1000
    
    # Train agent
    print("\n🚀 Starting training with moving goal...")
    agent.train(num_episodes=num_episodes, verbose=True, save_freq=100)
    
    # Evaluate
    agent.evaluate(num_episodes=100, render=False)
    
    # Generate results
    agent.generate_results_folder()
    
    print("\n🎉 Training and evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
