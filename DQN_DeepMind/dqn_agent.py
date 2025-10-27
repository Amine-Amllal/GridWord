import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import os
import sys
from datetime import datetime
from collections import deque
import random

# Add parent directory to path to import frozenlake_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frozenlake_env import make_frozen_lake


class DeepMindDQN:
    """
    Deep Q-Network following DeepMind's architecture.
    Implements a neural network with configurable layers for Q-value approximation.
    """
    
    def __init__(self, input_size, output_size, hidden_layers=[256, 256], learning_rate=0.00025):
        """
        Initialize DeepMind DQN architecture.
        
        Args:
            input_size (int): Number of input features (state encoding)
            output_size (int): Number of outputs (Q-values for each action)
            hidden_layers (list): List of hidden layer sizes (DeepMind default: [256, 256])
            learning_rate (float): Learning rate (DeepMind default: 0.00025)
        """
        self.learning_rate = learning_rate
        self.layers = []
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights using Xavier/He initialization with smaller initial values
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU activation with reduced scale for stability
            std = np.sqrt(2.0 / layer_sizes[i]) * 0.5  # Reduced scale
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append({'weight': weight, 'bias': bias})
        
        # For storing activations during forward pass
        self.activations = []
        self.z_values = []
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU for backpropagation"""
        return (x > 0).astype(float)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.array): Input features (batch_size, input_size)
            
        Returns:
            np.array: Q-values for each action (batch_size, output_size)
        """
        self.activations = [x]
        self.z_values = []
        
        activation = x
        
        # Pass through all layers
        for i, layer in enumerate(self.layers):
            z = np.dot(activation, layer['weight']) + layer['bias']
            self.z_values.append(z)
            
            # ReLU for hidden layers, linear for output layer
            if i < len(self.layers) - 1:
                activation = self.relu(z)
            else:
                activation = z  # Linear activation for Q-values
            
            self.activations.append(activation)
        
        return activation
    
    def backward(self, target, output):
        """
        Backward pass (backpropagation) with gradient descent and gradient clipping.
        
        Args:
            target (np.array): Target Q-values
            output (np.array): Predicted Q-values
        """
        batch_size = output.shape[0]
        gradients = []
        
        # Output layer gradient (MSE loss derivative)
        delta = (output - target) / batch_size
        
        # Backpropagate through layers
        for i in range(len(self.layers) - 1, -1, -1):
            # Compute gradients
            weight_gradient = np.dot(self.activations[i].T, delta)
            bias_gradient = np.sum(delta, axis=0, keepdims=True)
            
            # Gradient clipping for stability
            weight_gradient = np.clip(weight_gradient, -1.0, 1.0)
            bias_gradient = np.clip(bias_gradient, -1.0, 1.0)
            
            gradients.insert(0, {'weight': weight_gradient, 'bias': bias_gradient})
            
            # Propagate to previous layer
            if i > 0:
                delta = np.dot(delta, self.layers[i]['weight'].T)
                delta = delta * self.relu_derivative(self.z_values[i - 1])
        
        # Update weights with gradient descent
        for i, layer in enumerate(self.layers):
            layer['weight'] -= self.learning_rate * gradients[i]['weight']
            layer['bias'] -= self.learning_rate * gradients[i]['bias']
    
    def train_step(self, x, target):
        """
        Single training step: forward + backward pass.
        
        Args:
            x (np.array): Input features
            target (np.array): Target Q-values
            
        Returns:
            float: MSE loss
        """
        output = self.forward(x)
        loss = np.mean((output - target) ** 2)
        self.backward(target, output)
        return loss
    
    def copy_weights_from(self, other_network):
        """
        Copy weights from another network (for target network updates).
        
        Args:
            other_network (DeepMindDQN): Network to copy weights from
        """
        for i in range(len(self.layers)):
            self.layers[i]['weight'] = other_network.layers[i]['weight'].copy()
            self.layers[i]['bias'] = other_network.layers[i]['bias'].copy()


class DQNAgent:
    """
    Deep Q-Network Agent implementing DeepMind's DQN algorithm.
    
    Key features:
    - Experience Replay: Store and sample from past experiences
    - Target Network: Separate network for stable Q-value targets
    - Epsilon-greedy exploration
    - Batch training for efficient learning
    """
    
    def __init__(self, alpha=0.00025, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, hidden_layers=[256, 256], env_params=None, 
                 interactive_env=False, batch_size=32, memory_size=10000,
                 target_update_freq=1000):
        """
        Initialize DQN Agent.
        
        Args:
            alpha (float): Learning rate (DeepMind: 0.00025)
            gamma (float): Discount factor (DeepMind: 0.99)
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum epsilon
            hidden_layers (list): Network architecture (DeepMind: [256, 256])
            env_params (dict): Environment parameters
            interactive_env (bool): Create interactive environment
            batch_size (int): Minibatch size for training (DeepMind: 32)
            memory_size (int): Replay memory size (DeepMind: 1,000,000)
            target_update_freq (int): Steps between target network updates
        """
        # Create environment
        if interactive_env:
            self.env = make_frozen_lake(interactive=True)
        elif env_params:
            self.env = make_frozen_lake(**env_params)
        else:
            self.env = make_frozen_lake()
        
        # Adapt hyperparameters based on environment size
        env_size = self.env.nrow * self.env.ncol
        
        # Use smaller network for smaller environments
        if env_size <= 16 and hidden_layers == [256, 256]:
            hidden_layers = [128, 64]
            print(f"📐 Auto-adjusted network size to {hidden_layers} for {self.env.nrow}x{self.env.ncol} environment")
        elif env_size <= 36 and hidden_layers == [256, 256]:
            hidden_layers = [128, 128]
            print(f"📐 Auto-adjusted network size to {hidden_layers} for {self.env.nrow}x{self.env.ncol} environment")
        
        # Increase learning rate for smaller environments
        if env_size <= 16 and alpha == 0.00025:
            alpha = 0.001
            print(f"⚡ Auto-adjusted learning rate to {alpha} for faster convergence")
        elif env_size <= 36 and alpha == 0.00025:
            alpha = 0.0005
            print(f"⚡ Auto-adjusted learning rate to {alpha} for faster convergence")
        
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
        
        # State encoding: one-hot encoding
        self.input_size = self.env.nrow * self.env.ncol
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
        
        # Experience Replay Memory: (state, action, reward, next_state, done)
        self.memory = deque(maxlen=memory_size)
        
        # Training tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.loss_history = []
        self.epsilon_history = []
        self.steps_done = 0
        
        print("🤖 DeepMind DQN Agent Initialized")
        print("=" * 70)
        print(f"Environment: {self.env.nrow}x{self.env.ncol} FrozenLake")
        print(f"Network: {self.input_size} → {' → '.join(map(str, hidden_layers))} → {self.output_size}")
        print(f"Learning Rate: {alpha}")
        print(f"Discount Factor (γ): {gamma}")
        print(f"Batch Size: {batch_size}")
        print(f"Memory Size: {memory_size}")
        print(f"Target Update Frequency: {target_update_freq} steps")
        print("=" * 70)
    
    def encode_state(self, state):
        """
        Encode state as one-hot vector.
        
        Args:
            state (tuple): (row, col) position
            
        Returns:
            np.array: One-hot encoded state
        """
        row, col = state
        state_idx = row * self.env.ncol + col
        encoded = np.zeros((1, self.input_size))
        encoded[0, state_idx] = 1
        return encoded
    
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state (tuple): Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice([0, 1, 2, 3])
        else:
            # Exploitation: greedy action from Q-network
            encoded_state = self.encode_state(state)
            q_values = self.q_network.forward(encoded_state)
            return np.argmax(q_values[0])
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_batch(self):
        """
        Sample random batch from replay memory.
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = np.vstack([self.encode_state(exp[0]) for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.vstack([self.encode_state(exp[3]) for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def train_on_batch(self):
        """
        Train the Q-network on a batch from replay memory.
        
        Returns:
            float: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.sample_batch()
        
        # Compute target Q-values using target network
        current_q_values = self.q_network.forward(states)
        next_q_values = self.target_network.forward(next_states)
        
        # Compute targets
        target_q_values = current_q_values.copy()
        
        for i in range(len(states)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train network
        loss = self.q_network.train_step(states, target_q_values)
        
        return loss
    
    def update_target_network(self):
        """Update target network with weights from Q-network."""
        self.target_network.copy_weights_from(self.q_network)
    
    def train(self, num_episodes=1000, verbose=True, save_freq=100):
        """
        Train the DQN agent with improved convergence.
        
        Args:
            num_episodes (int): Number of training episodes
            verbose (bool): Print training progress
            save_freq (int): Frequency of progress updates
        """
        print("\n🎓 Starting DQN Training...")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Max Steps per Episode: 200")
        print(f"Warmup Episodes: {max(100, self.batch_size * 2)}")
        print("=" * 70)
        
        # Warmup phase: fill replay buffer with random exploration
        warmup_episodes = max(100, self.batch_size * 2)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
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
                action = self.choose_action(state, training=True)
                self.epsilon = original_epsilon
                
                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                
                # Reward shaping for better learning
                shaped_reward = reward
                if reward > 0:  # Reached goal
                    shaped_reward = 10.0  # Increased reward for success
                elif terminated:  # Fell in hole
                    shaped_reward = -5.0  # Penalty for failure
                else:  # Normal step
                    # Add small penalty for each step to encourage shorter paths
                    shaped_reward = -0.01
                    # Add distance-based reward shaping
                    goal_row, goal_col = self.env.goal
                    curr_row, curr_col = state
                    next_row, next_col = next_state
                    curr_dist = abs(curr_row - goal_row) + abs(curr_col - goal_col)
                    next_dist = abs(next_row - goal_row) + abs(next_col - goal_col)
                    if next_dist < curr_dist:
                        shaped_reward += 0.1  # Reward for moving closer to goal
                
                # Store experience with shaped reward
                self.store_experience(state, int(action), shaped_reward, next_state, terminated or truncated)
                
                # Only train after warmup period
                if episode >= warmup_episodes and len(self.memory) >= self.batch_size:
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
                
                print(f"Episode {episode + 1:4d} | "
                      f"Reward: {avg_reward:.3f} | "
                      f"Steps: {avg_steps:.1f} | "
                      f"Success: {success_rate:.1f}% | "
                      f"Loss: {avg_loss:.6f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Memory: {len(self.memory)}")
        
        print("\n✅ DQN Training Complete!")
        print("=" * 70)
        
        # Final statistics
        final_avg_reward = np.mean(self.episode_rewards[-100:])
        final_success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
        
        print(f"Final Performance (last 100 episodes):")
        print(f"  Average Reward: {final_avg_reward:.3f}")
        print(f"  Success Rate: {final_success_rate:.1f}%")
        print(f"  Final Epsilon: {self.epsilon:.3f}")
        print(f"  Total Steps: {self.steps_done}")
        print(f"  Replay Memory Size: {len(self.memory)}")
    
    def evaluate(self, num_episodes=100, render=False):
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            render (bool): Whether to render episodes
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n🔬 Evaluating DQN Agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_steps = []
        eval_successes = []
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < 200:
                action = self.choose_action(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(int(action))
                total_reward += reward
                steps += 1
                
                if render and ep < 5:  # Render first 5 episodes
                    self.env.render(mode='text')
                    time.sleep(0.5)
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
            eval_successes.append(total_reward > 0)
        
        metrics = {
            'avg_reward': np.mean(eval_rewards),
            'success_rate': np.mean(eval_successes) * 100,
            'avg_steps': np.mean(eval_steps),
            'std_reward': np.std(eval_rewards),
            'std_steps': np.std(eval_steps)
        }
        
        print("=" * 70)
        print("📊 Evaluation Results:")
        print(f"  Average Reward: {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Success Rate: {metrics['success_rate']:.1f}%")
        print(f"  Average Steps: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
        print("=" * 70)
        
        return metrics
    
    def plot_training_progress(self, save_path=None):
        """
        Plot training progress.
        
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
        ax1.set_title('Training Rewards', fontweight='bold')
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
        
        # Epsilon decay
        ax4 = axes[1, 1]
        ax4.plot(range(len(self.epsilon_history)), self.epsilon_history, 'orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate (Epsilon)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_policy(self, save_path=None):
        """
        Visualize the learned policy as a grid with action arrows.
        
        Args:
            save_path (str): Path to save the visualization
        """
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
                elif cell_type == 'G':
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
        
        # Draw policy arrows
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                if (i, j) not in self.env.terminal_states:
                    state = (i, j)
                    action = self.choose_action(state, training=False)
                    symbol = self.action_symbols[int(action)]
                    ax.text(j, i, symbol, ha='center', va='center',
                           fontsize=24, fontweight='bold', color='darkblue')
        
        ax.set_xlim(-0.5, self.env.ncol - 0.5)
        ax.set_ylim(-0.5, self.env.nrow - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.env.ncol))
        ax.set_yticks(range(self.env.nrow))
        ax.set_title(f'DQN Learned Policy - {self.env.nrow}x{self.env.ncol} FrozenLake',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Policy visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_results_folder(self, folder_name=None):
        """
        Generate comprehensive results folder with plots and statistics.
        
        Args:
            folder_name (str): Name of results folder
            
        Returns:
            str: Path to results folder
        """
        if folder_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"dqn_results_{timestamp}"
        
        # Create folder in DQN_DeepMind directory
        results_path = os.path.join(os.path.dirname(__file__), folder_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        print(f"\n📁 Generating results in: {results_path}")
        print("=" * 70)
        
        # 1. Training progress plot
        print("📈 Creating training progress plot...")
        self.plot_training_progress(save_path=os.path.join(results_path, 'training_progress.png'))
        
        # 2. Policy visualization
        print("🎯 Creating policy visualization...")
        self.visualize_policy(save_path=os.path.join(results_path, 'learned_policy.png'))
        
        # 3. Training summary
        print("📝 Creating training summary...")
        self._save_training_summary(results_path)
        
        print("\n✅ Results generation complete!")
        print("=" * 70)
        print(f"📂 Results saved in: {os.path.abspath(results_path)}")
        print(f"   📊 training_progress.png - Training metrics visualization")
        print(f"   🎯 learned_policy.png - Learned policy visualization")
        print(f"   📄 training_summary.txt - Detailed training statistics")
        print("=" * 70)
        
        return results_path
    
    def _save_training_summary(self, folder_path):
        """
        Save detailed training summary to text file.
        
        Args:
            folder_path (str): Path to results folder
        """
        summary_path = os.path.join(folder_path, 'training_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEPMIND DQN TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Environment
            f.write("ENVIRONMENT CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Grid Size: {self.env.nrow}x{self.env.ncol}\n")
            f.write(f"Total States: {self.env.nrow * self.env.ncol}\n")
            f.write(f"Start State: {self.env.start_state}\n")
            f.write(f"Goal State: {self.env.goal}\n")
            f.write(f"Holes: {self.env.holes}\n")
            f.write(f"Number of Holes: {len(self.env.holes)}\n\n")
            
            # Network Architecture
            f.write("NETWORK ARCHITECTURE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Algorithm: Deep Q-Network (DQN) - DeepMind Architecture\n")
            f.write(f"Input Size: {self.input_size} (one-hot encoded state)\n")
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
            f.write(f"Replay Memory Size: {len(self.memory)}\n\n")
            
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
            
            # Best/Worst Episodes
            best_ep = np.argmax(self.episode_rewards)
            worst_ep = np.argmin(self.episode_rewards)
            f.write(f"Best Episode: {best_ep} (Reward: {self.episode_rewards[best_ep]:.4f})\n")
            f.write(f"Worst Episode: {worst_ep} (Reward: {self.episode_rewards[worst_ep]:.4f})\n\n")
            
            # Loss Statistics
            if len(self.loss_history) > 0:
                f.write(f"Average Training Loss: {np.mean(self.loss_history):.6f}\n")
                f.write(f"Final Loss (last 100): {np.mean(self.loss_history[-100:]):.6f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("🤖 IMPROVED DEEPMIND DQN AGENT FOR FROZENLAKE")
    print("=" * 80)
    print("Deep Q-Network with Experience Replay and Target Network")
    print("\n✨ Recent Improvements:")
    print("  • Gradient clipping for stability")
    print("  • Adaptive learning rate based on environment size")
    print("  • Improved weight initialization")
    print("  • Reward shaping (goal: +10, hole: -5, step: -0.01)")
    print("  • Distance-based guidance (+0.1 for moving toward goal)")
    print("  • Warmup period for replay buffer")
    print("  • Smaller networks for faster convergence on small envs")
    print("=" * 80)
    
    # Environment selection
    print("\n🎯 Select Environment:")
    print("1. Default 5x5 FrozenLake")
    print("2. Small 3x3 (Quick Training)")
    print("3. Large 8x8 (Challenge)")
    print("4. Custom Interactive Environment")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        agent = DQNAgent()
    elif choice == "2":
        env_params = {'nrow': 3, 'ncol': 3, 'holes': [(1, 1)], 'goal': (2, 2)}
        agent = DQNAgent(env_params=env_params)
    elif choice == "3":
        env_params = {'nrow': 8, 'ncol': 8, 'holes': [(2, 2), (3, 5), (4, 3), (5, 6), (6, 1)], 
                     'goal': (7, 7)}
        agent = DQNAgent(env_params=env_params, memory_size=50000)
    else:
        agent = DQNAgent(interactive_env=True)
    
    # Training configuration
    print(f"\n⚙️ Training Configuration:")
    num_episodes = input(f"Number of episodes [default: 1000]: ").strip()
    num_episodes = int(num_episodes) if num_episodes else 1000
    
    # Train agent
    print("\n🚀 Starting training...")
    agent.train(num_episodes=num_episodes, verbose=True, save_freq=100)
    
    # Evaluate
    agent.evaluate(num_episodes=100, render=False)
    
    # Generate results
    agent.generate_results_folder()
    
    print("\n🎉 Training and evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()