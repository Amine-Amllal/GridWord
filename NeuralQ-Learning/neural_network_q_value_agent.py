import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import os
from datetime import datetime
from frozenlake_env import make_frozen_lake


class NeuralNetwork:
    """
    Simple feedforward neural network for Q-value approximation.
    Uses numpy for implementation without external deep learning libraries.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        """
        Initialize neural network with specified architecture.
        
        Args:
            input_size (int): Number of input features (state encoding)
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Number of outputs (Q-values for each action)
            learning_rate (float): Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = []
        
        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for better gradient flow
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append({'weight': weight, 'bias': bias})
        
        # For storing activations during forward pass (needed for backprop)
        self.activations = []
        self.z_values = []
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.array): Input features
            
        Returns:
            np.array: Q-values for each action
        """
        self.activations = [x]
        self.z_values = []
        
        activation = x
        
        # Pass through all layers
        for i, layer in enumerate(self.layers):
            z = np.dot(activation, layer['weight']) + layer['bias']
            self.z_values.append(z)
            
            # Use ReLU for hidden layers, linear for output layer
            if i < len(self.layers) - 1:
                activation = self.relu(z)
            else:
                activation = z  # Linear activation for output
            
            self.activations.append(activation)
        
        return activation
    
    def backward(self, target, output):
        """
        Backward pass (backpropagation) to compute gradients.
        
        Args:
            target (np.array): Target Q-values
            output (np.array): Predicted Q-values
        """
        # Compute gradients
        gradients = []
        
        # Output layer gradient (MSE loss derivative)
        delta = (output - target)
        
        # Backpropagate through layers
        for i in range(len(self.layers) - 1, -1, -1):
            # Compute weight and bias gradients
            weight_gradient = np.dot(self.activations[i].T, delta)
            bias_gradient = np.sum(delta, axis=0, keepdims=True)
            
            gradients.insert(0, {'weight': weight_gradient, 'bias': bias_gradient})
            
            # Propagate gradient to previous layer
            if i > 0:
                delta = np.dot(delta, self.layers[i]['weight'].T)
                delta = delta * self.relu_derivative(self.z_values[i - 1])
        
        # Update weights and biases using gradients
        for i, layer in enumerate(self.layers):
            layer['weight'] -= self.learning_rate * gradients[i]['weight']
            layer['bias'] -= self.learning_rate * gradients[i]['bias']
    
    def train_step(self, x, target):
        """
        Single training step: forward pass, compute loss, backward pass.
        
        Args:
            x (np.array): Input features
            target (np.array): Target Q-values
            
        Returns:
            float: Loss value
        """
        output = self.forward(x)
        loss = np.mean((output - target) ** 2)
        self.backward(target, output)
        return loss


class NeuralNetworkQValueAgent:
    """
    Deep Q-Learning agent using neural network for Q-value approximation.
    Unlike Q-table approaches, this agent can generalize across states.
    """
    
    def __init__(self, alpha=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, hidden_sizes=[64, 64], env_params=None, 
                 interactive_env=False, batch_size=32, memory_size=10000):
        """
        Initialize the Neural Network Q-Learning agent.
        
        Args:
            alpha (float): Learning rate for neural network
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum epsilon value
            hidden_sizes (list): List of hidden layer sizes
            env_params (dict): Environment parameters
            interactive_env (bool): Create environment interactively
            batch_size (int): Size of minibatch for training
            memory_size (int): Size of experience replay memory
        """
        if interactive_env:
            self.env = make_frozen_lake(interactive=True)
        elif env_params:
            self.env = make_frozen_lake(**env_params)
        else:
            self.env = make_frozen_lake()
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.action_mapping = {0: "LEFT ←", 1: "DOWN ↓", 2: "RIGHT →", 3: "UP ↑"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        # State encoding: one-hot encoding of position
        self.input_size = self.env.nrow * self.env.ncol
        self.output_size = 4  # 4 actions
        
        # Initialize neural network
        self.network = NeuralNetwork(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=self.output_size,
            learning_rate=alpha
        )
        
        # Experience replay memory
        self.memory = []
        self.memory_size = memory_size
        
        # For tracking and visualization
        self.reward_history = []
        self.epsilon_history = []
        self.episode_rewards = []
        self.episode_steps = []
        self.loss_history = []
        
        print(f"🧠 Neural Network Q-Learning Agent initialized")
        print(f"   Network architecture: {self.input_size} → {' → '.join(map(str, hidden_sizes))} → {self.output_size}")
        print(f"   Learning rate: {alpha}")
        print(f"   Memory size: {memory_size}")
        print(f"   Batch size: {batch_size}")
    
    def encode_state(self, state):
        """
        Encode state as one-hot vector.
        
        Args:
            state (tuple): (row, col) position
            
        Returns:
            np.array: One-hot encoded state
        """
        row, col = state
        state_index = row * self.env.ncol + col
        encoded = np.zeros((1, self.input_size))
        encoded[0, state_index] = 1.0
        return encoded
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions from neural network.
        
        Args:
            state (tuple): Current state
            
        Returns:
            np.array: Q-values for each action
        """
        encoded_state = self.encode_state(state)
        q_values = self.network.forward(encoded_state)
        return q_values[0]
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (tuple): Current state
            
        Returns:
            int: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)
    
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
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append(experience)
    
    def sample_batch(self):
        """
        Sample random batch from memory for training.
        
        Returns:
            list: Batch of experiences
        """
        if len(self.memory) < self.batch_size:
            return self.memory
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def train_on_batch(self):
        """
        Train network on a batch of experiences.
        
        Returns:
            float: Average loss for the batch
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = self.sample_batch()
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            # Encode states
            encoded_state = self.encode_state(state)
            encoded_next_state = self.encode_state(next_state)
            
            # Get current Q-values
            current_q_values = self.network.forward(encoded_state)
            
            # Compute target Q-values
            target_q_values = current_q_values.copy()
            
            if done:
                target_q_values[0, action] = reward
            else:
                next_q_values = self.network.forward(encoded_next_state)
                target_q_values[0, action] = reward + self.gamma * np.max(next_q_values)
            
            # Train on this example
            loss = self.network.train_step(encoded_state, target_q_values)
            total_loss += loss
        
        return total_loss / len(batch)
    
    def train(self, num_episodes=1000, max_steps_per_episode=100, 
              train_frequency=1, verbose=True, save_results=True):
        """
        Train the agent using Deep Q-Learning with experience replay.
        
        Args:
            num_episodes (int): Number of training episodes
            max_steps_per_episode (int): Maximum steps per episode
            train_frequency (int): Train network every N steps
            verbose (bool): Print training progress
            save_results (bool): Save training results and visualizations
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 Starting Deep Q-Learning Training")
            print(f"{'='*70}")
            print(f"Episodes: {num_episodes}")
            print(f"Max steps per episode: {max_steps_per_episode}")
            print(f"Train frequency: every {train_frequency} step(s)")
            print(f"{'='*70}\n")
        
        start_time = time.time()
        step_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps_per_episode:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                done = terminated or truncated
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                steps += 1
                step_count += 1
                
                # Train network periodically
                if step_count % train_frequency == 0:
                    loss = self.train_on_batch()
                    self.loss_history.append(loss)
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Print progress
            if verbose and (episode + 1) % (num_episodes // 10) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                avg_steps = np.mean(self.episode_steps[-100:]) if len(self.episode_steps) >= 100 else np.mean(self.episode_steps)
                success_rate = np.sum([r > 0 for r in self.episode_rewards[-100:]]) / min(100, len(self.episode_rewards))
                
                print(f"Episode {episode + 1:5d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:6.3f} | "
                      f"Avg Steps: {avg_steps:6.1f} | "
                      f"Success: {success_rate*100:5.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ Training Complete!")
            print(f"{'='*70}")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Average time per episode: {training_time/num_episodes:.3f} seconds")
            
            # Final statistics
            final_avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            final_success_rate = np.sum([r > 0 for r in self.episode_rewards[-100:]]) / min(100, len(self.episode_rewards))
            
            print(f"\n📊 Final Performance (last 100 episodes):")
            print(f"   Average reward: {final_avg_reward:.3f}")
            print(f"   Success rate: {final_success_rate*100:.1f}%")
            print(f"   Memory size: {len(self.memory)}")
            print(f"{'='*70}\n")
        
        if save_results:
            self.save_training_results()
            self.visualize_training()
    
    def test_agent(self, num_episodes=100, max_steps=100, render=False, verbose=True):
        """
        Test the trained agent.
        
        Args:
            num_episodes (int): Number of test episodes
            max_steps (int): Maximum steps per episode
            render (bool): Whether to render episodes
            verbose (bool): Print test results
            
        Returns:
            dict: Test statistics
        """
        test_rewards = []
        test_steps = []
        successes = 0
        
        # Save current epsilon and set to 0 for testing (no exploration)
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                if render:
                    self.env.render()
                    time.sleep(0.1)
                
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(int(action))
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            test_rewards.append(episode_reward)
            test_steps.append(steps)
            if episode_reward > 0:
                successes += 1
        
        # Restore epsilon
        self.epsilon = original_epsilon
        
        # Calculate statistics
        stats = {
            'avg_reward': np.mean(test_rewards),
            'avg_steps': np.mean(test_steps),
            'success_rate': successes / num_episodes,
            'total_successes': successes
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🧪 Test Results ({num_episodes} episodes)")
            print(f"{'='*70}")
            print(f"Average reward: {stats['avg_reward']:.3f}")
            print(f"Average steps: {stats['avg_steps']:.1f}")
            print(f"Success rate: {stats['success_rate']*100:.1f}% ({stats['total_successes']}/{num_episodes})")
            print(f"{'='*70}\n")
        
        return stats
    
    def save_training_results(self):
        """Save training results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"neural_q_learning_results_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        
        # Save training summary
        with open(os.path.join(folder_name, "training_summary.txt"), "w", encoding='utf-8') as f:
            f.write(f"Neural Network Q-Learning Training Results\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Environment: {self.env.nrow}x{self.env.ncol} FrozenLake\n")
            f.write(f"Network Architecture: {self.input_size} → {' → '.join(map(str, [64, 64]))} → {self.output_size}\n")
            f.write(f"Learning Rate: {self.alpha}\n")
            f.write(f"Discount Factor: {self.gamma}\n")
            f.write(f"Epsilon Decay: {self.epsilon_decay}\n")
            f.write(f"Memory Size: {self.memory_size}\n")
            f.write(f"Batch Size: {self.batch_size}\n\n")
            
            f.write(f"Training Statistics:\n")
            f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
            f.write(f"Final Average Reward: {np.mean(self.episode_rewards[-100:]):.3f}\n")
            f.write(f"Final Success Rate: {np.sum([r > 0 for r in self.episode_rewards[-100:]]) / 100 * 100:.1f}%\n")
        
        print(f"✅ Training results saved to {folder_name}/")
    
    def visualize_training(self):
        """Create visualization of training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) > 100:
            smoothed = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(self.episode_rewards)), smoothed, 
                          label='100-Episode Average', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Steps
        axes[0, 1].plot(self.episode_steps, alpha=0.3, label='Episode Steps')
        if len(self.episode_steps) > 100:
            smoothed = np.convolve(self.episode_steps, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(self.episode_steps)), smoothed,
                          label='100-Episode Average', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Steps per Episode')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Epsilon Decay
        axes[1, 0].plot(self.epsilon_history)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate (Epsilon) Decay')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss History
        if self.loss_history:
            axes[1, 1].plot(self.loss_history, alpha=0.5)
            if len(self.loss_history) > 100:
                smoothed = np.convolve(self.loss_history, np.ones(100)/100, mode='valid')
                axes[1, 1].plot(range(99, len(self.loss_history)), smoothed,
                              label='100-Step Average', linewidth=2, color='red')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss (MSE)')
            axes[1, 1].set_title('Training Loss Over Time')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'neural_q_learning_training_progress.png', dpi=300, bbox_inches='tight')
        print(f"📊 Training visualization saved to neural_q_learning_training_progress.png")
        plt.show()
    
    def visualize_policy(self):
        """Visualize the learned policy on the grid."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for i in range(self.env.nrow + 1):
            ax.axhline(y=i, color='black', linewidth=2)
        for j in range(self.env.ncol + 1):
            ax.axvline(x=j, color='black', linewidth=2)
        
        # Color cells and show policy
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                state = (i, j)
                cell_type = self.env.desc[i, j]
                
                # Color based on cell type
                if cell_type == 'S':
                    color = 'lightgreen'
                elif cell_type == 'F':
                    color = 'lightblue'
                elif cell_type == 'H':
                    color = 'red'
                elif cell_type == 'G':
                    color = 'gold'
                else:
                    color = 'white'
                
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0, 
                                        edgecolor='none', facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Show policy arrow if not terminal
                if state not in self.env.terminal_states:
                    q_values = self.get_q_values(state)
                    best_action = int(np.argmax(q_values))
                    arrow = self.action_symbols[best_action]
                    ax.text(j + 0.5, i + 0.5, arrow, ha='center', va='center',
                           fontsize=20, fontweight='bold', color='darkblue')
                
                # Show cell type label
                if cell_type in ['S', 'G', 'H']:
                    label = {'S': 'START', 'G': 'GOAL', 'H': 'HOLE'}[cell_type]
                    ax.text(j + 0.5, i + 0.15, label, ha='center', va='center',
                           fontsize=8, fontweight='bold')
        
        ax.set_xlim(0, self.env.ncol)
        ax.set_ylim(0, self.env.nrow)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.env.ncol + 1))
        ax.set_yticks(range(self.env.nrow + 1))
        ax.set_title('Learned Policy (Neural Network Q-Learning)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('neural_q_learning_policy.png', dpi=300, bbox_inches='tight')
        print(f"🗺️ Policy visualization saved to neural_q_learning_policy.png")
        plt.show()


def main():
    """Main function to demonstrate the Neural Network Q-Learning agent."""
    print("\n" + "="*70)
    print("🧠 NEURAL NETWORK Q-LEARNING AGENT FOR FROZENLAKE")
    print("="*70)
    
    # Create environment
    env_params = {
        'nrow': 5,
        'ncol': 5,
        'holes': [(1, 1), (2, 3), (3, 2)],
        'goal': (4, 4),
        'start_state': (0, 0)
    }
    
    # Initialize agent
    agent = NeuralNetworkQValueAgent(
        alpha=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_sizes=[64, 64],
        env_params=env_params,
        batch_size=32,
        memory_size=10000
    )
    
    # Train agent
    agent.train(
        num_episodes=2000,
        max_steps_per_episode=100,
        train_frequency=1,
        verbose=True,
        save_results=True
    )
    
    # Test agent
    agent.test_agent(num_episodes=100, verbose=True)
    
    # Visualize results
    agent.visualize_policy()
    
    print("\n" + "="*70)
    print("✅ TRAINING AND EVALUATION COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
