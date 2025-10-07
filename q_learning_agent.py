import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import os
from datetime import datetime
from frozenlake_env import make_frozen_lake

class QLearningAgent:
    """
    A Q-Learning agent with animated visualization for the FrozenLake environment.
    Implements the Q-Learning algorithm and provides visual animations of:
    1. Q-value evolution during training
    2. Policy derivation from Q-values
    3. Episode execution with learned policy
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.01, 
                 env_params=None, interactive_env=False):
        """4
        Initialize the Q-Learning agent.
        
        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum epsilon value
            env_params (dict): Environment parameters (nrow, ncol, holes, goal, start_state)
            interactive_env (bool): If True, create environment interactively with user input
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
        self.action_mapping = {0: "LEFT ←", 1: "DOWN ↓", 2: "RIGHT →", 3: "UP ↑"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        # Initialize Q-table: [row, col, action]
        self.Q = np.zeros((self.env.nrow, self.env.ncol, 4))
        
        # For animation tracking
        self.q_history = []
        self.policy_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.episode_rewards = []
        self.episode_steps = []
        
        # Adaptive parameters based on environment size
        self._adapt_hyperparameters_to_env_size()
        
    def _adapt_hyperparameters_to_env_size(self):
        """Adapt hyperparameters based on environment size for better learning"""
        env_size = self.env.nrow * self.env.ncol
        num_holes = len(self.env.holes)
        
        # Calculate complexity factor
        complexity = env_size + num_holes * 2  # Holes make it more complex
        
        # Store original values
        original_alpha = self.alpha
        original_epsilon_decay = self.epsilon_decay
        original_epsilon_min = self.epsilon_min
        
        # Adaptive parameters based on complexity
        if env_size >= 64:  # 8x8 or larger
            # Large environments need more exploration and slower learning
            if self.alpha <= 0.1:  # Only adjust if using default or low values
                self.alpha = 0.3
            if self.epsilon_decay >= 0.995:  # Only adjust if using default
                self.epsilon_decay = 0.9995  # Slower decay
            if self.epsilon_min <= 0.01:  # Only adjust if using default
                self.epsilon_min = 0.05  # Higher minimum exploration
                
        elif env_size >= 25:  # 5x5 to 7x7
            # Medium environments
            if self.alpha <= 0.1:
                self.alpha = 0.2
            if self.epsilon_decay >= 0.995:
                self.epsilon_decay = 0.998
            if self.epsilon_min <= 0.01:
                self.epsilon_min = 0.02
        
        # Store adaptive values for display
        self.adaptive_alpha = self.alpha
        self.adaptive_epsilon_decay = self.epsilon_decay
        self.adaptive_epsilon_min = self.epsilon_min
        
        if env_size >= 25 and (self.alpha != original_alpha or 
                              self.epsilon_decay != original_epsilon_decay or 
                              self.epsilon_min != original_epsilon_min):
            print(f"🧠 Adaptive hyperparameters applied for {self.env.nrow}x{self.env.ncol} environment:")
            print(f"   Learning rate: {original_alpha} → {self.alpha}")
            print(f"   Epsilon decay: {original_epsilon_decay} → {self.epsilon_decay}")
            print(f"   Min epsilon: {original_epsilon_min} → {self.epsilon_min}")
    
    def get_recommended_episodes(self):
        """Get sophisticated recommended number of training episodes based on environment complexity"""
        env_size = self.env.nrow * self.env.ncol
        num_holes = len(self.env.holes)
        
        # Calculate path complexity
        min_path_length = (self.env.nrow - 1) + (self.env.ncol - 1)
        grid_ratio = max(self.env.nrow, self.env.ncol) / min(self.env.nrow, self.env.ncol)
        
        # Base calculation factors
        base_factor = 30  # Base episodes per state
        
        # Complexity multipliers
        size_multiplier = 1.0
        hole_multiplier = 1.0
        shape_multiplier = 1.0
        
        # Adjust based on environment size (non-linear scaling)
        if env_size <= 9:  # 3x3 and smaller
            size_multiplier = 0.8  # Smaller environments need fewer episodes
            base_factor = 40
        elif env_size <= 25:  # 4x4 to 5x5
            size_multiplier = 1.0
            base_factor = 35
        elif env_size <= 49:  # 6x6 to 7x7
            size_multiplier = 1.3
            base_factor = 40
        elif env_size <= 100:  # 8x8 to 10x10
            size_multiplier = 1.8
            base_factor = 45
        else:  # Very large environments
            size_multiplier = 2.5
            base_factor = 50
        
        # Hole complexity (exponential impact)
        hole_density = num_holes / env_size if env_size > 0 else 0
        if hole_density > 0.3:  # High hole density
            hole_multiplier = 2.0
        elif hole_density > 0.2:  # Medium-high hole density
            hole_multiplier = 1.7
        elif hole_density > 0.1:  # Medium hole density
            hole_multiplier = 1.4
        elif hole_density > 0.05:  # Low-medium hole density
            hole_multiplier = 1.2
        else:  # Very low or no holes
            hole_multiplier = 1.0
        
        # Shape complexity (non-square grids are harder)
        if grid_ratio > 2.0:  # Very rectangular
            shape_multiplier = 1.4
        elif grid_ratio > 1.5:  # Moderately rectangular
            shape_multiplier = 1.2
        else:  # Square or close to square
            shape_multiplier = 1.0
        
        # Calculate recommended episodes
        base_episodes = env_size * base_factor
        complexity_factor = size_multiplier * hole_multiplier * shape_multiplier
        recommended = int(base_episodes * complexity_factor)
        
        # Add extra episodes for very long minimum paths
        if min_path_length > 15:
            path_bonus = (min_path_length - 15) * 50
            recommended += path_bonus
        
        # Set adaptive bounds based on environment size
        if env_size <= 9:
            min_episodes = 300
            max_episodes = 2000
        elif env_size <= 25:
            min_episodes = 500
            max_episodes = 4000
        elif env_size <= 64:
            min_episodes = 1500
            max_episodes = 8000
        else:
            min_episodes = 3000
            max_episodes = 20000
        
        final_episodes = max(min_episodes, min(recommended, max_episodes))
        
        # Store calculation details for debugging
        self._episode_calculation = {
            'base_episodes': base_episodes,
            'size_multiplier': size_multiplier,
            'hole_multiplier': hole_multiplier,
            'shape_multiplier': shape_multiplier,
            'complexity_factor': complexity_factor,
            'hole_density': hole_density,
            'grid_ratio': grid_ratio,
            'min_path_length': min_path_length,
            'final_episodes': final_episodes
        }
        
        return final_episodes
    
    def get_episode_calculation_details(self):
        """Get detailed explanation of how episodes were calculated"""
        if hasattr(self, '_episode_calculation'):
            calc = self._episode_calculation
            return f"""
📊 Episode Calculation Details:
   Base Episodes: {calc['base_episodes']} ({self.env.nrow * self.env.ncol} states × base factor)
   Size Multiplier: {calc['size_multiplier']:.1f}
   Hole Multiplier: {calc['hole_multiplier']:.1f} (density: {calc['hole_density']:.1%})
   Shape Multiplier: {calc['shape_multiplier']:.1f} (ratio: {calc['grid_ratio']:.1f})
   Complexity Factor: {calc['complexity_factor']:.2f}
   Min Path Length: {calc['min_path_length']} steps
   Final Episodes: {calc['final_episodes']}
"""
        return "Episode calculation details not available."
    
    def get_episode_options(self):
        """Get different episode options for various training intensities"""
        recommended = self.get_recommended_episodes()  # This triggers _episode_calculation
        
        return {
            'quick': max(300, recommended // 3),
            'standard': max(500, int(recommended * 0.6)),
            'recommended': recommended,
            'thorough': min(20000, int(recommended * 1.5)),
            'extensive': min(30000, int(recommended * 2.0))
        }
    
    def print_episode_options(self):
        """Print available episode training options"""
        options = self.get_episode_options()
        env_size = self.env.nrow * self.env.ncol
        
        print(f"\n🎯 Flexible Training Options for {self.env.nrow}x{self.env.ncol}:")
        print(f"   🚀 Quick:       {options['quick']:,} episodes (~{options['quick']//200:.0f}-{options['quick']//100:.0f} min)")
        print(f"   ⚡ Standard:    {options['standard']:,} episodes (~{options['standard']//200:.0f}-{options['standard']//100:.0f} min)")
        print(f"   🎯 Recommended: {options['recommended']:,} episodes (~{options['recommended']//200:.0f}-{options['recommended']//100:.0f} min) ⭐")
        print(f"   🔥 Thorough:    {options['thorough']:,} episodes (~{options['thorough']//200:.0f}-{options['thorough']//100:.0f} min)")
        print(f"   💎 Extensive:   {options['extensive']:,} episodes (~{options['extensive']//200:.0f}-{options['extensive']//100:.0f} min)")
        
        if env_size >= 64:
            print(f"   💡 For {self.env.nrow}x{self.env.ncol}, 'Recommended' or higher is suggested for good results")
        elif env_size >= 25:
            print(f"   💡 For {self.env.nrow}x{self.env.ncol}, 'Standard' or higher usually works well")
        else:
            print(f"   💡 For {self.env.nrow}x{self.env.ncol}, 'Quick' or 'Standard' is often sufficient")
    
    def get_recommended_max_steps(self):
        """Get recommended max steps per episode based on environment size"""
        # Minimum steps to reach goal in optimal path
        min_path = (self.env.nrow - 1) + (self.env.ncol - 1)
        # Add buffer for exploration and suboptimal paths
        buffer_multiplier = 3 if self.env.nrow * self.env.ncol <= 25 else 4
        return max(200, min_path * buffer_multiplier)
        
    def state_to_coords(self, state):
        """Convert state tuple to row, col coordinates"""
        if isinstance(state, tuple):
            return state
        else:
            return (state // self.env.ncol, state % self.env.ncol)
    
    def get_policy_from_q(self):
        """Derive policy from current Q-values (greedy policy)"""
        policy = np.zeros((self.env.nrow, self.env.ncol), dtype=int)
        for row in range(self.env.nrow):
            for col in range(self.env.ncol):
                if (row, col) not in self.env.terminal_states:
                    policy[row, col] = np.argmax(self.Q[row, col])
        return policy
    
    def get_value_function_from_q(self):
        """Derive value function from Q-values (max over actions)"""
        V = np.zeros((self.env.nrow, self.env.ncol))
        for row in range(self.env.nrow):
            for col in range(self.env.ncol):
                V[row, col] = np.max(self.Q[row, col])
        return V
    
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training (bool): Whether in training mode (uses epsilon-greedy) or evaluation mode (greedy)
        """
        row, col = self.state_to_coords(state)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(self.env.action_space)
        else:
            # Exploitation: greedy action
            return np.argmax(self.Q[row, col])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        row, col = self.state_to_coords(state)
        next_row, next_col = self.state_to_coords(next_state)
        
        # Current Q-value
        current_q = self.Q[row, col, action]
        
        # Target Q-value
        if done:
            target_q = reward  # No future rewards if episode is done
        else:
            max_next_q = np.max(self.Q[next_row, next_col])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        self.Q[row, col, action] += self.alpha * (target_q - current_q)
    
    def train(self, num_episodes=None, animate_every=50, verbose=True, 
              early_stopping=True, patience=200, min_improvement=0.01):
        """
        Train the Q-Learning agent with optional early stopping.
        
        Args:
            num_episodes (int): Number of training episodes (auto-calculated if None)
            animate_every (int): Store Q-values for animation every N episodes
            verbose (bool): Print training progress
            early_stopping (bool): Whether to use early stopping based on performance plateau
            patience (int): Number of episodes to wait for improvement before stopping
            min_improvement (float): Minimum improvement in average reward to reset patience counter
        """
        self.animate_every = animate_every
        
        # Auto-calculate episodes if not provided
        if num_episodes is None:
            num_episodes = self.get_recommended_episodes()
            
        recommended_episodes = self.get_recommended_episodes()
        max_steps_per_episode = self.get_recommended_max_steps()
        env_size = self.env.nrow * self.env.ncol
        
        print("🎓 Starting Q-Learning Training...")
        print("=" * 50)
        print(f"Environment: {self.env.nrow}x{self.env.ncol} FrozenLake ({env_size} states)")
        print(f"Episodes: {num_episodes} (Recommended: {recommended_episodes})")
        print(f"Learning Rate (α): {self.alpha}")
        print(f"Discount Factor (γ): {self.gamma}")
        print(f"Initial Epsilon (ε): {self.epsilon}")
        print(f"Epsilon Decay: {self.epsilon_decay}")
        print(f"Min Epsilon: {self.epsilon_min}")
        print(f"Max Steps/Episode: {max_steps_per_episode}")
        print(f"Holes: {len(self.env.holes)} ({len(self.env.holes)/(env_size)*100:.1f}% density), Goal: {self.env.goal}")
        
        if early_stopping:
            print(f"Early Stopping: Enabled (patience={patience}, min_improvement={min_improvement})")
        
        # Show detailed calculation if episodes differ significantly
        if abs(num_episodes - recommended_episodes) > 500 or verbose:
            print(self.get_episode_calculation_details())
        
        if num_episodes < recommended_episodes:
            print(f"⚠️  WARNING: {num_episodes} episodes might be insufficient for {self.env.nrow}x{self.env.ncol}")
            print(f"   Consider using at least {recommended_episodes} episodes for better learning")
            
        print("=" * 50)
        
        self.episode_rewards = []
        self.episode_steps = []
        self.q_history = []
        self.policy_history = []
        self.epsilon_history = []
        
        # Early stopping variables
        best_avg_reward = -np.inf
        episodes_without_improvement = 0
        early_stopped = False
        stopped_at_episode = num_episodes
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            
            terminated = False
            truncated = False
            
            max_steps = self.get_recommended_max_steps()
            while not (terminated or truncated) and steps < max_steps:
                # Choose action
                action = self.choose_action(state, training=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(int(action))
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, terminated or truncated)
                
                # Update tracking
                total_reward += reward
                steps += 1
                state = next_state
            
            # Store episode results
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Store for animation
            if episode % animate_every == 0:
                self.q_history.append(self.Q.copy())
                self.policy_history.append(self.get_policy_from_q())
                self.epsilon_history.append(self.epsilon)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")
                
                # Early stopping check
                if early_stopping and episode >= 100:
                    current_avg_reward = np.mean(self.episode_rewards[-100:])
                    
                    # Check if we have improvement
                    if current_avg_reward > best_avg_reward + min_improvement:
                        best_avg_reward = current_avg_reward
                        episodes_without_improvement = 0
                        if verbose:
                            print(f"   🎯 New best average reward: {best_avg_reward:.3f}")
                    else:
                        episodes_without_improvement += 100
                        
                    # Check if we should stop
                    if episodes_without_improvement >= patience:
                        early_stopped = True
                        stopped_at_episode = episode + 1
                        print(f"\n🛑 Early stopping triggered at episode {episode + 1}")
                        print(f"   No improvement for {episodes_without_improvement} episodes")
                        print(f"   Best average reward: {best_avg_reward:.3f}")
                        break
        
        # Store final state
        self.q_history.append(self.Q.copy())
        self.policy_history.append(self.get_policy_from_q())
        self.epsilon_history.append(self.epsilon)
        
        print("\n✅ Q-Learning Training Complete!")
        if early_stopped:
            print(f"   (Stopped early at episode {stopped_at_episode}/{num_episodes})")
        print("=" * 50)
        
        # Final statistics
        final_avg_reward = np.mean(self.episode_rewards[-100:])
        final_success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
        print(f"Final Performance (last 100 episodes):")
        print(f"  Average Reward: {final_avg_reward:.3f}")
        print(f"  Success Rate: {final_success_rate:.1f}%")
        print(f"  Final Epsilon: {self.epsilon:.3f}")
        
        return self.Q
    
    def animate_training_progress(self):
        """
        Create an animated visualization of the Q-Learning training progress.
        """
        if not self.q_history:
            print("❌ No training history found. Run train() first.")
            return
        
        print("🎬 Creating Q-Learning Training Animation...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        def animate_frame(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            current_q = self.q_history[frame]
            current_policy = self.policy_history[frame]
            current_epsilon = self.epsilon_history[frame]
            
            # Top-left: Value function (max Q-values)
            V = self.get_value_function_from_q()
            if frame < len(self.q_history) - 1:
                # Use Q-values from history
                V_frame = np.max(current_q, axis=2)
            else:
                V_frame = V
            
            im1 = ax1.imshow(V_frame, cmap='viridis', vmin=0, vmax=1)
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    ax1.text(j, i, f'{V_frame[i, j]:.3f}',
                           ha="center", va="center", color="white", fontweight='bold')
            ax1.set_title('Value Function (Max Q-values)', fontsize=12, fontweight='bold')
            ax1.set_xticks(range(self.env.ncol))
            ax1.set_yticks(range(self.env.nrow))
            
            # Top-right: Policy from Q-values
            self._render_policy_grid(ax2, current_policy)
            ax2.set_title('Policy from Q-values', fontsize=12, fontweight='bold')
            
            # Bottom-left: Q-values for best action
            best_q = np.zeros((self.env.nrow, self.env.ncol))
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    best_action = current_policy[i, j]
                    best_q[i, j] = current_q[i, j, best_action]
            
            im3 = ax3.imshow(best_q, cmap='plasma', vmin=0, vmax=1)
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    ax3.text(j, i, f'{best_q[i, j]:.3f}',
                           ha="center", va="center", color="white", fontweight='bold')
            ax3.set_title('Q-values for Best Actions', fontsize=12, fontweight='bold')
            ax3.set_xticks(range(self.env.ncol))
            ax3.set_yticks(range(self.env.nrow))
            
            # Bottom-right: Learning progress
            episode_idx = frame * 50  # Since we save every 50 episodes
            max_episode = min(episode_idx + 50, len(self.episode_rewards))
            
            if max_episode > 0:
                episodes = range(max_episode)
                rewards = self.episode_rewards[:max_episode]
                
                # Plot episode rewards
                ax4.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5)
                
                # Plot moving average
                if len(rewards) >= 50:
                    window = 50
                    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
                    ax4.plot(episodes, moving_avg, 'r-', linewidth=2, label='Moving Average (50)')
                
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Reward')
                ax4.set_title(f'Learning Progress\nEpsilon: {current_epsilon:.3f}', fontsize=12, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Set y-axis to show 0 and 1
                ax4.set_ylim(-0.1, 1.1)
            else:
                ax4.text(0.5, 0.5, 'Learning Starting...', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=14, fontweight='bold')
                ax4.set_title(f'Learning Progress\nEpsilon: {current_epsilon:.3f}', fontsize=12, fontweight='bold')
            
            fig.suptitle(f'Q-Learning Training - Episode {episode_idx}', 
                        fontsize=16, fontweight='bold')
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(self.q_history),
                                     interval=1000, repeat=True, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _render_policy_grid(self, ax, policy):
        """Render the policy as arrows on a grid"""
        # Draw environment grid
        for i in range(self.env.nrow + 1):
            ax.axhline(y=i - 0.5, color='black', linewidth=1)
        for j in range(self.env.ncol + 1):
            ax.axvline(x=j - 0.5, color='black', linewidth=1)
        
        # Color cells based on environment
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                cell_type = self.env.desc[i, j]
                if cell_type == 'S':
                    color = 'lightgreen'
                elif cell_type == 'G':
                    color = 'red'
                elif cell_type == 'H':
                    color = 'red'
                else:
                    color = 'lightblue'
                
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, alpha=0.3)
                ax.add_patch(rect)
        
        # Draw policy arrows
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                if (i, j) not in self.env.terminal_states:
                    action = policy[i, j]
                    symbol = self.action_symbols[action]
                    ax.text(j, i, symbol, ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='darkblue')
        
        ax.set_xlim(-0.5, self.env.ncol - 0.5)
        ax.set_ylim(-0.5, self.env.nrow - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.env.ncol))
        ax.set_yticks(range(self.env.nrow))
    
    def run_learned_episode(self, max_steps=100, delay=1.0):
        """
        Run an episode using the learned Q-values with visualization.
        
        Args:
            max_steps (int): Maximum steps to prevent infinite loops
            delay (float): Delay between steps for visualization
        """
        if np.all(self.Q == 0):
            print("❌ No Q-values found. Run train() first.")
            return None, None, False
        
        print("🎮 Running Episode with Learned Q-Values...")
        print("=" * 50)
        
        # Set up visualization
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        plt.show()
        
        # Reset environment
        state, info = self.env.reset()
        total_reward = 0
        step_count = 0
        episode_history = [state]
        action_history = []
        reward_history = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and step_count < max_steps:
            ax1.clear()
            ax2.clear()
            
            # Get action from learned Q-values (greedy, no exploration)
            action = self.choose_action(state, training=False)
            
            # Render current state
            self._render_learned_step(ax1, state, action, step_count)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(int(action))
            
            # Update tracking
            total_reward += reward
            step_count += 1
            episode_history.append(next_state)
            action_history.append(action)
            reward_history.append(reward)
            
            # Render Q-values
            self._render_q_values(ax2, state, action)
            
            print(f"Step {step_count}: {self.action_mapping[int(action)]} → {next_state} (Reward: {reward})")
            
            # Update display
            fig.suptitle(f'Q-Learning - Learned Policy Execution - Step {step_count}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            state = next_state
            
            if delay > 0:
                time.sleep(delay)
        
        # Final summary
        success = state in self.env.terminal_states and total_reward > 0
        
        # Show final results
        ax1.clear()
        ax2.clear()
        self._render_final_results(ax1, ax2, state, success, step_count, total_reward, episode_history)
        
        fig.suptitle(f'Episode Complete! Success: {success}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print("\n" + "=" * 50)
        print("📊 LEARNED EPISODE SUMMARY")
        print("=" * 50)
        print(f"🎯 Final State: {state}")
        print(f"🏆 Total Reward: {total_reward}")
        print(f"👣 Steps Taken: {step_count}")
        print(f"✅ Success: {success}")
        print(f"🔚 Terminated: {terminated}")
        print(f"⏱️ Truncated: {truncated}")
        
        plt.ioff()
        input("Press Enter to close the visualization...")
        plt.close(fig)
        
        return total_reward, step_count, success
    
    def _render_learned_step(self, ax, state, action, step_count):
        """Render current step with learned action"""
        # Draw environment
        for i in range(self.env.nrow + 1):
            ax.axhline(y=i, color='black', linewidth=2)
        for j in range(self.env.ncol + 1):
            ax.axvline(x=j, color='black', linewidth=2)
        
        # Color cells
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                cell_type = self.env.desc[i, j]
                if cell_type == 'S':
                    color = 'lightgreen'
                    text = 'START'
                elif cell_type == 'F':
                    color = 'lightblue'
                    text = ''
                elif cell_type == 'H':
                    color = 'red'
                    text = 'HOLE'
                elif cell_type == 'G':
                    color = 'red'
                    text = 'GOAL'
                else:
                    color = 'white'
                    text = ''
                
                rect = patches.Rectangle((j, self.env.nrow - i - 1), 1, 1,
                                       facecolor=color, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
                
                if text:
                    ax.text(j + 0.5, self.env.nrow - i - 0.5, text,
                           ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw agent
        agent_row, agent_col = state
        agent_circle = patches.Circle((agent_col + 0.5, self.env.nrow - agent_row - 0.5),
                                    0.25, color='green', alpha=0.9)
        ax.add_patch(agent_circle)
        
        # Draw learned action arrow
        if step_count >= 0:
            arrow_props = dict(arrowstyle='->', color='red', lw=4, alpha=0.9)
            if action == 0:  # LEFT
                ax.annotate('', xy=(agent_col + 0.2, self.env.nrow - agent_row - 0.5),
                           xytext=(agent_col + 0.8, self.env.nrow - agent_row - 0.5), arrowprops=arrow_props)
            elif action == 1:  # DOWN
                ax.annotate('', xy=(agent_col + 0.5, self.env.nrow - agent_row - 0.8),
                           xytext=(agent_col + 0.5, self.env.nrow - agent_row - 0.2), arrowprops=arrow_props)
            elif action == 2:  # RIGHT
                ax.annotate('', xy=(agent_col + 0.8, self.env.nrow - agent_row - 0.5),
                           xytext=(agent_col + 0.2, self.env.nrow - agent_row - 0.5), arrowprops=arrow_props)
            elif action == 3:  # UP
                ax.annotate('', xy=(agent_col + 0.5, self.env.nrow - agent_row - 0.2),
                           xytext=(agent_col + 0.5, self.env.nrow - agent_row - 0.8), arrowprops=arrow_props)
        
        ax.set_xlim(0, self.env.ncol)
        ax.set_ylim(0, self.env.nrow)
        ax.set_aspect('equal')
        ax.set_title(f'Current State: {state}\nLearned Action: {self.action_mapping[action]}',
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_q_values(self, ax, current_state, chosen_action):
        """Render Q-values for current state with chosen action highlighted"""
        row, col = self.state_to_coords(current_state)
        q_values = self.Q[row, col]
        
        # Create bar plot of Q-values
        actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        colors = ['red' if i == chosen_action else 'lightblue' for i in range(4)]
        
        bars = ax.bar(actions, q_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, q_val) in enumerate(zip(bars, q_values)):
            height = bar.get_height()
            label_color = 'white' if i == chosen_action else 'black'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{q_val:.3f}', ha='center', va='bottom', 
                   fontweight='bold', color=label_color)
        
        # Highlight chosen action
        bars[chosen_action].set_edgecolor('red')
        bars[chosen_action].set_linewidth(3)
        
        ax.set_title(f'Q-Values for State {current_state}\n(Red = Chosen Action)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Q-Value')
        ax.set_ylim(0, max(1.0, max(q_values) + 0.1))
        ax.grid(True, alpha=0.3)
    
    def _render_final_results(self, ax1, ax2, final_state, success, steps, total_reward, episode_history):
        """Render final episode results"""
        # Left plot: Final state
        self._render_learned_step(ax1, final_state, 0, steps)  # No action arrow
        
        if success:
            ax1.text(0.5, 0.95, '🎉 Q-LEARNING SUCCESS! 🎉', transform=ax1.transAxes,
                    ha='center', va='top', fontsize=16, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax1.text(0.5, 0.95, '❌ FAILED ❌', transform=ax1.transAxes,
                    ha='center', va='top', fontsize=16, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Right plot: Learning curve
        if len(self.episode_rewards) > 0:
            episodes = range(len(self.episode_rewards))
            ax2.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Episode Rewards')
            
            # Moving average
            if len(self.episode_rewards) >= 50:
                window = 50
                moving_avg = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                             for i in range(len(self.episode_rewards))]
                ax2.plot(episodes, moving_avg, 'r-', linewidth=2, label='Moving Average (50)')
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title(f'Q-Learning Training Progress\nFinal Episode: Steps {steps} | Reward {total_reward}',
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
    
    def print_q_values(self, state=None):
        """Print Q-values for a specific state or all states"""
        if state is None:
            print("\n🧠 Q-VALUES FOR ALL STATES")
            print("=" * 60)
            for row in range(self.env.nrow):
                for col in range(self.env.ncol):
                    if (row, col) not in self.env.terminal_states:
                        print(f"State ({row},{col}): {self.Q[row, col]}")
        else:
            row, col = self.state_to_coords(state)
            print(f"\n🧠 Q-VALUES FOR STATE {state}")
            print("=" * 40)
            for action, action_name in self.action_mapping.items():
                print(f"{action_name}: {self.Q[row, col, action]:.4f}")
    
    def print_policy(self):
        """Print the learned policy in a readable format"""
        policy = self.get_policy_from_q()
        
        print("\n📋 LEARNED POLICY FROM Q-VALUES")
        print("=" * 35)
        
        for i in range(self.env.nrow):
            row_str = "|"
            for j in range(self.env.ncol):
                if (i, j) in self.env.terminal_states:
                    if (i, j) == (4, 4):
                        row_str += " G |"
                    else:
                        row_str += " H |"
                else:
                    symbol = self.action_symbols[policy[i, j]]
                    row_str += f" {symbol} |"
            print(row_str)
        
        print("=" * 35)
        print("Legend: ← LEFT, ↓ DOWN, → RIGHT, ↑ UP, G GOAL, H HOLE")
    
    def generate_results_folder(self, folder_name=None):
        """
        Generate a comprehensive results folder with:
        1. Cumulative rewards plot
        2. GIF animation of the training process
        3. GIF animation of the final learned path
        4. Training summary text file
        """
        if not self.episode_rewards:
            print("❌ No training data available. Run train() first.")
            return

        # Create folder name with timestamp
        if folder_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"q_learning_results_{timestamp}"

        # Create the results folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        print(f"\n📁 Generating results in folder: {folder_name}")
        print("=" * 60)

        # 1. Generate cumulative rewards plot
        print("📈 Creating cumulative rewards plot...")
        self._save_rewards_plot(folder_name)

        # 2. Generate GIF animation of the training process
        print("🎬 Creating training process animation (this may take a moment)...")
        self._save_training_animation_gif(folder_name)

        # 3. Generate GIF animation for final learned behavior
        print("🚶‍♂️ Creating final pathfinding animation...")
        total_episodes = len(self.episode_rewards)
        # Use the index of the last q_history snapshot
        final_q_index = len(self.q_history) - 1
        if final_q_index >= 0:
            # Calculate the episode number corresponding to the last snapshot
            final_episode_num = (final_q_index) * self.animate_every
            print(f"   Creating animation with final learned policy (from episode ~{final_episode_num})...")
            self._save_pathfinding_gif(folder_name, "final_path", final_q_index, "Final Learned Policy")
        else:
            print("   ⚠️ Could not generate final path GIF: no Q-value history found.")

        # 4. Generate training summary
        print("📝 Creating training summary...")
        self._save_training_summary(folder_name)

        print("\n✅ Results generation complete!")
        print("=" * 60)
        print(f"📂 Results saved in: {os.path.abspath(folder_name)}")
        print(f"   📊 cumulative_rewards.png - Training progress chart")
        print(f"   🎬 training_animation.gif - Animation of Q-values and policy evolution")
        print(f"   🚶‍♂️ final_path.gif - Agent executing the final learned policy")
        print(f"   📄 training_summary.txt - Detailed training statistics")

        return folder_name
    
    def _save_training_animation_gif(self, folder_name):
        """Create and save an animated GIF of the Q-Learning training progress."""
        if not self.q_history:
            print("   - ⚠️ No training history for GIF. Skipping training animation.")
            return

        gif_path = os.path.join(folder_name, 'training_animation.gif')
        print(f"   - Generating training animation at {gif_path}...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        def animate_frame(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()

            current_q = self.q_history[frame]
            current_policy = self.policy_history[frame]
            current_epsilon = self.epsilon_history[frame]
            episode_idx = frame * self.animate_every

            # Top-left: Value function
            V_frame = np.max(current_q, axis=2)
            ax1.imshow(V_frame, cmap='viridis', vmin=0, vmax=1)
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    ax1.text(j, i, f'{V_frame[i, j]:.2f}', ha="center", va="center", color="white", fontsize=8)
            ax1.set_title('Value Function (Max Q)', fontsize=10)
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Top-right: Policy
            self._render_policy_grid(ax2, current_policy)
            ax2.set_title('Learned Policy', fontsize=10)

            # Bottom-left: Q-values for best action
            best_q = np.zeros((self.env.nrow, self.env.ncol))
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    best_action = current_policy[i, j]
                    best_q[i, j] = current_q[i, j, best_action]
            ax3.imshow(best_q, cmap='plasma', vmin=0, vmax=1)
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    ax3.text(j, i, f'{best_q[i, j]:.2f}', ha="center", va="center", color="white", fontsize=8)
            ax3.set_title('Q-Values for Best Action', fontsize=10)
            ax3.set_xticks([])
            ax3.set_yticks([])

            # Bottom-right: Learning progress
            max_episode = min(episode_idx + self.animate_every, len(self.episode_rewards))
            if max_episode > 0:
                episodes = range(max_episode)
                rewards = self.episode_rewards[:max_episode]
                ax4.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5)
                if len(rewards) >= 50:
                    window = 50
                    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
                    ax4.plot(episodes, moving_avg, 'r-', linewidth=2, label='Avg Reward (50 ep)')
                ax4.set_ylim(-0.1, 1.1)
                ax4.legend(fontsize=8)
            ax4.set_title(f'Learning Progress (ε: {current_epsilon:.3f})', fontsize=10)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.grid(True, alpha=0.3)

            fig.suptitle(f'Q-Learning Training - Episode {episode_idx}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            return []

        anim = animation.FuncAnimation(fig, animate_frame, frames=len(self.q_history), interval=500, blit=False)
        
        try:
            anim.save(gif_path, writer='pillow', fps=2)
            print(f"   ✅ Training animation saved successfully.")
        except Exception as e:
            print(f"   ❌ ERROR: Failed to save training animation GIF: {e}")
        
        plt.close(fig)

    def _save_rewards_plot(self, folder_name):
        """Save cumulative rewards plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(self.episode_rewards))
        ax.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Episode Rewards')
        
        # Moving average
        if len(self.episode_rewards) >= 50:
            window = 50
            moving_avg = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                         for i in range(len(self.episode_rewards))]
            ax.plot(episodes, moving_avg, 'r-', linewidth=2, label='Moving Average (50 episodes)')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.set_title(f'Q-Learning Training Progress - {self.env.nrow}x{self.env.ncol} FrozenLake', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # Add statistics text box
        final_avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        final_success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100 if len(self.episode_rewards) >= 100 else np.mean([r > 0 for r in self.episode_rewards]) * 100
        
        stats_text = f'Final Avg Reward: {final_avg_reward:.3f}\nFinal Success Rate: {final_success_rate:.1f}%\nTotal Episodes: {len(self.episode_rewards)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, 'cumulative_rewards.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _save_pathfinding_gif(self, folder_name, stage_name, episode_idx, stage_label):
        """Generate GIF animation showing pathfinding at a specific training stage"""
        # Restore Q-values from the specific episode
        if episode_idx < len(self.q_history):
            q_snapshot = self.q_history[episode_idx].copy()
        else:
            q_snapshot = self.Q.copy()
        
        # Run an episode using the Q-values from that stage
        fig, ax = plt.subplots(figsize=(10, 10))
        frames = []
        
        # Simulate episode
        state, _ = self.env.reset()
        path = [state]
        max_steps = 100
        visited_states = {state}  # Track visited states to detect loops
        
        print(f"      Starting from: {state}")
        
        for step in range(max_steps):
            # Choose action based on Q-values from that stage
            row, col = self.state_to_coords(state)
            action = np.argmax(q_snapshot[row, col])
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(int(action))
            
            # Debug print
            if step < 5 or terminated or truncated:  # Print first 5 steps and ending
                print(f"      Step {step}: {state} --{self.action_mapping[int(action)]}--> {next_state} (reward={reward}, done={terminated or truncated})")
            
            path.append(next_state)
            
            if terminated or truncated:
                print(f"      Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
                break
            
            # Check if we're stuck in a loop (visiting same state repeatedly)
            if next_state in visited_states and len(path) > 10:
                # If we've been here before, try to continue for a few more steps
                # but break if we're truly stuck
                if path[-5:].count(next_state) >= 3:  # Same state 3 times in last 5 steps
                    print(f"      Warning: Agent stuck in loop at {next_state}, stopping...")
                    break
            
            visited_states.add(next_state)
            state = next_state
        
        # Debug: Print path information
        print(f"      Path length: {len(path)} steps")
        print(f"      Path: {path[:10]}{'...' if len(path) > 10 else ''}")
        
        # If path is too short, something is wrong
        if len(path) < 2:
            print(f"      ERROR: Path is too short! Agent might be starting at goal or episode ended immediately.")
            print(f"      Start state: {self.env.start_state}, Goal state: {self.env.goal}")
            return
        
        # Create animation frames - make sure we have enough frames by repeating first and last
        print(f"      Creating animation frames...")
        
        # Add 3 frames at start (pause at beginning)
        for _ in range(3):
            self._create_pathfinding_frame(fig, ax, path, 0, stage_label, episode_idx, frames)
        
        # Create frames for each step
        for frame_idx in range(len(path)):
            self._create_pathfinding_frame(fig, ax, path, frame_idx, stage_label, episode_idx, frames)
        
        # Add 5 frames at end (pause at goal)
        for _ in range(5):
            self._create_pathfinding_frame(fig, ax, path, len(path) - 1, stage_label, episode_idx, frames)
        
        plt.close(fig)
        
        print(f"      Created {len(frames)} animation frames")
        
        if len(frames) == 0:
            print(f"      ERROR: No frames were created! Cannot generate GIF.")
            return
        elif len(frames) == 1:
            print(f"      WARNING: Only 1 frame created. GIF will appear static.")
        
        # Save as GIF - try imageio first (better for animations)
        gif_path = os.path.join(folder_name, f'pathfinding_{stage_name}.gif')
        
        print(f"      DEBUG: frames list has {len(frames)} numpy arrays")
        print(f"      DEBUG: First frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
        
        try:
            import imageio
            print(f"      DEBUG: Using imageio for GIF creation...")
            imageio.mimsave(gif_path, frames, duration=0.5, loop=0)
            print(f"      ✅ GIF saved successfully with imageio: {gif_path}")
            
            # Verify
            from PIL import Image
            test_img = Image.open(gif_path)
            print(f"      DEBUG: Saved GIF has {getattr(test_img, 'n_frames', 'N/A')} frames")
        except ImportError:
            print(f"      DEBUG: imageio not available, using PIL...")
            try:
                from PIL import Image
                
                # Convert numpy arrays to PIL Images - ensure RGB mode
                pil_frames = []
                for i, frame in enumerate(frames):
                    img = Image.fromarray(frame.astype('uint8'), 'RGB')
                    pil_frames.append(img)
                
                print(f"      DEBUG: Converted to {len(pil_frames)} PIL Images")
                print(f"      Saving {len(pil_frames)} frames to GIF...")
                
                # Save as animated GIF with explicit parameters
                pil_frames[0].save(
                    gif_path,
                    format='GIF',
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=500,  # milliseconds per frame
                    loop=0,  # 0 means infinite loop
                    optimize=False  # Don't optimize, might remove frames
                )
                
                print(f"      ✅ GIF saved successfully with PIL: {gif_path}")
                
                # Verify
                test_img = Image.open(gif_path)
                print(f"      DEBUG: Saved GIF has {getattr(test_img, 'n_frames', 'N/A')} frames")
            except Exception as e:
                print(f"      ERROR: Failed to save GIF with PIL: {e}")
                # Fallback to matplotlib animation
                fig_save, ax_save = plt.subplots(figsize=(10, 10))
                im = ax_save.imshow(frames[0])
                ax_save.axis('off')
                
                def update(frame_idx):
                    im.set_array(frames[frame_idx])
                    return [im]
                
                anim = animation.FuncAnimation(fig_save, update, frames=len(frames), 
                                              interval=500, blit=True, repeat=True)
                
                anim.save(gif_path, writer='pillow', fps=2)
                plt.close(fig_save)
        except Exception as e:
            print(f"      ERROR: Failed to create GIF: {e}")
    
    def _create_pathfinding_frame(self, fig, ax, path, frame_idx, stage_label, episode_idx, frames):
        """Create a single frame for the pathfinding animation"""
        state = path[frame_idx]
        ax.clear()
        
        # Draw grid
        for i in range(self.env.nrow + 1):
            ax.axhline(y=i, color='black', linewidth=2)
        for j in range(self.env.ncol + 1):
            ax.axvline(x=j, color='black', linewidth=2)
        
        # Color cells
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                cell_type = self.env.desc[i, j]
                if cell_type == 'S':
                    color = 'lightgreen'
                    text = 'START'
                elif cell_type == 'F':
                    color = 'lightblue'
                    text = ''
                elif cell_type == 'H':
                    color = 'red'
                    text = 'HOLE'
                elif cell_type == 'G':
                    color = 'red'  # GOAL IS RED!
                    text = 'GOAL'
                else:
                    color = 'white'
                    text = ''
                
                rect = patches.Rectangle((j, self.env.nrow - i - 1), 1, 1,
                                       facecolor=color, edgecolor='black', alpha=0.7, linewidth=2)
                ax.add_patch(rect)
                
                if text:
                    ax.text(j + 0.5, self.env.nrow - i - 0.5, text,
                           ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw path taken SO FAR (not the entire path, only up to current frame)
        for path_idx in range(min(frame_idx + 1, len(path))):
            path_state = path[path_idx]
            path_row, path_col = self.state_to_coords(path_state)
            # Make older positions more transparent
            alpha = 0.2 + 0.3 * (path_idx / max(1, frame_idx))
            # Don't draw circle at current position (agent will be there)
            if path_idx < frame_idx:
                circle = patches.Circle((path_col + 0.5, self.env.nrow - path_row - 0.5),
                                      0.12, color='lightblue', alpha=alpha, zorder=5)
                ax.add_patch(circle)
        
        # Draw current agent position - MUCH LARGER AND MORE VISIBLE
        agent_row, agent_col = self.state_to_coords(state)
        # Add a yellow glow behind the agent
        glow_circle = patches.Circle((agent_col + 0.5, self.env.nrow - agent_row - 0.5),
                                    0.42, color='yellow', alpha=0.5, zorder=9)
        ax.add_patch(glow_circle)
        # Main agent circle
        agent_circle = patches.Circle((agent_col + 0.5, self.env.nrow - agent_row - 0.5),
                                    0.35, color='lime', alpha=1.0, zorder=10, 
                                    edgecolor='darkgreen', linewidth=3)
        ax.add_patch(agent_circle)
        # Add text "A" in the agent
        ax.text(agent_col + 0.5, self.env.nrow - agent_row - 0.5, 'A',
               ha='center', va='center', fontsize=12, fontweight='bold', 
               color='darkgreen', zorder=11)
        
        ax.set_xlim(0, self.env.ncol)
        ax.set_ylim(0, self.env.nrow)
        ax.set_aspect('equal')
        ax.set_title(f'{stage_label} (Episode {episode_idx})\nStep {frame_idx}/{len(path)-1}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert plot to image
        fig.canvas.draw()
        # Get the RGBA buffer and convert to RGB numpy array
        buf = fig.canvas.buffer_rgba()  # type: ignore
        image = np.frombuffer(buf, dtype='uint8')  # type: ignore
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        # Copy the RGB data so each frame keeps its own pixels (avoids all frames sharing the same buffer)
        rgb_image = np.array(image[:, :, :3], copy=True)
        frames.append(rgb_image)
    
    def _save_training_summary(self, folder_name):
        """Save detailed training summary to text file"""
        summary_path = os.path.join(folder_name, 'training_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Q-LEARNING TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Environment info
            f.write("ENVIRONMENT CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Grid Size: {self.env.nrow}x{self.env.ncol}\n")
            f.write(f"Total States: {self.env.nrow * self.env.ncol}\n")
            f.write(f"Start State: {self.env.start_state}\n")
            f.write(f"Goal State: {self.env.goal}\n")
            f.write(f"Holes: {self.env.holes}\n")
            f.write(f"Number of Holes: {len(self.env.holes)}\n")
            f.write(f"Hole Density: {len(self.env.holes)/(self.env.nrow * self.env.ncol)*100:.1f}%\n\n")
            
            # Hyperparameters
            f.write("HYPERPARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Learning Rate (α): {self.alpha}\n")
            f.write(f"Discount Factor (γ): {self.gamma}\n")
            f.write(f"Initial Epsilon (ε): 0.9\n")
            f.write(f"Epsilon Decay: {self.epsilon_decay}\n")
            f.write(f"Min Epsilon: {self.epsilon_min}\n")
            f.write(f"Final Epsilon: {self.epsilon:.4f}\n\n")
            
            # Training statistics
            f.write("TRAINING STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
            f.write(f"Total Steps: {sum(self.episode_steps)}\n")
            f.write(f"Average Steps per Episode: {np.mean(self.episode_steps):.2f}\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            
            # Overall performance
            avg_reward_all = np.mean(self.episode_rewards)
            success_rate_all = np.mean([r > 0 for r in self.episode_rewards]) * 100
            
            f.write(f"Overall Average Reward: {avg_reward_all:.4f}\n")
            f.write(f"Overall Success Rate: {success_rate_all:.2f}%\n\n")
            
            # First 100 episodes
            success_rate_first100 = 0.0  # Default value
            if len(self.episode_rewards) >= 100:
                avg_reward_first100 = np.mean(self.episode_rewards[:100])
                success_rate_first100 = np.mean([r > 0 for r in self.episode_rewards[:100]]) * 100
                f.write(f"First 100 Episodes:\n")
                f.write(f"  Average Reward: {avg_reward_first100:.4f}\n")
                f.write(f"  Success Rate: {success_rate_first100:.2f}%\n\n")
            
            # Last 100 episodes
            if len(self.episode_rewards) >= 100:
                avg_reward_last100 = np.mean(self.episode_rewards[-100:])
                success_rate_last100 = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
                f.write(f"Last 100 Episodes:\n")
                f.write(f"  Average Reward: {avg_reward_last100:.4f}\n")
                f.write(f"  Success Rate: {success_rate_last100:.2f}%\n\n")
                
                # Improvement
                improvement = success_rate_last100 - success_rate_first100
                f.write(f"Improvement: {improvement:+.2f}% success rate\n\n")
            
            # Best and worst episodes
            best_episode = np.argmax(self.episode_rewards)
            worst_episode = np.argmin(self.episode_rewards)
            f.write(f"Best Episode: {best_episode} (Reward: {self.episode_rewards[best_episode]:.4f})\n")
            f.write(f"Worst Episode: {worst_episode} (Reward: {self.episode_rewards[worst_episode]:.4f})\n\n")
            
            # Policy summary
            f.write("LEARNED POLICY\n")
            f.write("-" * 70 + "\n")
            policy = self.get_policy_from_q()
            for i in range(self.env.nrow):
                row_str = ""
                for j in range(self.env.ncol):
                    if (i, j) in self.env.terminal_states:
                        if (i, j) == self.env.goal:
                            row_str += "  G  "
                        else:
                            row_str += "  H  "
                    else:
                        symbol = self.action_symbols[policy[i, j]]
                        row_str += f"  {symbol}  "
                f.write(row_str + "\n")
            
            f.write("\nLegend: ← LEFT, ↓ DOWN, → RIGHT, ↑ UP, G GOAL, H HOLE\n\n")
            
            f.write("=" * 70 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n")

def main():
    """Main demonstration function"""
    print("🚀 FrozenLake Q-Learning Agent")
    print("=" * 50)
    print("This agent uses Q-Learning to learn the optimal action-value function")
    print("through exploration and exploitation with animated visualizations!")
    print("=" * 50)
    
    # Environment selection
    print("\n🎯 Choose your FrozenLake environment:")
    print("1. Default 5x5 environment")
    print("2. Custom 3x3 environment (beginner)")
    print("3. Custom 4x6 environment with holes (intermediate)")
    print("4. Create your own environment (interactive)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            else:
                print("❌ Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create agent based on choice
    if choice == '1':
        print("\n🎮 Creating default 5x5 FrozenLake environment...")
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.9, 
                              epsilon_decay=0.995, epsilon_min=0.01)
    elif choice == '2':
        print("\n🎮 Creating custom 3x3 beginner environment...")
        env_params = {'nrow': 3, 'ncol': 3, 'holes': [(1, 1)], 'goal': (2, 2), 'start_state': (0, 0)}
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.9, 
                              epsilon_decay=0.995, epsilon_min=0.01, env_params=env_params)
    elif choice == '3':
        print("\n🎮 Creating custom 4x6 intermediate environment...")
        env_params = {'nrow': 4, 'ncol': 6, 'holes': [(1, 2), (2, 3), (2, 4), (3, 1)], 
                      'goal': (3, 5), 'start_state': (0, 0)}
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.9, 
                              epsilon_decay=0.995, epsilon_min=0.01, env_params=env_params)
    else:  # choice == '4'
        print("\n🎮 Creating interactive environment...")
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.9, 
                              epsilon_decay=0.995, epsilon_min=0.01, interactive_env=True)
    
    # Display environment info
    env_size = agent.env.nrow * agent.env.ncol
    print(f"\n📏 Environment: {agent.env.nrow}x{agent.env.ncol} ({env_size} states)")
    print(f"🏁 Start: {agent.env.start_state}")
    print(f"🎯 Goal: {agent.env.goal}")
    print(f"🕳️ Holes: {agent.env.holes if agent.env.holes else 'None'}")
    recommended_eps = agent.get_recommended_episodes()
    print(f"🧠 Adaptive Parameters: α={agent.adaptive_alpha:.3f}, ε-decay={agent.adaptive_epsilon_decay:.4f}")
    print(f"📊 Flexible Episodes: {recommended_eps} (auto-calculated)")
    
    if env_size >= 64:
        print(f"🚨 Large Environment Detected! Using enhanced parameters for better learning.")
        print(f"   ⏱️ Training will take longer but results will be much better!")
        print(f"   🔢 Episodes scaled by complexity: {agent.get_episode_calculation_details().split('Final Episodes:')[1].strip()}")
    elif env_size >= 25:
        print(f"⚠️ Medium Environment: Parameters adjusted for optimal learning.")
    
    # Show calculation details for complex environments
    if env_size >= 36 or len(agent.env.holes) > env_size * 0.15:
        print(f"\n💡 Why {recommended_eps} episodes?")
        calc = agent._episode_calculation
        print(f"   • Environment complexity factor: {calc['complexity_factor']:.2f}x")
        print(f"   • Hole density impact: {calc['hole_multiplier']:.1f}x")
        if calc['shape_multiplier'] > 1.1:
            print(f"   • Non-square shape penalty: {calc['shape_multiplier']:.1f}x")
    
    # Show episode options
    agent.print_episode_options()
    
    # Ask user for training preference
    print(f"\n🎮 Choose training intensity:")
    print(f"   1. Quick (fastest)")
    print(f"   2. Standard (balanced)")
    print(f"   3. Recommended (best results) ⭐")
    print(f"   4. Thorough (maximum quality)")
    print(f"   5. Custom (specify episodes)")
    
    while True:
        try:
            intensity_choice = input("\nEnter choice (1-5) or press Enter for Recommended: ").strip()
            if not intensity_choice:
                intensity_choice = "3"
            
            if intensity_choice in ['1', '2', '3', '4', '5']:
                break
            else:
                print("❌ Please enter 1, 2, 3, 4, or 5")
        except KeyboardInterrupt:
            print("\n👋 Training cancelled!")
            return
    
    # Get episode count based on choice
    options = agent.get_episode_options()
    if intensity_choice == '1':
        episodes = options['quick']
        print(f"\n🚀 Training with Quick intensity: {episodes:,} episodes")
    elif intensity_choice == '2':
        episodes = options['standard']
        print(f"\n⚡ Training with Standard intensity: {episodes:,} episodes")
    elif intensity_choice == '3':
        episodes = options['recommended']
        print(f"\n🎯 Training with Recommended intensity: {episodes:,} episodes ⭐")
    elif intensity_choice == '4':
        episodes = options['thorough']
        print(f"\n🔥 Training with Thorough intensity: {episodes:,} episodes")
    else:  # Custom
        while True:
            try:
                episodes = int(input(f"Enter custom episode count (100-30000): "))
                if 100 <= episodes <= 30000:
                    print(f"\n🎛️ Training with Custom intensity: {episodes:,} episodes")
                    break
                else:
                    print("❌ Episodes must be between 100 and 30000")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Training cancelled!")
                return
    
    # Train the agent with selected episodes
    Q = agent.train(num_episodes=episodes, animate_every=max(20, episodes//50), verbose=True)
    
    # Generate results folder with visualizations
    print("\n📦 Generating comprehensive results folder...")
    results_folder = agent.generate_results_folder()
    
    # Print results
    agent.print_policy()
    
    # Show training animation
    print("\n🎬 Showing Q-Learning training animation...")
    print("This shows the evolution of Q-values and policy during learning!")
    print("Close the animation window when ready to proceed...")
    anim = agent.animate_training_progress()
    plt.show()
    
    # Test learned policy
    print("\n🎮 Testing learned policy...")
    total_reward, steps, success = agent.run_learned_episode(max_steps=50, delay=0.8)
    
    # Final performance evaluation
    print("\n🔬 Evaluating learned policy over 100 test episodes...")
    test_rewards = []
    test_steps = []
    test_successes = []
    
    for _ in range(100):
        state, _ = agent.env.reset()
        total_reward = 0
        step_count = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and step_count < 200:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = agent.env.step(int(action))
            total_reward += reward
            step_count += 1
        
        test_rewards.append(total_reward)
        test_steps.append(step_count)
        test_successes.append(total_reward > 0)
    
    print(f"\n🎯 FINAL EVALUATION RESULTS:")
    print(f"   Environment: {agent.env.nrow}x{agent.env.ncol} FrozenLake")
    print(f"   Success Rate: {np.mean(test_successes) * 100:.1f}%")
    print(f"   Average Reward: {np.mean(test_rewards):.3f}")
    print(f"   Average Steps: {np.mean(test_steps):.1f}")
    print(f"   Algorithm: Q-Learning (Model-Free)")
    print(f"   Total Holes: {len(agent.env.holes)}")
    
    if results_folder:
        print(f"\n{'='*70}")
        print(f"🎉 Training Complete! All results saved in: {os.path.abspath(results_folder)}")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()