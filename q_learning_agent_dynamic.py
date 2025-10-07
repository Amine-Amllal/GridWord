import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import os
from datetime import datetime
from frozenlake_env import make_frozen_lake

class DynamicQLearningAgent:
    """
    A Dynamic Q-Learning agent for FrozenLake environment where goals and holes move ea        print("Starting Dynamic Q-Learning Training...")
        print("=" * 50)
        print(f"Environment: {self.base_env.nrow}x{self.base_env.ncol} Dynamic FrozenLake")
        print(f"Episodes: {num_episodes}")
        print(f"Movement Probability: {self.move_probability:.0%}")
        print(f"Movement frequency: {self.move_frequency}")
        print(f"Learning Rate (alpha): {self.alpha}")
        print(f"Discount Factor (gamma): {self.gamma}")
        print(f"Initial Epsilon: {self.epsilon}")
        print("=" * 50).
    This creates a much more challenging adaptive learning scenario where the agent must:
    1. Learn that the environment    print(f"\nDynamic Q-Learning Complete!")
    print("The agent has learned to adapt to constantly changing environments!")
    print("\nAll results have been automatically saved to a timestamped folder containing:")
    print("  • Agent pathfinding animation (GIF)")
    print("  • Q-values table and best policy (CSV)")
    print("  • Cumulative reward analysis (PNG)")
    print("  • Complete training summary (TXT)")hanges constantly
    2. Adapt to new goal and hole positions each episode
    3. Balance exploration vs exploitation in a non-stationary environment
    4. Maintain learning capability despite changing rewards
    """
    
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.9, epsilon_decay=0.999, epsilon_min=0.1, 
                 env_params=None, interactive_env=False, move_probability=1.0, move_frequency='episode'):
        """
        Initialize the Dynamic Q-Learning agent.
        
        Args:
            alpha (float): Learning rate (higher for dynamic environments)
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate (higher for dynamic environments)
            epsilon_decay (float): Epsilon decay rate (slower for dynamic environments)
            epsilon_min (float): Minimum epsilon value (higher for dynamic environments)
            env_params (dict): Environment parameters (nrow, ncol, holes, goal, start_state)
            interactive_env (bool): If True, create environment interactively with user input
            move_probability (float): Probability that elements move each episode (0.0-1.0)
            move_frequency (str): When to move elements ('episode', 'step', 'random')
        """
        # Create base environment
        if interactive_env:
            self.base_env = make_frozen_lake(interactive=True)
        elif env_params:
            self.base_env = make_frozen_lake(**env_params)
        else:
            self.base_env = make_frozen_lake()
            
        # Dynamic environment parameters
        self.move_probability = move_probability
        self.move_frequency = move_frequency
        self.original_holes = self.base_env.holes.copy()
        self.original_goal = self.base_env.goal
        self.num_holes = len(self.original_holes)
        
        # Current dynamic state
        self.current_holes = self.original_holes.copy()
        self.current_goal = self.original_goal
        
        # Learning parameters (adjusted for dynamic environment)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_mapping = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
        self.action_symbols = {0: "L", 1: "D", 2: "R", 3: "U"}
        
        # Initialize Q-table: [row, col, action] - needs to work across different configurations
        self.Q = np.zeros((self.base_env.nrow, self.base_env.ncol, 4))
        
        # Tracking for dynamic environment
        self.episode_configurations = []  # Track goal/hole positions per episode
        self.movement_history = []  # Track when movements occurred
        self.adaptation_rewards = []  # Track how well agent adapts
        
        # For animation tracking
        self.q_history = []
        self.policy_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.episode_rewards = []
        self.episode_steps = []
        
        # Adaptive parameters based on environment size
        self._adapt_hyperparameters_to_env_size()
        
        print(f"Dynamic FrozenLake Agent Created!")
        print(f"   Base Environment: {self.base_env.nrow}x{self.base_env.ncol}")
        print(f"   Dynamic Elements: {self.num_holes} holes + 1 goal")
        print(f"   Movement Probability: {self.move_probability:.0%}")
        print(f"   Movement Frequency: {self.move_frequency}")
        print(f"   Adapted Learning Parameters for Dynamic Environment")
    
    def _adapt_hyperparameters_to_env_size(self):
        """Adapt hyperparameters for dynamic environment - more aggressive exploration"""
        env_size = self.base_env.nrow * self.base_env.ncol
        num_holes = len(self.base_env.holes)
        
        # Store original values
        original_alpha = self.alpha
        original_epsilon_decay = self.epsilon_decay
        original_epsilon_min = self.epsilon_min
        
        # Dynamic environments need more aggressive exploration and learning
        if env_size >= 64:  # 8x8 or larger
            if self.alpha <= 0.2:
                self.alpha = 0.4  # Even higher learning rate for dynamic
            if self.epsilon_decay >= 0.999:
                self.epsilon_decay = 0.9999  # Very slow decay
            if self.epsilon_min <= 0.1:
                self.epsilon_min = 0.15  # Maintain high exploration
        elif env_size >= 25:  # 5x5 to 7x7
            if self.alpha <= 0.2:
                self.alpha = 0.3
            if self.epsilon_decay >= 0.999:
                self.epsilon_decay = 0.9995
            if self.epsilon_min <= 0.1:
                self.epsilon_min = 0.12
        else:  # Smaller environments
            if self.alpha <= 0.2:
                self.alpha = 0.25
            if self.epsilon_decay >= 0.999:
                self.epsilon_decay = 0.999
            if self.epsilon_min <= 0.1:
                self.epsilon_min = 0.1
        
        # Store adaptive values
        self.adaptive_alpha = self.alpha
        self.adaptive_epsilon_decay = self.epsilon_decay
        self.adaptive_epsilon_min = self.epsilon_min
        
        if (self.alpha != original_alpha or 
            self.epsilon_decay != original_epsilon_decay or 
            self.epsilon_min != original_epsilon_min):
            print(f"Dynamic environment adaptations applied:")
            print(f"   Learning rate: {original_alpha} -> {self.alpha}")
            print(f"   Epsilon decay: {original_epsilon_decay} -> {self.epsilon_decay}")
            print(f"   Min epsilon: {original_epsilon_min} -> {self.epsilon_min}")
    
    def _generate_valid_positions(self, exclude_positions):
        """Generate list of valid positions excluding specified positions"""
        valid_positions = []
        for row in range(self.base_env.nrow):
            for col in range(self.base_env.ncol):
                if (row, col) not in exclude_positions:
                    valid_positions.append((row, col))
        return valid_positions
    
    def _move_elements(self, force_move=False):
        """Move holes and goal to new random positions"""
        if not force_move and np.random.random() > self.move_probability:
            return False  # No movement this time
        
        # Start with current start position as excluded
        excluded_positions = [self.base_env.start_state]
        new_holes = []
        
        # Place holes in new positions
        for _ in range(self.num_holes):
            valid_positions = self._generate_valid_positions(excluded_positions + new_holes)
            if valid_positions:
                new_hole = np.random.choice(len(valid_positions))
                new_hole_pos = valid_positions[new_hole]
                new_holes.append(new_hole_pos)
            else:
                # If no valid positions, keep some holes in original positions
                if self.original_holes:
                    new_holes.append(self.original_holes[len(new_holes) % len(self.original_holes)])
        
        # Place goal in new position
        excluded_for_goal = excluded_positions + new_holes
        valid_goal_positions = self._generate_valid_positions(excluded_for_goal)
        if valid_goal_positions:
            new_goal_idx = np.random.choice(len(valid_goal_positions))
            new_goal = valid_goal_positions[new_goal_idx]
        else:
            # Fallback to original goal if no valid positions
            new_goal = self.original_goal
        
        # Update current configuration
        old_config = (self.current_holes.copy(), self.current_goal)
        self.current_holes = new_holes
        self.current_goal = new_goal
        new_config = (self.current_holes.copy(), self.current_goal)
        
        # Track the movement
        self.movement_history.append({
            'old_config': old_config,
            'new_config': new_config,
            'episode': len(self.episode_configurations)
        })
        
        return True  # Movement occurred
    
    def _update_environment_description(self):
        """Update the environment description matrix with current hole/goal positions"""
        # Reset environment to all frozen
        self.base_env.desc = np.array([['F' for _ in range(self.base_env.ncol)] 
                                      for _ in range(self.base_env.nrow)])
        
        # Set start position
        start_row, start_col = self.base_env.start_state
        self.base_env.desc[start_row, start_col] = 'S'
        
        # Set current holes
        for hole_row, hole_col in self.current_holes:
            if 0 <= hole_row < self.base_env.nrow and 0 <= hole_col < self.base_env.ncol:
                self.base_env.desc[hole_row, hole_col] = 'H'
        
        # Set current goal
        goal_row, goal_col = self.current_goal
        if 0 <= goal_row < self.base_env.nrow and 0 <= goal_col < self.base_env.ncol:
            self.base_env.desc[goal_row, goal_col] = 'G'
        
        # Update environment's terminal states
        self.base_env.terminal_states = self.current_holes + [self.current_goal]
        self.base_env.goal = self.current_goal
        self.base_env.holes = self.current_holes
    
    def reset_episode(self):
        """Reset for new episode with potentially new hole/goal configuration"""
        # Move elements based on frequency setting
        if self.move_frequency == 'episode':
            moved = self._move_elements()
        else:
            moved = False
        
        # Update environment
        self._update_environment_description()
        
        # Store configuration for this episode
        self.episode_configurations.append({
            'episode': len(self.episode_configurations),
            'holes': self.current_holes.copy(),
            'goal': self.current_goal,
            'moved': moved
        })
        
        # Reset environment state
        state, info = self.base_env.reset()
        
        return state, info, moved
    
    def step(self, action):
        """Take a step in the environment"""
        # Move elements if frequency is 'step'
        if self.move_frequency == 'step':
            if self._move_elements():
                self._update_environment_description()
        
        # Take step in base environment
        next_state, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Check if reached current goal
        if next_state == self.current_goal:
            reward = 1.0
            terminated = True
        elif next_state in self.current_holes:
            reward = 0.0
            terminated = True
        
        return next_state, reward, terminated, truncated, info
    
    def state_to_coords(self, state):
        """Convert state tuple to row, col coordinates"""
        if isinstance(state, tuple):
            return state
        else:
            return (state // self.base_env.ncol, state % self.base_env.ncol)
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        row, col = self.state_to_coords(state)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(self.base_env.action_space)
        else:
            # Exploitation: greedy action
            return np.argmax(self.Q[row, col])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-Learning update rule"""
        row, col = self.state_to_coords(state)
        next_row, next_col = self.state_to_coords(next_state)
        
        # Current Q-value
        current_q = self.Q[row, col, action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_row, next_col])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        self.Q[row, col, action] += self.alpha * (target_q - current_q)
    
    def get_recommended_episodes(self):
        """Get recommended episodes for dynamic environment (needs more training)"""
        env_size = self.base_env.nrow * self.base_env.ncol
        base_episodes = env_size * 80  # More episodes for dynamic environment
        
        # Dynamic environment multiplier
        dynamic_multiplier = 2.0  # Dynamic environments need 2x more episodes
        
        if self.move_probability >= 0.8:
            dynamic_multiplier = 3.0  # Very dynamic = 3x episodes
        elif self.move_probability >= 0.5:
            dynamic_multiplier = 2.5  # Moderately dynamic = 2.5x episodes
        
        recommended = int(base_episodes * dynamic_multiplier)
        
        # Set bounds
        min_episodes = 1000
        max_episodes = 50000
        
        return max(min_episodes, min(recommended, max_episodes))
    
    def train(self, num_episodes=None, animate_every=100, verbose=True):
        """Train the dynamic Q-Learning agent"""
        if num_episodes is None:
            num_episodes = self.get_recommended_episodes()
        
        print("Starting Dynamic Q-Learning Training...")
        print("=" * 50)
        print(f"Environment: {self.base_env.nrow}x{self.base_env.ncol} Dynamic FrozenLake")
        print(f"Episodes: {num_episodes}")
        print(f"Movement Probability: {self.move_probability:.0%}")
        print(f"Movement Frequency: {self.move_frequency}")
        print(f"Learning Rate (alpha): {self.alpha}")
        print(f"Discount Factor (gamma): {self.gamma}")
        print(f"Initial Epsilon: {self.epsilon}")
        print("=" * 50)
        
        self.episode_rewards = []
        self.episode_steps = []
        self.adaptation_rewards = []
        movement_count = 0
        
        for episode in range(num_episodes):
            state, info, moved = self.reset_episode()
            if moved:
                movement_count += 1
            
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            max_steps = 200  # Prevent infinite loops
            while not (terminated or truncated) and steps < max_steps:
                # Choose action
                action = self.choose_action(state, training=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.step(int(action))
                
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
                self.epsilon_history.append(self.epsilon)
            
            # Print progress
            if verbose and (episode + 1) % 200 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                success_rate = np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100
                recent_movements = sum([1 for i in range(max(0, len(self.movement_history) - 100), 
                                              len(self.movement_history))])
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Movements: {recent_movements}/100 | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        print("\\nDynamic Q-Learning Training Complete!")
        print("=" * 50)
        
        # Final statistics
        final_avg_reward = np.mean(self.episode_rewards[-200:]) if len(self.episode_rewards) >= 200 else np.mean(self.episode_rewards)
        final_success_rate = np.mean([r > 0 for r in self.episode_rewards[-200:]]) * 100 if len(self.episode_rewards) >= 200 else 0
        
        print(f"Final Performance (last 200 episodes):")
        print(f"  Average Reward: {final_avg_reward:.3f}")
        print(f"  Success Rate: {final_success_rate:.1f}%")
        print(f"  Total Environment Changes: {movement_count}")
        print(f"  Adaptation Rate: {movement_count/num_episodes:.2%}")
        print(f"  Final Epsilon: {self.epsilon:.3f}")
        
        # Automatically create folder with all outputs
        output_folder = self.save_all_outputs()
        
        return self.Q
    
    def test_adaptation(self, test_episodes=50):
        """Test how well the agent adapts to environment changes"""
        print(f"\\nTesting Dynamic Adaptation ({test_episodes} episodes)...")
        
        adaptation_results = []
        
        for episode in range(test_episodes):
            state, info, moved = self.reset_episode()
            
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < 200:
                action = self.choose_action(state, training=False)  # No exploration
                next_state, reward, terminated, truncated, info = self.step(int(action))
                total_reward += reward
                steps += 1
                state = next_state
            
            adaptation_results.append({
                'episode': episode,
                'reward': total_reward,
                'steps': steps,
                'environment_changed': moved,
                'goal_position': self.current_goal,
                'hole_positions': self.current_holes.copy()
            })
        
        # Analyze adaptation performance
        total_successes = sum([1 for r in adaptation_results if r['reward'] > 0])
        success_rate = total_successes / test_episodes * 100
        avg_steps = np.mean([r['steps'] for r in adaptation_results])
        environment_changes = sum([1 for r in adaptation_results if r['environment_changed']])
        
        print(f"Adaptation Test Results:")
        print(f"   Success Rate: {success_rate:.1f}% ({total_successes}/{test_episodes})")
        print(f"   Average Steps: {avg_steps:.1f}")
        print(f"   Environment Changes: {environment_changes}/{test_episodes}")
        
        if success_rate >= 60:
            performance = "Excellent"
        elif success_rate >= 40:
            performance = "Good" 
        elif success_rate >= 20:
            performance = "Learning"
        else:
            performance = "Needs More Training"
        print(f"   Adaptation Performance: {performance}")
        
        return adaptation_results
    
    def show_current_environment(self):
        """Display current environment configuration"""
        print(f"\\nCurrent Environment Configuration:")
        print(f"   Goal: {self.current_goal}")
        print(f"   Holes: {self.current_holes}")
        
        print("\\n   Grid Layout:")
        for i in range(self.base_env.nrow):
            row_str = "   "
            for j in range(self.base_env.ncol):
                row_str += self.base_env.desc[i, j] + " "
            print(row_str)
    
    def print_dynamic_stats(self):
        """Print statistics about the dynamic environment"""
        if not self.episode_configurations:
            print("No episodes completed yet.")
            return
        
        total_episodes = len(self.episode_configurations)
        total_movements = len(self.movement_history)
        
        print(f"\\nDynamic Environment Statistics:")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Environment Changes: {total_movements}")
        print(f"   Change Rate: {total_movements/total_episodes:.2%}")
        print(f"   Unique Goal Positions: {len(set([ep['goal'] for ep in self.episode_configurations]))}")
        
        # Show some example configurations
        if len(self.episode_configurations) >= 5:
            print(f"\\n   Recent Configurations:")
            for i, config in enumerate(self.episode_configurations[-5:]):
                status = "Changed" if config['moved'] else "Same"
                print(f"     Episode {config['episode']}: Goal {config['goal']}, Holes {len(config['holes'])} {status}")
    
    def save_q_values_table(self, save_path):
        """Save the Q-values table as a formatted CSV file"""
        print(f"Saving Q-values table to: {save_path}")
        
        # Create a comprehensive Q-table output
        with open(save_path, 'w') as f:
            f.write("Q-Learning Agent - Best Q-Values Table\\n")
            f.write(f"Environment: {self.base_env.nrow}x{self.base_env.ncol} Dynamic FrozenLake\\n")
            f.write(f"Training Episodes: {len(self.episode_rewards)}\\n")
            f.write(f"Final Success Rate: {np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100:.1f}%\\n")
            f.write("\\n")
            
            f.write("Q-Values by Position and Action:\\n")
            f.write("Row,Col,LEFT,DOWN,RIGHT,UP,Best_Action,Best_Value\\n")
            
            for row in range(self.base_env.nrow):
                for col in range(self.base_env.ncol):
                    q_vals = self.Q[row, col]
                    best_action = int(np.argmax(q_vals))
                    best_value = np.max(q_vals)
                    f.write(f"{row},{col},{q_vals[0]:.4f},{q_vals[1]:.4f},{q_vals[2]:.4f},{q_vals[3]:.4f},"
                           f"{self.action_mapping[best_action]},{best_value:.4f}\\n")
            
            f.write("\\nPolicy Grid (Best Actions):\\n")
            for row in range(self.base_env.nrow):
                row_str = ""
                for col in range(self.base_env.ncol):
                    best_action = int(np.argmax(self.Q[row, col]))
                    row_str += f"{self.action_symbols[best_action]:>3}"
                f.write(row_str + "\\n")
    
    def create_output_folder(self, base_name="dynamic_q_learning_results"):
        """Create a timestamped folder for saving outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{base_name}_{timestamp}"
        
        # Create folder in the same directory as the script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_dir, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        print(f"\\nCreated output folder: {folder_path}")
        return folder_path
    
    def save_all_outputs(self, folder_path=None):
        """Save all outputs (animation, Q-table, reward curve) to a folder"""
        if folder_path is None:
            folder_path = self.create_output_folder()
        
        print(f"\\n{'='*50}")
        print("SAVING ALL OUTPUTS")
        print(f"{'='*50}")
        
        # 1. Save Q-values table
        q_table_path = os.path.join(folder_path, "q_values_table.csv")
        self.save_q_values_table(q_table_path)
        
        # 2. Save cumulative reward curve
        reward_curve_path = os.path.join(folder_path, "cumulative_rewards.png")
        self.plot_cumulative_rewards(save_path=reward_curve_path, show_plot=False)
        
        # 3. Save agent animation
        animation_path = os.path.join(folder_path, "agent_pathfinding_animation.gif")
        print(f"Creating and saving pathfinding animation...")
        self.animate_agent_pathfinding(num_episodes=5, save_path=animation_path)
        
        # 4. Save summary text file
        summary_path = os.path.join(folder_path, "training_summary.txt")
        self.save_training_summary(summary_path)
        
        print(f"\\n{'='*50}")
        print("ALL OUTPUTS SAVED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Folder location: {folder_path}")
        print(f"Contents:")
        print(f"  - q_values_table.csv (Best Q-values and policy)")
        print(f"  - cumulative_rewards.png (Reward curve analysis)")
        print(f"  - agent_pathfinding_animation.gif (Agent behavior animation)")
        print(f"  - training_summary.txt (Complete training summary)")
        print(f"{'='*50}")
        
        return folder_path
    
    def save_training_summary(self, save_path):
        """Save a comprehensive training summary to a text file"""
        print(f"Saving training summary to: {save_path}")
        
        with open(save_path, 'w') as f:
            f.write("DYNAMIC Q-LEARNING AGENT - TRAINING SUMMARY\\n")
            f.write("="*60 + "\\n\\n")
            
            # Environment info
            f.write("ENVIRONMENT CONFIGURATION\\n")
            f.write("-"*30 + "\\n")
            f.write(f"Grid Size: {self.base_env.nrow}x{self.base_env.ncol}\\n")
            f.write(f"Start Position: {self.base_env.start_state}\\n")
            f.write(f"Current Goal: {self.current_goal}\\n")
            f.write(f"Current Holes: {self.current_holes}\\n")
            f.write(f"Movement Probability: {self.move_probability:.0%}\\n")
            f.write(f"Movement Frequency: {self.move_frequency}\\n\\n")
            
            # Training parameters
            f.write("TRAINING PARAMETERS\\n")
            f.write("-"*30 + "\\n")
            f.write(f"Learning Rate (alpha): {self.alpha}\\n")
            f.write(f"Discount Factor (gamma): {self.gamma}\\n")
            f.write(f"Initial Epsilon: 0.9\\n")
            f.write(f"Final Epsilon: {self.epsilon:.4f}\\n")
            f.write(f"Epsilon Decay: {self.epsilon_decay}\\n")
            f.write(f"Minimum Epsilon: {self.epsilon_min}\\n")
            f.write(f"Total Episodes: {len(self.episode_rewards)}\\n\\n")
            
            # Performance metrics
            if self.episode_rewards:
                f.write("PERFORMANCE METRICS\\n")
                f.write("-"*30 + "\\n")
                f.write(f"Total Reward: {sum(self.episode_rewards)}\\n")
                f.write(f"Average Reward: {np.mean(self.episode_rewards):.4f}\\n")
                f.write(f"Final 100 Episodes Avg: {np.mean(self.episode_rewards[-100:]):.4f}\\n")
                f.write(f"Overall Success Rate: {np.mean([r > 0 for r in self.episode_rewards]) * 100:.1f}%\\n")
                f.write(f"Final 100 Episodes Success Rate: {np.mean([r > 0 for r in self.episode_rewards[-100:]]) * 100:.1f}%\\n")
                f.write(f"Average Steps per Episode: {np.mean(self.episode_steps):.1f}\\n\\n")
            
            # Dynamic environment stats
            if hasattr(self, 'episode_configurations') and self.episode_configurations:
                f.write("DYNAMIC ENVIRONMENT STATISTICS\\n")
                f.write("-"*30 + "\\n")
                total_episodes = len(self.episode_configurations)
                total_movements = len(self.movement_history) if hasattr(self, 'movement_history') else 0
                f.write(f"Total Episodes: {total_episodes}\\n")
                f.write(f"Environment Changes: {total_movements}\\n")
                f.write(f"Change Rate: {total_movements/total_episodes:.2%}\\n")
                unique_goals = len(set([ep['goal'] for ep in self.episode_configurations]))
                f.write(f"Unique Goal Positions: {unique_goals}\\n\\n")
            
            # Best policy
            f.write("LEARNED POLICY (Best Actions per Position)\\n")
            f.write("-"*30 + "\\n")
            for row in range(self.base_env.nrow):
                row_str = ""
                for col in range(self.base_env.ncol):
                    best_action = int(np.argmax(self.Q[row, col]))
                    row_str += f"{self.action_symbols[best_action]:>4}"
                f.write(row_str + "\\n")
            
            f.write("\\nLEGEND: L=LEFT, D=DOWN, R=RIGHT, U=UP\\n")
            f.write(f"\\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")

    def get_cumulative_rewards(self):
        """
        Calculate and return cumulative reward statistics over training episodes.
        
        Returns:
            dict: Dictionary containing cumulative reward analysis including:
                - cumulative_sum: Running sum of rewards
                - moving_averages: Moving averages over different windows
                - success_rates: Success rate over time
                - adaptation_metrics: How well agent adapts to environment changes
        """
        if not self.episode_rewards:
            print("No training data available. Please train the agent first.")
            return None
        
        episodes = list(range(1, len(self.episode_rewards) + 1))
        rewards = np.array(self.episode_rewards)
        
        # Calculate cumulative sum
        cumulative_sum = np.cumsum(rewards)
        
        # Calculate moving averages with different window sizes
        windows = [10, 50, 100]
        moving_averages = {}
        
        for window in windows:
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                # Pad the beginning with NaN to maintain length
                padded_avg = np.full(len(rewards), np.nan)
                padded_avg[window-1:] = moving_avg
                moving_averages[f'ma_{window}'] = padded_avg
        
        # Calculate success rates over time
        success_indicators = (rewards > 0).astype(int)
        success_rates = {}
        
        for window in windows:
            if len(rewards) >= window:
                success_rate = np.convolve(success_indicators, np.ones(window)/window, mode='valid')
                padded_rate = np.full(len(rewards), np.nan)
                padded_rate[window-1:] = success_rate * 100  # Convert to percentage
                success_rates[f'success_rate_{window}'] = padded_rate
        
        # Calculate adaptation metrics (performance after environment changes)
        adaptation_performance = []
        if hasattr(self, 'movement_history') and self.movement_history:
            for movement in self.movement_history:
                episode_idx = movement.get('episode', 0)
                if episode_idx < len(self.episode_rewards):
                    # Look at performance in next 5 episodes after change
                    post_change_episodes = min(5, len(self.episode_rewards) - episode_idx)
                    if post_change_episodes > 0:
                        post_change_rewards = rewards[episode_idx:episode_idx + post_change_episodes]
                        adaptation_performance.append({
                            'episode': episode_idx,
                            'avg_reward_after_change': np.mean(post_change_rewards),
                            'success_rate_after_change': np.mean(post_change_rewards > 0) * 100
                        })
        
        return {
            'episodes': episodes,
            'rewards': rewards,
            'cumulative_sum': cumulative_sum,
            'moving_averages': moving_averages,
            'success_rates': success_rates,
            'adaptation_performance': adaptation_performance,
            'total_episodes': len(episodes),
            'total_reward': cumulative_sum[-1] if len(cumulative_sum) > 0 else 0,
            'average_reward': np.mean(rewards),
            'final_100_avg': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'overall_success_rate': np.mean(rewards > 0) * 100
        }
    
    def plot_cumulative_rewards(self, save_path=None, show_plot=True):
        """
        Plot comprehensive cumulative reward analysis.
        
        Args:
            save_path (str): Path to save the plot (optional)
            show_plot (bool): Whether to display the plot
        """
        reward_data = self.get_cumulative_rewards()
        if reward_data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dynamic Q-Learning Agent: Cumulative Reward Analysis', fontsize=16, fontweight='bold')
        
        episodes = reward_data['episodes']
        rewards = reward_data['rewards']
        cumulative = reward_data['cumulative_sum']
        
        # Plot 1: Cumulative Reward Over Time
        axes[0, 0].plot(episodes, cumulative, 'b-', linewidth=2, label='Cumulative Reward')
        axes[0, 0].set_title('Cumulative Reward Over Episodes')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Moving Averages
        axes[0, 1].plot(episodes, rewards, 'lightgray', alpha=0.5, label='Episode Reward')
        for ma_name, ma_values in reward_data['moving_averages'].items():
            window_size = int(ma_name.split('_')[1])
            axes[0, 1].plot(episodes, ma_values, linewidth=2, label=f'MA-{window_size}')
        axes[0, 1].set_title('Episode Rewards and Moving Averages')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Success Rate Over Time
        for sr_name, sr_values in reward_data['success_rates'].items():
            window_size = int(sr_name.split('_')[2])
            axes[1, 0].plot(episodes, sr_values, linewidth=2, label=f'Success Rate (MA-{window_size})')
        axes[1, 0].set_title('Success Rate Over Episodes')
        axes[1, 0].set_xlabel('Episodes')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Adaptation Performance
        if reward_data['adaptation_performance']:
            adaptation_episodes = [ap['episode'] for ap in reward_data['adaptation_performance']]
            adaptation_rewards = [ap['avg_reward_after_change'] for ap in reward_data['adaptation_performance']]
            adaptation_success = [ap['success_rate_after_change'] for ap in reward_data['adaptation_performance']]
            
            ax4_twin = axes[1, 1].twinx()
            line1 = axes[1, 1].scatter(adaptation_episodes, adaptation_rewards, 
                                     c='red', s=50, alpha=0.7, label='Avg Reward (5 episodes)')
            line2 = ax4_twin.scatter(adaptation_episodes, adaptation_success, 
                                   c='blue', s=30, alpha=0.7, label='Success Rate (5 episodes)', marker='^')
            
            axes[1, 1].set_title('Performance After Environment Changes')
            axes[1, 1].set_xlabel('Episode (Environment Change)')
            axes[1, 1].set_ylabel('Average Reward', color='red')
            ax4_twin.set_ylabel('Success Rate (%)', color='blue')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = axes[1, 1].get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Environment Changes\\nDetected During Training', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Performance After Environment Changes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cumulative reward plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def animate_agent_pathfinding(self, num_episodes=5, delay=0.5, save_path=None):
        """
        Create an animated visualization of the agent trying to find the path to the goal.
        Shows how the agent explores and adapts to changing environments.
        
        Args:
            num_episodes (int): Number of episodes to animate
            delay (float): Delay between steps in seconds
            save_path (str): Path to save animation as GIF (optional)
        """
        print(f"Creating pathfinding animation for {num_episodes} episodes...")
        
        # Store original training mode
        original_epsilon = self.epsilon
        
        # Use trained policy (low exploration for cleaner visualization)
        self.epsilon = 0.1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Animation data storage
        animation_data = []
        
        for episode in range(num_episodes):
            print(f"Recording episode {episode + 1}/{num_episodes}...")
            
            # Reset environment and track if it changed
            state, info, environment_changed = self.reset_episode()
            
            episode_data = {
                'episode': episode,
                'environment_changed': environment_changed,
                'goal': self.current_goal,
                'holes': self.current_holes.copy(),
                'path': [self.state_to_coords(state)],
                'actions': [],
                'rewards': [],
                'step_count': 0
            }
            
            terminated = False
            truncated = False
            max_steps = 50  # Limit steps for animation
            
            while not (terminated or truncated) and episode_data['step_count'] < max_steps:
                # Choose action (mostly exploitation)
                action = self.choose_action(state, training=False)
                episode_data['actions'].append(action)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.step(int(action))
                episode_data['rewards'].append(reward)
                episode_data['path'].append(self.state_to_coords(next_state))
                episode_data['step_count'] += 1
                
                state = next_state
                
                # Break if reached goal or fell in hole
                if terminated:
                    break
            
            animation_data.append(episode_data)
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        # Create the animation
        def animate_frame(frame):
            ax.clear()
            
            episode_idx = frame // 60  # Each episode gets 60 frames
            if episode_idx >= len(animation_data):
                episode_idx = len(animation_data) - 1
                
            step_idx = frame % 60
            episode_data = animation_data[episode_idx]
            
            # Limit step display to actual path length
            max_step_idx = min(step_idx, len(episode_data['path']) - 1)
            
            # Draw grid
            nrows, ncols = self.base_env.nrow, self.base_env.ncol
            
            # Draw environment
            for i in range(nrows):
                for j in range(ncols):
                    # Grid cell
                    rect = patches.Rectangle((j, nrows-1-i), 1, 1, 
                                          linewidth=1, edgecolor='black', 
                                          facecolor='lightblue', alpha=0.3)
                    ax.add_patch(rect)
                    
            # Draw current episode's holes
            for hole_pos in episode_data['holes']:
                hole_rect = patches.Rectangle((hole_pos[1], nrows-1-hole_pos[0]), 1, 1,
                                            linewidth=2, edgecolor='red',
                                            facecolor='darkred', alpha=0.7)
                ax.add_patch(hole_rect)
                ax.text(hole_pos[1] + 0.5, nrows-1-hole_pos[0] + 0.5, 'H', 
                       ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            
            # Draw current episode's goal
            goal_pos = episode_data['goal']
            goal_rect = patches.Rectangle((goal_pos[1], nrows-1-goal_pos[0]), 1, 1,
                                        linewidth=2, edgecolor='green',
                                        facecolor='lightgreen', alpha=0.8)
            ax.add_patch(goal_rect)
            ax.text(goal_pos[1] + 0.5, nrows-1-goal_pos[0] + 0.5, 'G', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='darkgreen')
            
            # Draw start position
            start_pos = self.base_env.start_state
            start_rect = patches.Rectangle((start_pos[1], nrows-1-start_pos[0]), 1, 1,
                                         linewidth=2, edgecolor='blue',
                                         facecolor='lightcyan', alpha=0.6)
            ax.add_patch(start_rect)
            ax.text(start_pos[1] + 0.5, nrows-1-start_pos[0] + 0.5, 'S', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
            
            # Draw path up to current step
            if max_step_idx > 0:
                path_x = [pos[1] + 0.5 for pos in episode_data['path'][:max_step_idx + 1]]
                path_y = [nrows-1-pos[0] + 0.5 for pos in episode_data['path'][:max_step_idx + 1]]
                
                # Draw path line
                ax.plot(path_x, path_y, 'o-', color='orange', linewidth=3, 
                       markersize=8, alpha=0.8, label='Agent Path')
                
                # Draw current agent position
                current_pos = episode_data['path'][max_step_idx]
                agent_circle = patches.Circle((current_pos[1] + 0.5, nrows-1-current_pos[0] + 0.5), 
                                            0.3, facecolor='green', edgecolor='darkgreen', linewidth=2)
                ax.add_patch(agent_circle)
            
            # Set up plot
            ax.set_xlim(0, ncols)
            ax.set_ylim(0, nrows)
            ax.set_aspect('equal')
            ax.set_xticks(range(ncols + 1))
            ax.set_yticks(range(nrows + 1))
            ax.grid(True)
            
            # Title with episode info
            env_status = "CHANGED" if episode_data['environment_changed'] else "SAME"
            episode_result = "SUCCESS" if episode_data['rewards'] and episode_data['rewards'][-1] > 0 else "EXPLORING"
            ax.set_title(f"Dynamic Q-Learning Agent Pathfinding\\n"
                        f"Episode {episode_idx + 1}/{len(animation_data)} | "
                        f"Step {max_step_idx}/{len(episode_data['path'])-1} | "
                        f"Environment: {env_status} | Status: {episode_result}", 
                        fontsize=12, fontweight='bold')
            
            return []  # Return empty list for FuncAnimation
        
        # Calculate total frames (60 per episode)
        total_frames = len(animation_data) * 60
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=total_frames, 
                                     interval=100, repeat=True, blit=False)
        
        if save_path:
            print(f"Saving animation to {save_path} (this may take a while)...")
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to: {save_path}")
            plt.close(fig)  # Close the figure to avoid displaying it
        else:
            plt.show()
        return anim
    
    def quick_pathfinding_demo(self, episodes=3):
        """
        Quick demonstration of agent pathfinding with step-by-step visualization.
        """
        print(f"\\nQuick Pathfinding Demo ({episodes} episodes)")
        print("=" * 40)
        
        original_epsilon = self.epsilon
        self.epsilon = 0.05  # Low exploration for demo
        
        for episode in range(episodes):
            print(f"\\n--- Episode {episode + 1} ---")
            
            state, info, moved = self.reset_episode()
            print(f"Environment changed: {'Yes' if moved else 'No'}")
            print(f"Goal: {self.current_goal}, Holes: {self.current_holes}")
            
            path = [self.state_to_coords(state)]
            actions_taken = []
            rewards_received = []
            
            terminated = False
            truncated = False
            steps = 0
            max_steps = 20
            
            while not (terminated or truncated) and steps < max_steps:
                action = self.choose_action(state, training=False)
                actions_taken.append(self.action_mapping[int(action)])
                
                next_state, reward, terminated, truncated, info = self.step(int(action))
                rewards_received.append(reward)
                path.append(self.state_to_coords(next_state))
                
                print(f"  Step {steps + 1}: {self.action_mapping[int(action)]} -> {self.state_to_coords(next_state)} (reward: {reward})")
                
                state = next_state
                steps += 1
                
                if terminated:
                    if reward > 0:
                        print(f"  SUCCESS! Reached goal in {steps} steps!")
                    else:
                        print(f"  FAILED! Fell into hole at {self.state_to_coords(state)}")
                    break
            
            if not terminated and steps >= max_steps:
                print(f"  Episode ended after {max_steps} steps (max limit)")
            
            print(f"  Path: {' -> '.join([str(pos) for pos in path])}")
            print(f"  Actions: {' -> '.join(actions_taken)}")
            print(f"  Total reward: {sum(rewards_received)}")
        
        self.epsilon = original_epsilon
        print(f"\\nPathfinding demo complete!")

def main():
    """Main demonstration of dynamic Q-Learning"""
    print("Dynamic FrozenLake Q-Learning Agent")
    print("=" * 50)
    print("This agent learns in a constantly changing environment!")
    print("Goals and holes move each episode, requiring continuous adaptation.")
    print("=" * 50)
    
    # Environment selection
    print("\\nChoose your Dynamic FrozenLake environment:")
    print("1. Small Dynamic (4x4) - Good for testing")
    print("2. Medium Dynamic (6x6) - Balanced challenge")
    print("3. Large Dynamic (8x8) - Advanced adaptation")
    print("4. Custom Dynamic (interactive)")
    
    while True:
        try:
            choice = input("\\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\\nGoodbye!")
            return
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create agent based on choice
    if choice == '1':
        print("\\nCreating Small Dynamic 4x4 environment...")
        env_params = {'nrow': 4, 'ncol': 4, 'holes': [(1, 1), (2, 2)], 'goal': (3, 3), 'start_state': (0, 0)}
        agent = DynamicQLearningAgent(
            alpha=0.3, gamma=0.95, epsilon=0.9, 
            epsilon_decay=0.999, epsilon_min=0.1,
            env_params=env_params, move_probability=0.8
        )
    elif choice == '2':
        print("\\nCreating Medium Dynamic 6x6 environment...")
        env_params = {'nrow': 6, 'ncol': 6, 'holes': [(2, 2), (3, 4), (4, 1)], 'goal': (5, 5), 'start_state': (0, 0)}
        agent = DynamicQLearningAgent(
            alpha=0.3, gamma=0.95, epsilon=0.9,
            epsilon_decay=0.9995, epsilon_min=0.12,
            env_params=env_params, move_probability=0.7
        )
    elif choice == '3':
        print("\\nCreating Large Dynamic 8x8 environment...")
        env_params = {'nrow': 8, 'ncol': 8, 'holes': [(2, 3), (4, 5), (6, 2), (7, 6)], 'goal': (7, 7), 'start_state': (0, 0)}
        agent = DynamicQLearningAgent(
            alpha=0.4, gamma=0.95, epsilon=0.9,
            epsilon_decay=0.9999, epsilon_min=0.15,
            env_params=env_params, move_probability=0.6
        )
    else:  # choice == '4'
        print("\\nCreating Custom Dynamic environment...")
        agent = DynamicQLearningAgent(
            alpha=0.3, gamma=0.95, epsilon=0.9,
            epsilon_decay=0.9995, epsilon_min=0.12,
            interactive_env=True, move_probability=0.7
        )
    
    # Show initial environment
    agent.show_current_environment()
    
    # Training
    recommended_episodes = agent.get_recommended_episodes()
    print(f"\\nRecommended episodes for dynamic environment: {recommended_episodes}")
    
    # Ask for training intensity
    print(f"\\nChoose training intensity:")
    print(f"   1. Quick ({recommended_episodes//3:,} episodes)")
    print(f"   2. Standard ({recommended_episodes//2:,} episodes)")
    print(f"   3. Recommended ({recommended_episodes:,} episodes) [Best]")
    
    intensity_choice = input("\\nEnter choice (1-3) or press Enter for Standard: ").strip()
    if intensity_choice == "1":
        episodes = recommended_episodes // 3
    elif intensity_choice == "3":
        episodes = recommended_episodes
    else:
        episodes = recommended_episodes // 2
    
    # Train the agent
    print(f"\\nTraining Dynamic Agent with {episodes:,} episodes...")
    Q = agent.train(num_episodes=episodes, animate_every=episodes//20, verbose=True)
    
    # Show dynamic statistics
    agent.print_dynamic_stats()
    
    # Test adaptation
    agent.test_adaptation(test_episodes=20)
    
    # Show final environment
    agent.show_current_environment()
    
    print(f"\\nDynamic Q-Learning Complete!")
    print("The agent has learned to adapt to constantly changing environments!")
    print("\\nAll results have been automatically saved to a timestamped folder containing:")
    print("  • Agent pathfinding animation (GIF)")
    print("  • Q-values table and best policy (CSV)")
    print("  • Cumulative reward analysis (PNG)")
    print("  • Complete training summary (TXT)")

if __name__ == "__main__":
    main()