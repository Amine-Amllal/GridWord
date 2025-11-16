import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
from frozenlake_env import make_frozen_lake

class QLearningAgent:
    """
    A Q-Learning agent with animated visualization for the FrozenLake environment.
    Implements the Q-Learning algorithm and provides visual animations of:
    1. Q-value evolution during training
    2. Policy derivation from Q-values
    3. Episode execution with learned policy
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum epsilon value
        """
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
        row , col = self.state_to_coords(next_state)
        phi_next_state = np.abs(row - 4) + np.abs(col - 4)
        new_row, new_col = self.state_to_coords(state)
        phi_state = np.abs(new_row - 4) + np.abs(new_col - 4)
        reward = reward + self.gamma * phi_next_state - phi_state
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
    
    def train(self, num_episodes=1000, animate_every=50, verbose=True):
        """
        Train the Q-Learning agent.
        
        Args:
            num_episodes (int): Number of training episodes
            animate_every (int): Store Q-values for animation every N episodes
            verbose (bool): Print training progress
        """
        print("🎓 Starting Q-Learning Training...")
        print("=" * 50)
        print(f"Episodes: {num_episodes}")
        print(f"Learning Rate (α): {self.alpha}")
        print(f"Discount Factor (γ): {self.gamma}")
        print(f"Initial Epsilon (ε): {self.epsilon}")
        print("=" * 50)
        
        self.episode_rewards = []
        self.episode_steps = []
        self.q_history = []
        self.policy_history = []
        self.epsilon_history = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < 200:  # Max 200 steps per episode
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
        
        # Store final state
        self.q_history.append(self.Q.copy())
        self.policy_history.append(self.get_policy_from_q())
        self.epsilon_history.append(self.epsilon)
        
        print("\n✅ Q-Learning Training Complete!")
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
                    color = 'gold'
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
                    color = 'gold'
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

def main():
    """Main demonstration function"""
    print("🚀 FrozenLake Q-Learning Agent")
    print("=" * 50)
    print("This agent uses Q-Learning to learn the optimal action-value function")
    print("through exploration and exploitation with animated visualizations!")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create agent
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.9, 
                          epsilon_decay=0.995, epsilon_min=0.01)
    
    # Train the agent
    Q = agent.train(num_episodes=1000, animate_every=50, verbose=True)
    
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
    print(f"   Success Rate: {np.mean(test_successes) * 100:.1f}%")
    print(f"   Average Reward: {np.mean(test_rewards):.3f}")
    print(f"   Average Steps: {np.mean(test_steps):.1f}")
    print(f"   Algorithm: Q-Learning (Model-Free)")

if __name__ == "__main__":
    main()