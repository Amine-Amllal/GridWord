import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
from frozenlake_env import make_frozen_lake

class ValueIterationAgent:
    """
    A Value Iteration agent with animated visualization for the FrozenLake environment.
    Implements the Value Iteration algorithm and provides visual animations of:
    1. Value function convergence during training
    2. Policy execution with optimal actions
    """
    
    def __init__(self, gamma=0.9, theta=1e-6, env_params=None, interactive_env=False):
        """
        Initialize the Value Iteration agent.
        
        Args:
            gamma (float): Discount factor
            theta (float): Convergence threshold for value iteration
            env_params (dict): Environment parameters (nrow, ncol, holes, goal, start_state)
            interactive_env (bool): If True, create environment interactively with user input
        """
        if interactive_env:
            self.env = make_frozen_lake(interactive=True)
        elif env_params:
            self.env = make_frozen_lake(**env_params)
        else:
            self.env = make_frozen_lake()
        self.gamma = gamma
        self.theta = theta
        self.action_mapping = {0: "LEFT ←", 1: "DOWN ↓", 2: "RIGHT →", 3: "UP ↑"}
        self.action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
        # Initialize value function and policy
        self.V = np.zeros((self.env.nrow, self.env.ncol))
        self.policy = np.zeros((self.env.nrow, self.env.ncol), dtype=int)
        
        # For animation tracking
        self.value_history = []
        self.policy_history = []
        self.iteration_count = 0
        
    def state_to_coords(self, state):
        """Convert state tuple to row, col coordinates"""
        if isinstance(state, tuple):
            return state
        else:
            return (state // self.env.ncol, state % self.env.ncol)
    
    def get_transition_prob_and_reward(self, state, action):
        """
        Get transition probabilities and rewards for a given state-action pair.
        In this deterministic environment, probability is 1.0 for the next state.
        """
        row, col = self.state_to_coords(state)
        
        # Initialize next position
        next_row = row
        next_col = col
        
        # Simulate the action
        if action == 0:  # LEFT
            next_col = max(col - 1, 0)
        elif action == 1:  # DOWN
            next_row = min(row + 1, self.env.nrow - 1)
        elif action == 2:  # RIGHT
            next_col = min(col + 1, self.env.ncol - 1)
        elif action == 3:  # UP
            next_row = max(row - 1, 0)
        
        next_state = (next_row, next_col)
        
        # Calculate reward
        reward = 0.0
        if next_state == (4, 4):  # Goal
            reward = 1.0
        
        return [(1.0, next_state, reward, next_state in self.env.terminal_states)]
    
    def value_iteration(self, animate=True):
        """
        Perform Value Iteration algorithm with optional animation.
        
        Args:
            animate (bool): Whether to store iterations for animation
        """
        print("🔄 Starting Value Iteration...")
        print("=" * 50)
        
        self.iteration_count = 0
        self.value_history = []
        self.policy_history = []
        
        while True:
            delta = 0
            new_V = self.V.copy()
            new_policy = self.policy.copy()
            
            # Update value function for each state
            for row in range(self.env.nrow):
                for col in range(self.env.ncol):
                    state = (row, col)
                    
                    # Skip terminal states
                    if state in self.env.terminal_states:
                        continue
                    
                    old_v = self.V[row, col]
                    
                    # Calculate value for each action
                    action_values = []
                    for action in self.env.action_space:
                        transitions = self.get_transition_prob_and_reward(state, action)
                        action_value = 0
                        
                        for prob, next_state, reward, done in transitions:
                            next_row, next_col = self.state_to_coords(next_state)
                            action_value += prob * (reward + self.gamma * self.V[next_row, next_col])
                        
                        action_values.append(action_value)
                    
                    # Update value and policy
                    new_V[row, col] = max(action_values)
                    new_policy[row, col] = np.argmax(action_values)
                    
                    delta = max(delta, abs(old_v - new_V[row, col]))
            
            self.V = new_V
            self.policy = new_policy
            self.iteration_count += 1
            
            # Store for animation
            if animate:
                self.value_history.append(self.V.copy())
                self.policy_history.append(self.policy.copy())
            
            print(f"Iteration {self.iteration_count}: Max value change = {delta:.6f}")
            
            # Check convergence
            if delta < self.theta:
                break
        
        print(f"✅ Value Iteration converged after {self.iteration_count} iterations!")
        print("=" * 50)
        
        return self.V, self.policy
    
    def animate_value_iteration(self):
        """
        Create an animated visualization of the Value Iteration convergence.
        """
        if not self.value_history:
            print("❌ No value iteration history found. Run value_iteration(animate=True) first.")
            return
        
        print("🎬 Creating Value Iteration Animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        def animate_frame(frame):
            ax1.clear()
            ax2.clear()
            
            # Left plot: Value function heatmap
            current_values = self.value_history[frame]
            im = ax1.imshow(current_values, cmap='viridis', vmin=0, vmax=1)
            
            # Add value text on cells
            artists = []
            for i in range(self.env.nrow):
                for j in range(self.env.ncol):
                    text = ax1.text(j, i, f'{current_values[i, j]:.3f}',
                                   ha="center", va="center", color="white", fontweight='bold')
                    artists.append(text)
            
            ax1.set_title(f'Value Function - Iteration {frame + 1}', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(self.env.ncol))
            ax1.set_yticks(range(self.env.nrow))
            
            # Right plot: Policy arrows
            current_policy = self.policy_history[frame]
            self._render_policy_grid(ax2, current_policy)
            ax2.set_title(f'Policy - Iteration {frame + 1}', fontsize=14, fontweight='bold')
            
            fig.suptitle(f'Value Iteration Convergence - Iteration {frame + 1}/{len(self.value_history)}', 
                        fontsize=16, fontweight='bold')
            
            return artists
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(self.value_history),
                                     interval=500, repeat=True, blit=False)
        
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
    
    def run_optimal_episode(self, max_steps=100, delay=1.0):
        """
        Run an episode using the learned optimal policy with visualization.
        
        Args:
            max_steps (int): Maximum steps to prevent infinite loops
            delay (float): Delay between steps for visualization
        """
        if np.all(self.policy == 0) and np.all(self.V == 0):
            print("❌ No policy found. Run value_iteration() first.")
            return None, None, False
        
        print("🎮 Running Optimal Policy Episode...")
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
            
            # Get optimal action from policy
            row, col = self.state_to_coords(state)
            action = int(self.policy[row, col])
            
            # Render current state
            self._render_optimal_step(ax1, state, action, step_count)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Update tracking
            total_reward += reward
            step_count += 1
            episode_history.append(next_state)
            action_history.append(action)
            reward_history.append(reward)
            
            # Render statistics
            self._render_value_function(ax2, state)
            
            print(f"Step {step_count}: {self.action_mapping[action]} → {next_state} (Reward: {reward})")
            
            # Update display
            fig.suptitle(f'Optimal Policy Execution - Step {step_count}', fontsize=16, fontweight='bold')
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
        print("📊 OPTIMAL EPISODE SUMMARY")
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
    
    def _render_optimal_step(self, ax, state, action, step_count):
        """Render current step with optimal action"""
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
        
        # Draw optimal action arrow
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
        ax.set_title(f'Current State: {state}\nOptimal Action: {self.action_mapping[action]}',
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _render_value_function(self, ax, current_state):
        """Render the learned value function"""
        im = ax.imshow(self.V, cmap='viridis', vmin=0, vmax=1)
        
        # Add value text and highlight current state
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                color = 'white'
                if (i, j) == current_state:
                    # Highlight current state
                    rect = patches.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                           facecolor='red', alpha=0.3, linewidth=3, edgecolor='red')
                    ax.add_patch(rect)
                    color = 'yellow'
                
                ax.text(j, i, f'{self.V[i, j]:.3f}',
                       ha="center", va="center", color=color, fontweight='bold')
        
        ax.set_title('Value Function\n(Current state highlighted)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(self.env.ncol))
        ax.set_yticks(range(self.env.nrow))
    
    def _render_final_results(self, ax1, ax2, final_state, success, steps, total_reward, episode_history):
        """Render final episode results"""
        # Left plot: Final state
        self._render_optimal_step(ax1, final_state, 0, steps)  # No action arrow
        
        if success:
            ax1.text(0.5, 0.95, '🎉 OPTIMAL SUCCESS! 🎉', transform=ax1.transAxes,
                    ha='center', va='top', fontsize=16, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax1.text(0.5, 0.95, '❌ FAILED ❌', transform=ax1.transAxes,
                    ha='center', va='top', fontsize=16, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Right plot: Episode path on value function
        im = ax2.imshow(self.V, cmap='viridis', vmin=0, vmax=1, alpha=0.7)
        
        # Draw path
        if len(episode_history) > 1:
            path_rows = [s[0] for s in episode_history]
            path_cols = [s[1] for s in episode_history]
            ax2.plot(path_cols, path_rows, 'ro-', linewidth=3, markersize=8, alpha=0.9, label='Path')
            
            # Mark start and end
            ax2.plot(path_cols[0], path_rows[0], 'go', markersize=12, label='Start')
            ax2.plot(path_cols[-1], path_rows[-1], 'rs', markersize=12, label='End')
        
        # Add value text
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                ax2.text(j, i, f'{self.V[i, j]:.3f}',
                        ha="center", va="center", color='white', fontweight='bold')
        
        ax2.set_title(f'Episode Path on Value Function\nSteps: {steps} | Reward: {total_reward}',
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(range(self.env.ncol))
        ax2.set_yticks(range(self.env.nrow))
        ax2.legend()
    
    def print_policy(self):
        """Print the learned policy in a readable format"""
        if np.all(self.policy == 0) and np.all(self.V == 0):
            print("❌ No policy found. Run value_iteration() first.")
            return
        
        print("\n📋 LEARNED OPTIMAL POLICY")
        print("=" * 30)
        
        for i in range(self.env.nrow):
            row_str = "|"
            for j in range(self.env.ncol):
                if (i, j) in self.env.terminal_states:
                    if (i, j) == (4, 4):
                        row_str += " G |"
                    else:
                        row_str += " H |"
                else:
                    symbol = self.action_symbols[self.policy[i, j]]
                    row_str += f" {symbol} |"
            print(row_str)
        
        print("=" * 30)
        print("Legend: ← LEFT, ↓ DOWN, → RIGHT, ↑ UP, G GOAL, H HOLE")
    
    def print_value_function(self):
        """Print the learned value function"""
        if np.all(self.V == 0):
            print("❌ No value function found. Run value_iteration() first.")
            return
        
        print("\n💰 LEARNED VALUE FUNCTION")
        print("=" * 40)
        
        for i in range(self.env.nrow):
            row_str = "|"
            for j in range(self.env.ncol):
                row_str += f" {self.V[i, j]:.3f} |"
            print(row_str)
        
        print("=" * 40)

def main():
    """Main demonstration function"""
    print("🚀 FrozenLake Value Iteration Agent")
    print("=" * 50)
    print("This agent uses Value Iteration to learn the optimal policy")
    print("and provides beautiful animated visualizations!")
    print("=" * 50)
    
    # Create agent
    agent = ValueIterationAgent(gamma=0.9, theta=1e-6)
    
    # Run Value Iteration
    V, policy = agent.value_iteration(animate=True)
    
    # Print results
    agent.print_value_function()
    agent.print_policy()
    
    # Show convergence animation
    print("\n🎬 Showing Value Iteration convergence animation...")
    print("Close the animation window when ready to proceed...")
    anim = agent.animate_value_iteration()
    plt.show()
    
    # Run optimal episode
    print("\n🎮 Running episode with optimal policy...")
    total_reward, steps, success = agent.run_optimal_episode(max_steps=50, delay=0.8)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Success Rate: {'100%' if success else '0%'}")
    print(f"   Average Steps: {steps}")
    print(f"   Total Reward: {total_reward}")

if __name__ == "__main__":
    main()