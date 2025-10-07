import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from frozenlake_env import make_frozen_lake

class VisualizedRandomAgent:
    """
    A random agent with enhanced matplotlib visualization for the FrozenLake environment.
    """
    def __init__(self):
        self.env = make_frozen_lake()  # Following gym.make() philosophy
        self.action_mapping = {0: "LEFT ←", 1: "DOWN ↓", 2: "RIGHT →", 3: "UP ↑"}
        
    def run_visual_episode(self, max_steps=100, delay=1.5):
        """
        Run an episode with beautiful matplotlib visualization.
        
        Args:
            max_steps (int): Maximum steps to prevent infinite loops
            delay (float): Delay between steps for better visualization
        """
        # Set up matplotlib for interactive plotting - single window
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        plt.show()  # Show the window initially
        
        # Initialize episode following Gymnasium philosophy
        state, info = self.env.reset()
        total_reward = 0
        step_count = 0
        episode_history = [state]
        action_history = []
        reward_history = []
        
        print("🎮 Starting Visualized Random Agent Episode")
        print("=" * 50)
        
        terminated = False
        truncated = False
        while not (terminated or truncated) and step_count < max_steps:
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Select random action
            action = np.random.choice(self.env.action_space)
            
            # Render current state on the left subplot
            self.env.render_game_state(ax1, state, step_count, action, self.action_mapping)
            
            # Take step in environment following Gymnasium API
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Update tracking variables
            total_reward += reward
            step_count += 1
            episode_history.append(next_state)
            action_history.append(action)
            reward_history.append(reward)
            
            # Render statistics on the right subplot
            self._render_statistics(ax2, step_count, action_history, reward_history, total_reward)
            
            # Display step information
            print(f"Step {step_count}: {self.action_mapping[action]} → {next_state} (Reward: {reward})")
            
            # Update display in the same window
            fig.suptitle(f'FrozenLake Random Agent - Step {step_count}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            state = next_state
            
            if delay > 0:
                time.sleep(delay)
        
        # Final episode summary
        success = state in self.env.terminal_states and total_reward > 0
        
        # Show final state in the same window
        ax1.clear()
        ax2.clear()
        self._render_final_state(ax1, state, success, step_count, total_reward)
        self._render_episode_path(ax2, episode_history, action_history)
        
        fig.suptitle(f'Episode Complete! Success: {success}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        print("\n" + "=" * 50)
        print("📊 EPISODE SUMMARY")
        print("=" * 50)
        print(f"🎯 Final State: {state}")
        print(f"🏆 Total Reward: {total_reward}")
        print(f"👣 Steps Taken: {step_count}")
        print(f"✅ Success: {success}")
        print(f"🔚 Terminated: {terminated}")
        print(f"⏱️ Truncated: {truncated}")
        
        if step_count >= max_steps:
            print("⏰ Episode terminated due to max steps limit!")
        
        # Keep the final plot open in the same window
        plt.ioff()
        input("Press Enter to close the visualization...")  # Wait for user input
        plt.close(fig)
        self.env.close()  # Clean up environment resources
        
        return total_reward, step_count, success
    
    def _render_statistics(self, ax, step_count, action_history, reward_history, total_reward):
        """Render episode statistics"""
        if len(action_history) == 0:
            ax.text(0.5, 0.5, 'Episode Starting...', ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold')
            ax.set_title('Episode Statistics', fontsize=12, fontweight='bold')
            return
        
        # Action distribution
        action_counts = {i: action_history.count(i) for i in range(4)}
        actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        counts = [action_counts[i] for i in range(4)]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = ax.bar(actions, counts, color=colors, alpha=0.7)
        ax.set_title('Action Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Add text statistics
        ax.text(0.02, 0.98, f'Steps: {step_count}', transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.88, f'Total Reward: {total_reward}', transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _render_final_state(self, ax, final_state, success, steps, total_reward):
        """Render the final state with success/failure indication"""
        self.env.render_game_state(ax, final_state, steps, 0, self.action_mapping)  # No action arrow
        
        # Add success/failure overlay
        if success:
            ax.text(0.5, 0.95, '🎉 SUCCESS! 🎉', transform=ax.transAxes, ha='center', va='top',
                   fontsize=16, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(0.5, 0.95, '❌ FAILED ❌', transform=ax.transAxes, ha='center', va='top',
                   fontsize=16, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax.set_title(f'Final State: {final_state} | Steps: {steps} | Reward: {total_reward}', 
                    fontsize=12, fontweight='bold')
    
    def _render_episode_path(self, ax, episode_history, action_history):
        """Render the path taken during the episode"""
        if len(episode_history) < 2:
            ax.text(0.5, 0.5, 'No path to show', ha='center', va='center', transform=ax.transAxes,
                   fontsize=14)
            ax.set_title('Episode Path', fontsize=12, fontweight='bold')
            return
        
        # Convert states to coordinates
        x_coords = [state[1] + 0.5 for state in episode_history]
        y_coords = [self.env.nrow - state[0] - 0.5 for state in episode_history]
        
        # Draw the path
        ax.plot(x_coords, y_coords, 'ro-', linewidth=2, markersize=8, alpha=0.7, label='Agent Path')
        
        # Mark start and end
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=12, label='End')
        
        # Add step numbers
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=8, 
                   fontweight='bold', color='white')
        
        ax.set_xlim(0, self.env.ncol)
        ax.set_ylim(0, self.env.nrow)
        ax.set_aspect('equal')
        ax.set_title('Episode Path', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("🚀 FrozenLake Visual Random Agent")
    print("Following Gymnasium Philosophy:")
    print("env = make_frozen_lake()")
    print("state, info = env.reset()")
    print("terminated = False")
    print("while not terminated:")
    print("    state, reward, terminated, truncated, info = env.step(action)")
    print()
    print("This will show a beautiful matplotlib visualization of the agent's journey!")
    print("Close the matplotlib window to end the program.")
    print()
    
    # Create and run the visualized agent following Gymnasium philosophy
    visual_agent = VisualizedRandomAgent()
    visual_agent.run_visual_episode(max_steps=100, delay=0.3)