"""
Quick script to generate a GIF from an already trained DQN agent.
Simply run this after training your agent with dqn_agent.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frozenlake_env import make_frozen_lake
from dqn_agent import DQNAgent


def create_navigation_gif(agent, save_folder='dqn_results', num_episodes=5, fps=2):
    """
    Create an animated GIF showing the trained agent navigating the environment.
    
    Args:
        agent: Trained DQNAgent instance
        save_folder: Folder name to save the GIF (relative to script location)
        num_episodes: Number of episodes to show
        fps: Animation speed (frames per second)
    """
    print(f"\n🎬 Creating Agent Navigation GIF")
    print("=" * 70)
    
    # Collect trajectory data from multiple episodes
    episodes_data = []
    
    for ep in range(num_episodes):
        trajectory = []
        state, _ = agent.env.reset()
        final_reward = 0
        
        for step in range(100):  # Max 100 steps per episode
            action = agent.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = agent.env.step(int(action))
            final_reward = reward
            
            trajectory.append({
                'state': state,
                'action': int(action),
                'next_state': next_state,
                'reward': reward,
                'done': terminated or truncated
            })
            
            state = next_state
            
            if terminated or truncated:
                break
        
        episodes_data.append(trajectory)
        status = "✅ Success" if final_reward > 0 else "❌ Failed"
        print(f"  Episode {ep + 1}: {len(trajectory)} steps - {status}")
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate total frames (with pauses between episodes)
    total_frames = sum(len(ep) for ep in episodes_data) + len(episodes_data) * 3
    
    print(f"\n🎨 Rendering {total_frames} frames...")
    
    def get_frame_data(frame_num):
        """Get episode and step indices for a given frame number."""
        frame_count = 0
        for ep_idx, episode in enumerate(episodes_data):
            ep_frames = len(episode) + 3  # episode steps + 3 pause frames
            if frame_num < frame_count + ep_frames:
                step_idx = frame_num - frame_count
                if step_idx >= len(episode):
                    step_idx = len(episode) - 1  # Show last frame during pause
                return ep_idx, step_idx, (frame_num >= frame_count + len(episode))
            frame_count += ep_frames
        return len(episodes_data) - 1, len(episodes_data[-1]) - 1, True
    
    def animate(frame):
        """Draw each frame of the animation."""
        ax.clear()
        
        ep_idx, step_idx, is_pause = get_frame_data(frame)
        trajectory = episodes_data[ep_idx]
        
        if step_idx >= len(trajectory):
            step_idx = len(trajectory) - 1
        
        step_data = trajectory[step_idx]
        state = step_data['state']
        action = step_data['action'] if not is_pause else None
        
        # Draw grid lines
        for i in range(agent.env.nrow + 1):
            ax.axhline(y=i - 0.5, color='black', linewidth=2.5)
        for j in range(agent.env.ncol + 1):
            ax.axvline(x=j - 0.5, color='black', linewidth=2.5)
        
        # Draw environment cells
        for i in range(agent.env.nrow):
            for j in range(agent.env.ncol):
                cell_type = agent.env.desc[i, j]
                
                # Cell colors and labels
                if cell_type == 'S':
                    color, label, text_color = 'lightgreen', 'START', 'darkgreen'
                elif cell_type == 'G':
                    color, label, text_color = 'gold', 'GOAL', 'darkgoldenrod'
                elif cell_type == 'H':
                    color, label, text_color = 'red', 'HOLE', 'darkred'
                else:
                    color, label, text_color = 'lightblue', '', 'black'
                
                # Draw cell
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, alpha=0.6, edgecolor='none')
                ax.add_patch(rect)
                
                # Draw label
                if label:
                    ax.text(j, i + 0.35, label, ha='center', va='center',
                           fontsize=9, fontweight='bold', color=text_color, alpha=0.8)
        
        # Draw agent (robot emoji)
        row, col = state
        circle = patches.Circle((col, row), 0.35, facecolor='blue',
                               edgecolor='darkblue', linewidth=3, zorder=10)
        ax.add_patch(circle)
        ax.text(col, row, '🤖', ha='center', va='center',
               fontsize=26, zorder=11)
        
        # Draw action arrow
        if action is not None:
            arrows = {0: ('←', -0.6, 0), 1: ('↓', 0, 0.6), 2: ('→', 0.6, 0), 3: ('↑', 0, -0.6)}
            symbol, dx, dy = arrows[action]
            
            ax.annotate('', xy=(col + dx * 0.7, row + dy * 0.7),
                       xytext=(col, row),
                       arrowprops=dict(arrowstyle='->', lw=4, color='navy'),
                       zorder=9)
        
        # Configure axes
        ax.set_xlim(-0.5, agent.env.ncol - 0.5)
        ax.set_ylim(-0.5, agent.env.nrow - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(agent.env.ncol))
        ax.set_yticks(range(agent.env.nrow))
        ax.tick_params(labelsize=10)
        
        # Title
        status_text = ""
        if step_idx == len(trajectory) - 1:
            if step_data['reward'] > 0:
                status_text = "\n✅ Goal Reached!"
            elif step_data['done']:
                status_text = "\n❌ Fell in Hole"
        
        ax.set_title(f'DQN Agent - Finding the Way\n'
                    f'Episode {ep_idx + 1}/{num_episodes} | Step {step_idx + 1}/{len(trajectory)}'
                    f'{status_text}',
                    fontsize=14, fontweight='bold', pad=15)
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                  interval=1000//fps, blit=True, repeat=True)
    
    # Save GIF
    results_path = os.path.join(os.path.dirname(__file__), save_folder)
    os.makedirs(results_path, exist_ok=True)
    gif_path = os.path.join(results_path, 'agent_navigation.gif')
    
    print(f"\n💾 Saving GIF...")
    print(f"   Location: {gif_path}")
    print(f"   (This may take a moment...)")
    
    try:
        anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        print(f"\n✅ Success! GIF created")
        print(f"📂 {os.path.abspath(gif_path)}")
        print("=" * 70)
        return gif_path
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure pillow is installed: pip install pillow")
        return None
    finally:
        plt.close()


def quick_demo():
    """Quick demonstration - train a small agent and generate GIF."""
    print("\n" + "=" * 70)
    print("🤖 DQN Agent GIF Generator - Quick Demo")
    print("=" * 70)
    
    # Train a small agent quickly
    print("\n🚀 Training agent on 3x3 environment (quick demo)...")
    env_params = {'nrow': 3, 'ncol': 3, 'holes': [(1, 1)], 'goal': (2, 2)}
    agent = DQNAgent(env_params=env_params)
    agent.train(num_episodes=300, verbose=True, save_freq=100)
    
    # Evaluate
    print("\n🔬 Evaluating agent...")
    agent.evaluate(num_episodes=50, render=False)
    
    # Generate all results
    results_folder = agent.generate_results_folder()
    
    # Create GIF
    gif_path = create_navigation_gif(agent, save_folder=os.path.basename(results_folder),
                                    num_episodes=5, fps=2)
    
    print("\n🎉 Demo complete!")
    print(f"📂 Check the results folder: {results_folder}")


if __name__ == "__main__":
    # Check if being run as part of a larger script
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        print("\n" + "=" * 70)
        print("🎬 GIF Generator for Trained DQN Agent")
        print("=" * 70)
        print("\nThis script creates an animated GIF of your trained agent.")
        print("\nUsage:")
        print("  1. First train an agent using dqn_agent.py")
        print("  2. Then import this module and call create_navigation_gif()")
        print("\nOr run with --demo flag for a quick demonstration:")
        print("  python create_gif_from_trained_agent.py --demo")
        print("=" * 70)
        
        # Offer to run demo
        run_demo = input("\nRun quick demo now? (y/n): ").strip().lower()
        if run_demo == 'y':
            quick_demo()
