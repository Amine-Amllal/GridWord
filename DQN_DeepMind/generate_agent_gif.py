import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import sys

# Add parent directory to path to import frozenlake_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frozenlake_env import make_frozen_lake
from dqn_agent import DQNAgent


def generate_agent_animation_gif(agent, num_episodes=5, max_steps=100, save_path=None, fps=2):
    """
    Generate an animated GIF showing the agent navigating the environment.
    
    Args:
        agent: Trained DQNAgent
        num_episodes: Number of episodes to animate
        max_steps: Maximum steps per episode
        save_path: Path to save the GIF
        fps: Frames per second for the GIF
    """
    print(f"\n🎬 Generating Agent Navigation GIF...")
    print("=" * 70)
    
    # Collect episodes
    all_episodes_data = []
    
    for ep in range(num_episodes):
        episode_data = []
        state, _ = agent.env.reset()
        final_reward = 0
        
        for step in range(max_steps):
            action = agent.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = agent.env.step(int(action))
            final_reward = reward
            
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated
            })
            
            state = next_state
            
            if terminated or truncated:
                # Add final state
                episode_data.append({
                    'state': state,
                    'action': None,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated
                })
                break
        
        all_episodes_data.append(episode_data)
        print(f"  Episode {ep + 1}: {len(episode_data)} steps, Success: {final_reward > 0}")
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def init():
        """Initialize animation"""
        ax.clear()
        return []
    
    def animate(frame_num):
        """Animate each frame"""
        ax.clear()
        
        # Calculate which episode and step
        frame_count = 0
        current_episode = 0
        current_step = 0
        
        for ep_idx, episode in enumerate(all_episodes_data):
            if frame_num < frame_count + len(episode):
                current_episode = ep_idx
                current_step = frame_num - frame_count
                break
            frame_count += len(episode)
            # Add pause frames between episodes
            if frame_num < frame_count + 2:
                current_episode = ep_idx
                current_step = len(episode) - 1
                break
            frame_count += 2
        
        episode_data = all_episodes_data[current_episode]
        if current_step >= len(episode_data):
            current_step = len(episode_data) - 1
        
        step_data = episode_data[current_step]
        state = step_data['state']
        action = step_data['action']
        
        # Draw grid
        for i in range(agent.env.nrow + 1):
            ax.axhline(y=i - 0.5, color='black', linewidth=2)
        for j in range(agent.env.ncol + 1):
            ax.axvline(x=j - 0.5, color='black', linewidth=2)
        
        # Color cells
        for i in range(agent.env.nrow):
            for j in range(agent.env.ncol):
                cell_type = agent.env.desc[i, j]
                
                if cell_type == 'S':
                    color = 'lightgreen'
                    text = 'START'
                    text_color = 'darkgreen'
                elif cell_type == 'G':
                    color = 'gold'
                    text = 'GOAL'
                    text_color = 'darkgoldenrod'
                elif cell_type == 'H':
                    color = 'red'
                    text = 'HOLE'
                    text_color = 'darkred'
                else:
                    color = 'lightblue'
                    text = ''
                    text_color = 'black'
                
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=color, alpha=0.5)
                ax.add_patch(rect)
                
                if text:
                    ax.text(j, i + 0.3, text, ha='center', va='center',
                           fontsize=8, fontweight='bold', color=text_color, alpha=0.7)
        
        # Draw agent
        agent_row, agent_col = state
        agent_circle = patches.Circle((agent_col, agent_row), 0.3,
                                     facecolor='blue', edgecolor='darkblue',
                                     linewidth=3, zorder=10)
        ax.add_patch(agent_circle)
        ax.text(agent_col, agent_row, '🤖', ha='center', va='center',
               fontsize=24, zorder=11)
        
        # Draw action arrow
        if action is not None:
            action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
            action_deltas = {0: (0, -0.5), 1: (0.5, 0), 2: (0, 0.5), 3: (-0.5, 0)}
            
            symbol = action_symbols[int(action)]
            delta = action_deltas[int(action)]
            
            # Draw arrow
            ax.annotate('', xy=(agent_col + delta[1], agent_row + delta[0]),
                       xytext=(agent_col, agent_row),
                       arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'),
                       zorder=9)
        
        # Set display properties
        ax.set_xlim(-0.5, agent.env.ncol - 0.5)
        ax.set_ylim(-0.5, agent.env.nrow - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(agent.env.ncol))
        ax.set_yticks(range(agent.env.nrow))
        
        # Title with episode info
        status = ""
        if step_data['terminated'] or step_data['truncated']:
            if step_data['reward'] > 0:
                status = " - ✅ GOAL REACHED!"
            else:
                status = " - ❌ FELL IN HOLE"
        
        ax.set_title(f'DQN Agent Navigation\nEpisode {current_episode + 1}/{num_episodes} | Step {current_step + 1}{status}',
                    fontsize=14, fontweight='bold')
        
        return []
    
    # Calculate total frames
    total_frames = sum(len(ep) + 2 for ep in all_episodes_data)  # +2 pause frames between episodes
    
    # Create animation
    print(f"\n🎨 Creating animation with {total_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=total_frames, interval=1000//fps,
                                  blit=True, repeat=True)
    
    # Save as GIF
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'dqn_results', 'agent_navigation.gif')
    
    print(f"💾 Saving GIF to: {save_path}")
    print("   (This may take a minute...)")
    
    try:
        anim.save(save_path, writer='pillow', fps=fps, dpi=100)
        print(f"\n✅ GIF successfully created!")
        print(f"📂 Saved to: {os.path.abspath(save_path)}")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error saving GIF: {e}")
        print("💡 Tip: Make sure 'pillow' is installed (pip install pillow)")
        
    plt.close()
    
    return save_path


def load_or_train_agent():
    """Load a trained agent or train a new one."""
    print("\n🤖 DQN Agent GIF Generator")
    print("=" * 70)
    print("\nOptions:")
    print("1. Train new agent (quick 3x3)")
    print("2. Train new agent (default 5x5)")
    print("3. Train new agent (custom)")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        print("\n🚀 Training on 3x3 environment (fast)...")
        env_params = {'nrow': 3, 'ncol': 3, 'holes': [(1, 1)], 'goal': (2, 2)}
        agent = DQNAgent(env_params=env_params)
        agent.train(num_episodes=500, verbose=True, save_freq=100)
    elif choice == "2":
        print("\n🚀 Training on default 5x5 environment...")
        agent = DQNAgent()
        agent.train(num_episodes=1000, verbose=True, save_freq=100)
    else:
        print("\n🚀 Training on custom environment...")
        agent = DQNAgent(interactive_env=True)
        num_episodes = int(input("Number of episodes [1000]: ").strip() or "1000")
        agent.train(num_episodes=num_episodes, verbose=True, save_freq=100)
    
    return agent


def main():
    """Main function to generate agent navigation GIF."""
    # Train or load agent
    agent = load_or_train_agent()
    
    # Evaluate performance
    print("\n🔬 Evaluating agent...")
    metrics = agent.evaluate(num_episodes=100, render=False)
    
    # Generate results folder with plots
    results_folder = agent.generate_results_folder()
    
    # Generate GIF
    print("\n🎬 Generating agent navigation GIF...")
    num_episodes = int(input("Number of episodes to show in GIF [5]: ").strip() or "5")
    fps = int(input("Frames per second [2]: ").strip() or "2")
    
    gif_path = os.path.join(results_folder, 'agent_navigation.gif')
    generate_agent_animation_gif(agent, num_episodes=num_episodes, max_steps=100, 
                                save_path=gif_path, fps=fps)
    
    print("\n🎉 All done!")
    print("=" * 70)
    print(f"📂 Results folder: {results_folder}")
    print(f"   📊 training_progress.png")
    print(f"   🎯 learned_policy.png")
    print(f"   📄 training_summary.txt")
    print(f"   🎬 agent_navigation.gif (NEW!)")
    print("=" * 70)


if __name__ == "__main__":
    main()
