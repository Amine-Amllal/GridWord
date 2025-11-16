"""
Run a visual simulation of a trained PPO agent on FrozenLake.
This script shows the agent navigating the grid step-by-step.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozenlake_env import FrozenLakeEnv
from stable_baselines3 import PPO


class FrozenLakeGymWrapper(gym.Env):
    """
    Gymnasium wrapper for FrozenLake environment.
    Converts the custom FrozenLake environment to be compatible with Stable Baselines3.
    """
    
    def __init__(self, nrow=5, ncol=5, holes=None, goal=None, start_state=(0, 0)):
        super().__init__()
        
        # Create the base FrozenLake environment
        if holes is None:
            holes = [(1, 1), (1, 3), (2, 3), (3, 0), (3, 2)]
        if goal is None:
            goal = (nrow - 1, ncol - 1)
        
        self.env = FrozenLakeEnv(
            nrow=nrow,
            ncol=ncol,
            holes=holes,
            goal=goal,
            start_state=start_state
        )
        
        self.nrow = nrow
        self.ncol = ncol
        self.holes = holes
        self.goal = goal
        self.start_state = start_state
        
        # Define action and observation spaces for Stable Baselines3
        self.action_space = spaces.Discrete(4)  # 4 actions: LEFT, DOWN, RIGHT, UP
        self.observation_space = spaces.Discrete(nrow * ncol)  # Flat observation space
        
    def _state_to_obs(self, state):
        """Convert (row, col) state tuple to flat observation integer."""
        row, col = state
        return row * self.ncol + col
    
    def _obs_to_state(self, obs):
        """Convert flat observation integer to (row, col) state tuple."""
        row = obs // self.ncol
        col = obs % self.ncol
        return (row, col)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        state, info = self.env.reset()
        obs = self._state_to_obs(state)
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        state, reward, terminated, truncated, info = self.env.step(action)
        obs = self._state_to_obs(state)
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        pass


def visualize_episode(env, model, episode_num=1, delay=0.5):
    """
    Visualize a single episode with the trained agent.
    
    Args:
        env: The FrozenLake environment
        model: Trained PPO model
        episode_num: Episode number (for display)
        delay: Delay between steps in seconds
    """
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()  # Interactive mode
    
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
    print(f"\n{'='*70}")
    print(f"Episode {episode_num} - Watching Agent Navigate")
    print(f"{'='*70}")
    
    while not done:
        # Clear previous plot
        ax.clear()
        
        # Get agent's action
        action, _states = model.predict(obs, deterministic=True)
        # Convert action to int if it's a numpy array
        action = int(action)
        
        # Get current state for visualization
        current_state = env._obs_to_state(obs)
        
        # Render the environment
        env.env.render_game_state(
            ax=ax,
            state=current_state,
            step_count=step_count,
            current_action=action,
            action_mapping={0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
        )
        
        plt.draw()
        plt.pause(delay)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action_names[action]}, "
              f"State={env._obs_to_state(obs)}, Reward={reward:.1f}")
        
        if done:
            # Show final state
            ax.clear()
            final_state = env._obs_to_state(obs)
            # Ensure action is int for final rendering
            final_action = int(action) if hasattr(action, '__iter__') else action
            env.env.render_game_state(
                ax=ax,
                state=final_state,
                step_count=step_count,
                current_action=final_action,
                action_mapping={0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
            )
            
            if reward > 0:
                result_text = "🎉 SUCCESS! Reached the Goal! 🎉"
                ax.text(env.ncol/2, -0.5, result_text, 
                       ha='center', fontsize=14, fontweight='bold', 
                       color='green', bbox=dict(boxstyle='round', facecolor='lightgreen'))
            else:
                result_text = "💀 FAILED - Fell into a Hole 💀"
                ax.text(env.ncol/2, -0.5, result_text,
                       ha='center', fontsize=14, fontweight='bold',
                       color='red', bbox=dict(boxstyle='round', facecolor='lightcoral'))
            
            plt.draw()
            plt.pause(2)  # Show final state for 2 seconds
    
    plt.ioff()
    plt.close()
    
    print(f"Episode finished: Total Reward = {total_reward:.1f}, Steps = {step_count}")
    print(f"{'='*70}\n")
    
    return total_reward, step_count


def run_visual_simulation(
    model_path=None,
    nrow=5,
    ncol=5,
    holes=None,
    goal=None,
    start_state=(0, 0),
    n_episodes=5,
    delay=0.5,
    train_if_no_model=True
):
    """
    Run a visual simulation of the PPO agent.
    
    Args:
        model_path: Path to saved model (if None, will train a new one)
        nrow, ncol: Grid dimensions
        holes: Hole positions
        goal: Goal position
        start_state: Starting position
        n_episodes: Number of episodes to visualize
        delay: Delay between steps in seconds
        train_if_no_model: Train a new model if no model_path provided
    """
    # Set defaults
    if holes is None:
        holes = [(1, 1), (1, 3), (2, 3), (3, 0), (3, 2)]
    if goal is None:
        goal = (nrow - 1, ncol - 1)
    
    # Load or train model
    if model_path is None:
        if train_if_no_model:
            print("🚀 No model provided. Training a new PPO agent...")
            print("This will take a few minutes...\n")
            
            # Train using simplified approach to avoid progress_bar dependency
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.monitor import Monitor
            from datetime import datetime
            import os
            
            # Create results directory inside StableBase3Zoo_PPO
            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(script_dir, f"ppo_visual_demo_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create environment
            def create_env():
                env = FrozenLakeGymWrapper(nrow=nrow, ncol=ncol, holes=holes, goal=goal, start_state=start_state)
                return Monitor(env)
            
            vec_env = make_vec_env(create_env, n_envs=1, seed=42)
            
            # Create and train model
            print("Training PPO agent...")
            model = PPO("MlpPolicy", vec_env, verbose=1, seed=42)
            model.learn(total_timesteps=50000, progress_bar=False)  # Disable progress bar
            
            # Save model
            model_path = f"{results_dir}/ppo_frozenlake_final"
            model.save(model_path)
            
            vec_env.close()
            print(f"\n✅ Training complete! Model saved to: {model_path}\n")
        else:
            print("❌ No model provided and training disabled.")
            return
    
    # Load the model
    print(f"📦 Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = FrozenLakeGymWrapper(
        nrow=nrow,
        ncol=ncol,
        holes=holes,
        goal=goal,
        start_state=start_state
    )
    
    print("\n" + "="*70)
    print("🎮 Starting Visual Simulation")
    print("="*70)
    print(f"Grid: {nrow}x{ncol}")
    print(f"Start: {start_state}, Goal: {goal}")
    print(f"Holes: {holes}")
    print(f"Episodes: {n_episodes}")
    print(f"Delay: {delay}s per step")
    print("="*70)
    
    # Run episodes
    all_rewards = []
    all_steps = []
    
    for episode in range(1, n_episodes + 1):
        reward, steps = visualize_episode(env, model, episode, delay)
        all_rewards.append(reward)
        all_steps.append(steps)
        
        if episode < n_episodes:
            time.sleep(1)  # Pause between episodes
    
    env.close()
    
    # Print summary
    successes = sum(1 for r in all_rewards if r > 0)
    print("\n" + "="*70)
    print("📊 SIMULATION SUMMARY")
    print("="*70)
    print(f"Episodes Run: {n_episodes}")
    print(f"Successes: {successes}/{n_episodes} ({(successes/n_episodes)*100:.1f}%)")
    print(f"Average Reward: {np.mean(all_rewards):.3f}")
    print(f"Average Steps: {np.mean(all_steps):.1f}")
    print("="*70)


def quick_demo():
    """Quick demo on a 3x3 grid for fast testing."""
    print("\n🎯 Quick Demo - 3x3 Grid")
    run_visual_simulation(
        nrow=3,
        ncol=3,
        holes=[(1, 1)],
        goal=(2, 2),
        start_state=(0, 0),
        n_episodes=3,
        delay=0.8,
        train_if_no_model=True
    )


def standard_demo():
    """Standard demo on the default 5x5 grid."""
    print("\n🎯 Standard Demo - 5x5 Grid")
    run_visual_simulation(
        nrow=5,
        ncol=5,
        holes=[(1, 1), (1, 3), (2, 3), (3, 0), (3, 2)],
        goal=(4, 4),
        start_state=(0, 0),
        n_episodes=5,
        delay=0.5,
        train_if_no_model=True
    )


def use_existing_model_demo(model_path):
    """Demo using an existing trained model."""
    print(f"\n🎯 Demo with Existing Model")
    run_visual_simulation(
        model_path=model_path,
        nrow=5,
        ncol=5,
        holes=[(1, 1), (1, 3), (2, 3), (3, 0), (3, 2)],
        goal=(4, 4),
        start_state=(0, 0),
        n_episodes=5,
        delay=0.5,
        train_if_no_model=False
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎮 PPO Agent Visual Simulation")
    print("="*70)
    print("\nChoose a demo:")
    print("1. Quick Demo (3x3 grid, trains quickly)")
    print("2. Standard Demo (5x5 grid, default setup)")
    print("3. Use existing model (provide path)")
    print("="*70)
    
    choice = input("\nEnter your choice (1-3, or press Enter for Quick Demo): ").strip()
    
    if choice == "2":
        standard_demo()
    elif choice == "3":
        model_path = input("Enter model path (e.g., ppo_results_*/ppo_frozenlake_final): ").strip()
        if os.path.exists(model_path + ".zip"):
            use_existing_model_demo(model_path)
        else:
            print(f"❌ Model not found at: {model_path}")
            print("Running quick demo instead...")
            quick_demo()
    else:
        quick_demo()
    
    print("\n✨ Simulation complete!")
