"""
Quick test script to verify GIF generation works correctly.
This will train a small agent and generate a GIF.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_agent import DQNAgent

def quick_test():
    """Quick test of GIF generation feature."""
    print("\n" + "=" * 70)
    print("🧪 Testing GIF Generation Feature")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Train a small DQN agent (3x3 environment)")
    print("  2. Generate results including an animated GIF")
    print("  3. Verify all files are created")
    print("\nEstimated time: 2-3 minutes")
    print("=" * 70)
    
    input("\nPress Enter to start test...")
    
    # Create small agent for quick testing
    print("\n📝 Creating 3x3 environment...")
    env_params = {'nrow': 3, 'ncol': 3, 'holes': [(1, 1)], 'goal': (2, 2)}
    agent = DQNAgent(env_params=env_params)
    
    # Train for fewer episodes (quick test)
    print("\n🚀 Training agent (300 episodes)...")
    agent.train(num_episodes=300, verbose=True, save_freq=100)
    
    # Evaluate
    print("\n🔬 Evaluating agent...")
    metrics = agent.evaluate(num_episodes=50, render=False)
    
    # Generate results with GIF
    print("\n📁 Generating results (including GIF)...")
    results_folder = agent.generate_results_folder(include_gif=True)
    
    # Verify files
    print("\n✅ Verifying files...")
    expected_files = [
        'training_progress.png',
        'learned_policy.png',
        'training_summary.txt',
        'agent_navigation.gif'
    ]
    
    all_files_exist = True
    for filename in expected_files:
        filepath = os.path.join(results_folder, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_kb = size / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {filename} (MISSING!)")
            all_files_exist = False
    
    print("\n" + "=" * 70)
    if all_files_exist:
        print("🎉 SUCCESS! All files generated correctly.")
        print(f"\n📂 Results location: {os.path.abspath(results_folder)}")
        print("\n💡 Open the GIF to see your agent in action!")
    else:
        print("❌ FAILURE! Some files are missing.")
        print("\n🐛 Troubleshooting:")
        print("  1. Make sure pillow is installed: pip install pillow")
        print("  2. Check for error messages above")
        print("  3. Try running dqn_agent.py directly")
    print("=" * 70)
    
    return all_files_exist


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
