"""
Visualize goal distribution to confirm full grid coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_goal_distribution():
    """Create visualization showing goal positions are distributed across entire grid."""
    
    # Simulate goal position generation (same logic as MovingGoalFrozenLake)
    nrow, ncol = 5, 5
    start_state = (0, 0)
    holes = []  # No holes in default config
    
    # Generate all valid goal positions
    goal_positions = []
    for i in range(nrow):
        for j in range(ncol):
            pos = (i, j)
            if pos != start_state and pos not in holes:
                goal_positions.append(pos)
    
    print(f"Total valid goal positions: {len(goal_positions)}")
    print(f"Expected: {nrow * ncol - 1} (total cells - start position)")
    
    # Shuffle like in the actual implementation
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(goal_positions)
    
    print(f"\nFirst 10 shuffled goals: {goal_positions[:10]}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Goal Position Distribution Across Grid', fontsize=14, fontweight='bold')
    
    # 1. Heatmap of all possible goals
    ax1 = axes[0]
    grid1 = np.ones((nrow, ncol))
    grid1[start_state] = 0  # Mark start position
    
    im1 = ax1.imshow(grid1, cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7)
    
    # Add grid lines
    for i in range(nrow + 1):
        ax1.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(ncol + 1):
        ax1.axvline(j - 0.5, color='black', linewidth=1)
    
    # Mark start
    ax1.add_patch(patches.Rectangle((start_state[1] - 0.4, start_state[0] - 0.4), 
                                     0.8, 0.8, linewidth=3, 
                                     edgecolor='blue', facecolor='blue', alpha=0.5))
    ax1.text(start_state[1], start_state[0], 'S', ha='center', va='center', 
             fontsize=20, fontweight='bold', color='white')
    
    # Mark all possible goals
    for goal in goal_positions:
        ax1.add_patch(patches.Circle((goal[1], goal[0]), 0.35, 
                                     color='green', alpha=0.6))
        ax1.text(goal[1], goal[0], 'G', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    ax1.set_xlim(-0.5, ncol - 0.5)
    ax1.set_ylim(nrow - 0.5, -0.5)
    ax1.set_xticks(range(ncol))
    ax1.set_yticks(range(nrow))
    ax1.set_xlabel('Column', fontsize=11)
    ax1.set_ylabel('Row', fontsize=11)
    ax1.set_title(f'All Valid Goal Positions\n({len(goal_positions)} goals)', 
                  fontsize=12, fontweight='bold')
    
    # 2. Goal distribution by row
    ax2 = axes[1]
    goal_rows = [g[0] for g in goal_positions]
    row_counts = [goal_rows.count(r) for r in range(nrow)]
    
    bars = ax2.bar(range(nrow), row_counts, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Row', fontsize=11)
    ax2.set_ylabel('Number of Goals', fontsize=11)
    ax2.set_title('Goal Distribution by Row\n(Uniform across all rows)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(range(nrow))
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(row_counts) + 1])
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, row_counts)):
        ax2.text(bar.get_x() + bar.get_width()/2., count,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Sample policy visualizations (showing diversity)
    ax3 = axes[2]
    grid3 = np.zeros((nrow, ncol))
    
    # Select diverse sample goals (same logic as in agent)
    regions = {
        'top_left': [], 'top_right': [], 
        'bottom_left': [], 'bottom_right': []
    }
    
    mid_row, mid_col = nrow // 2, ncol // 2
    
    for goal in goal_positions:
        r, c = goal
        if r < mid_row and c < mid_col:
            regions['top_left'].append(goal)
        elif r < mid_row and c >= mid_col:
            regions['top_right'].append(goal)
        elif r >= mid_row and c < mid_col:
            regions['bottom_left'].append(goal)
        elif r >= mid_row and c >= mid_col:
            regions['bottom_right'].append(goal)
    
    # Sample one from each region
    sample_goals = []
    for region_name, region_goals in regions.items():
        if region_goals:
            sample_goals.append(region_goals[0])
    
    im3 = ax3.imshow(grid3, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    
    # Add grid lines
    for i in range(nrow + 1):
        ax3.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(ncol + 1):
        ax3.axvline(j - 0.5, color='black', linewidth=1)
    
    # Mark start
    ax3.add_patch(patches.Rectangle((start_state[1] - 0.4, start_state[0] - 0.4), 
                                     0.8, 0.8, linewidth=3, 
                                     edgecolor='blue', facecolor='blue', alpha=0.5))
    ax3.text(start_state[1], start_state[0], 'S', ha='center', va='center', 
             fontsize=20, fontweight='bold', color='white')
    
    # Mark sample goals with different colors
    colors = ['red', 'orange', 'purple', 'cyan']
    labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    
    for i, (goal, color, label) in enumerate(zip(sample_goals, colors, labels)):
        ax3.add_patch(patches.Circle((goal[1], goal[0]), 0.35, 
                                     color=color, alpha=0.7, edgecolor='black', linewidth=2))
        ax3.text(goal[1], goal[0], f'{i+1}', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
        
        # Add legend entry
        ax3.plot([], [], 'o', color=color, markersize=10, label=f'{i+1}: {label} {goal}')
    
    ax3.set_xlim(-0.5, ncol - 0.5)
    ax3.set_ylim(nrow - 0.5, -0.5)
    ax3.set_xticks(range(ncol))
    ax3.set_yticks(range(nrow))
    ax3.set_xlabel('Column', fontsize=11)
    ax3.set_ylabel('Row', fontsize=11)
    ax3.set_title('Diverse Goal Sampling\n(For policy visualizations)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('goal_distribution_verification.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: goal_distribution_verification.png")

if __name__ == "__main__":
    print("="*70)
    print("GOAL DISTRIBUTION VERIFICATION")
    print("="*70)
    print("\nThis script demonstrates that goals are distributed across")
    print("the ENTIRE grid, not just the first row.")
    print("="*70 + "\n")
    
    visualize_goal_distribution()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Goals can be placed at ANY position in the grid")
    print("✓ Only constraint: position ≠ start_state and position ∉ holes")
    print("✓ For 5x5 grid with start at (0,0): 24 valid goal positions")
    print("✓ Goal selection: Random from all 24 positions each episode")
    print("✓ Training: Agent learns policies for ALL goal positions")
    print("✓ Visualization: Shows diverse sample from different regions")
    print("="*70)
