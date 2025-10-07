import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Any

class FrozenLakeEnv:
    """
    A flexible FrozenLake environment following the Gymnasium philosophy.
    S: Start
    F: Frozen lake (safe)
    H: Hole (terminal)
    G: Goal (terminal)
    """
    def __init__(self, nrow=5, ncol=5, holes=None, goal=None, start_state=(0, 0)):
        """
        Initialize FrozenLake environment with flexible dimensions and layout.
        
        Args:
            nrow (int): Number of rows
            ncol (int): Number of columns
            holes (list): List of (row, col) tuples for hole positions
            goal (tuple): (row, col) tuple for goal position
            start_state (tuple): (row, col) tuple for start position
        """
        self.nrow = nrow
        self.ncol = ncol
        self.num_states = self.nrow * self.ncol
        self.num_actions = 4  # LEFT, DOWN, RIGHT, UP
        
        # Initialize grid with all frozen cells
        self.desc = np.array([['F' for _ in range(ncol)] for _ in range(nrow)])
        
        # Set start position
        self.start_state = start_state
        start_row, start_col = start_state
        if 0 <= start_row < nrow and 0 <= start_col < ncol:
            self.desc[start_row, start_col] = 'S'
        else:
            raise ValueError(f"Start state {start_state} is out of bounds for grid {nrow}x{ncol}")
        
        # Set holes
        if holes is None:
            holes = []
        self.holes = holes
        for hole_row, hole_col in holes:
            if 0 <= hole_row < nrow and 0 <= hole_col < ncol:
                if (hole_row, hole_col) != start_state:
                    self.desc[hole_row, hole_col] = 'H'
                else:
                    raise ValueError(f"Hole position {(hole_row, hole_col)} conflicts with start position")
            else:
                raise ValueError(f"Hole position {(hole_row, hole_col)} is out of bounds for grid {nrow}x{ncol}")
        
        # Set goal
        if goal is None:
            goal = (nrow - 1, ncol - 1)  # Default to bottom-right
        self.goal = goal
        goal_row, goal_col = goal
        if 0 <= goal_row < nrow and 0 <= goal_col < ncol:
            if (goal_row, goal_col) != start_state and (goal_row, goal_col) not in holes:
                self.desc[goal_row, goal_col] = 'G'
            else:
                raise ValueError(f"Goal position {goal} conflicts with start or hole positions")
        else:
            raise ValueError(f"Goal position {goal} is out of bounds for grid {nrow}x{ncol}")
        
        # Set terminal states (holes and goal)
        self.terminal_states = holes + [goal]
        
        self.state = self.start_state
        self.last_action = None
        self.action_space = [0, 1, 2, 3]  # LEFT, DOWN, RIGHT, UP
        self.observation_space = (self.nrow, self.ncol)

    def reset(self, seed=None, options=None) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Reset the environment to initial state following Gymnasium API"""
        if seed is not None:
            self.seed(seed)
        self.state = self.start_state
        self.last_action = None
        info = {}
        return self.state, info

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict[str, Any]]:
        assert action in self.action_space, "Invalid action."
        row, col = self.state
        self.last_action = action
        # Define action effects
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)
        next_state = (row, col)
        self.state = next_state
        terminated = next_state in self.terminal_states
        truncated = False  # No time limit in this environment
        reward = 0.0
        if next_state == self.goal:  # Goal
            reward = 1.0
        info = {}
        return next_state, reward, terminated, truncated, info

    def render(self, mode='matplotlib'):
        if mode == 'text':
            # Text-based rendering (original)
            desc = self.desc.copy()
            row, col = self.state
            desc[row, col] = 'A'  # Agent
            print("\n".join(["".join(row) for row in desc]))
        elif mode == 'matplotlib':
            # Matplotlib-based rendering
            self._render_matplotlib()
    
    def _render_matplotlib(self, fig=None, ax=None):
        """Render the environment using matplotlib"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            new_figure = True
        else:
            new_figure = False
            ax.clear()
        
        # Create grid
        for i in range(self.nrow + 1):
            ax.axhline(y=i, color='black', linewidth=2)
        for j in range(self.ncol + 1):
            ax.axvline(x=j, color='black', linewidth=2)
        
        # Color the cells based on their type
        for i in range(self.nrow):
            for j in range(self.ncol):
                cell_type = self.desc[i, j]
                if cell_type == 'S':  # Start
                    color = 'lightgreen'
                    text = 'START'
                elif cell_type == 'F':  # Frozen (safe)
                    color = 'lightblue'
                    text = ''
                elif cell_type == 'H':  # Hole
                    color = 'red'
                    text = 'HOLE'
                elif cell_type == 'G':  # Goal
                    color = 'gold'
                    text = 'GOAL'
                else:
                    color = 'white'
                    text = ''
                
                # Draw cell background
                rect = patches.Rectangle((j, self.nrow - i - 1), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add text label
                if text:
                    ax.text(j + 0.5, self.nrow - i - 0.5, text, 
                           ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw agent
        agent_row, agent_col = self.state
        agent_circle = patches.Circle((agent_col + 0.5, self.nrow - agent_row - 0.5), 
                                    0.3, color='green', alpha=0.8)
        ax.add_patch(agent_circle)
        ax.text(agent_col + 0.5, self.nrow - agent_row - 0.5, 'AGENT', 
               ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Set up the plot
        ax.set_xlim(0, self.ncol)
        ax.set_ylim(0, self.nrow)
        ax.set_aspect('equal')
        ax.set_title(f'FrozenLake {self.nrow}x{self.ncol} - Agent at {self.state}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(self.ncol + 1))
        ax.set_yticks(range(self.nrow + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if new_figure:
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
        
        return fig, ax

    def render_game_state(self, ax, state, step_count, current_action, action_mapping):
        """Render the current game state with agent and action arrow"""
        # Create grid
        for i in range(self.nrow + 1):
            ax.axhline(y=i, color='black', linewidth=2)
        for j in range(self.ncol + 1):
            ax.axvline(x=j, color='black', linewidth=2)
        
        # Color the cells
        for i in range(self.nrow):
            for j in range(self.ncol):
                cell_type = self.desc[i, j]
                
                if cell_type == 'S':
                    color = 'lightgreen'
                    text = 'START'
                    text_color = 'darkgreen'
                elif cell_type == 'F':
                    color = 'lightblue'
                    text = ''
                    text_color = 'black'
                elif cell_type == 'H':
                    color = 'red'
                    text = 'HOLE'
                    text_color = 'white'
                elif cell_type == 'G':
                    color = 'gold'
                    text = 'GOAL'
                    text_color = 'darkred'
                else:
                    color = 'white'
                    text = ''
                    text_color = 'black'
                
                # Draw cell
                rect = patches.Rectangle((j, self.nrow - i - 1), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
                ax.add_patch(rect)
                
                # Add text
                if text:
                    ax.text(j + 0.5, self.nrow - i - 0.5, text, 
                           ha='center', va='center', fontsize=10, fontweight='bold', color=text_color)
        
        # Draw agent with action arrow
        agent_row, agent_col = state
        
        # Agent circle
        agent_circle = patches.Circle((agent_col + 0.5, self.nrow - agent_row - 0.5), 
                                    0.25, color='green', alpha=0.9, zorder=10)
        ax.add_patch(agent_circle)
        
        # Action arrow
        if step_count > 0:  # Don't show arrow on first step
            arrow_props = dict(arrowstyle='->', color='red', lw=3, alpha=0.8)
            if current_action == 0:  # LEFT
                ax.annotate('', xy=(agent_col + 0.2, self.nrow - agent_row - 0.5),
                           xytext=(agent_col + 0.8, self.nrow - agent_row - 0.5), arrowprops=arrow_props)
            elif current_action == 1:  # DOWN
                ax.annotate('', xy=(agent_col + 0.5, self.nrow - agent_row - 0.8),
                           xytext=(agent_col + 0.5, self.nrow - agent_row - 0.2), arrowprops=arrow_props)
            elif current_action == 2:  # RIGHT
                ax.annotate('', xy=(agent_col + 0.8, self.nrow - agent_row - 0.5),
                           xytext=(agent_col + 0.2, self.nrow - agent_row - 0.5), arrowprops=arrow_props)
            elif current_action == 3:  # UP
                ax.annotate('', xy=(agent_col + 0.5, self.nrow - agent_row - 0.2),
                           xytext=(agent_col + 0.5, self.nrow - agent_row - 0.8), arrowprops=arrow_props)
        
        # Setup
        ax.set_xlim(0, self.ncol)
        ax.set_ylim(0, self.nrow)
        ax.set_aspect('equal')
        ax.set_title(f'Current State: {state}\nNext Action: {action_mapping[current_action]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    def seed(self, seed=None):
        np.random.seed(seed)
    
    def close(self):
        """Close the environment (for Gymnasium compatibility)"""
        plt.close('all')  # Close all matplotlib figures


def get_user_input_for_environment():
    """
    Get user input for creating a custom FrozenLake environment.
    
    Returns:
        tuple: (nrow, ncol, holes, goal, start_state)
    """
    print("\n🏒 Welcome to FrozenLake Environment Creator! 🏒")
    print("=" * 50)
    
    # Get grid dimensions
    while True:
        try:
            nrow = int(input("Enter number of rows (minimum 2): "))
            if nrow >= 2:
                break
            else:
                print("❌ Number of rows must be at least 2.")
        except ValueError:
            print("❌ Please enter a valid integer.")
    
    while True:
        try:
            ncol = int(input("Enter number of columns (minimum 2): "))
            if ncol >= 2:
                break
            else:
                print("❌ Number of columns must be at least 2.")
        except ValueError:
            print("❌ Please enter a valid integer.")
    
    print(f"\n📏 Grid size: {nrow} x {ncol}")
    print(f"📍 Valid positions: row (0-{nrow-1}), column (0-{ncol-1})")
    
    # Get start position
    print("\n🏁 Start Position:")
    while True:
        try:
            start_input = input(f"Enter start position as 'row,col' (default: 0,0): ").strip()
            if not start_input:
                start_state = (0, 0)
            else:
                start_row, start_col = map(int, start_input.split(','))
                start_state = (start_row, start_col)
            
            if 0 <= start_state[0] < nrow and 0 <= start_state[1] < ncol:
                break
            else:
                print(f"❌ Start position must be within grid bounds (0-{nrow-1}, 0-{ncol-1}).")
        except ValueError:
            print("❌ Please enter position as 'row,col' (e.g., '0,0').")
    
    # Get goal position
    print("\n🎯 Goal Position:")
    while True:
        try:
            goal_input = input(f"Enter goal position as 'row,col' (default: {nrow-1},{ncol-1}): ").strip()
            if not goal_input:
                goal = (nrow-1, ncol-1)
            else:
                goal_row, goal_col = map(int, goal_input.split(','))
                goal = (goal_row, goal_col)
            
            if 0 <= goal[0] < nrow and 0 <= goal[1] < ncol:
                if goal != start_state:
                    break
                else:
                    print("❌ Goal position cannot be the same as start position.")
            else:
                print(f"❌ Goal position must be within grid bounds (0-{nrow-1}, 0-{ncol-1}).")
        except ValueError:
            print("❌ Please enter position as 'row,col' (e.g., '4,4').")
    
    # Get holes
    print("\n🕳️  Hole Positions:")
    print("Enter hole positions one by one. Press Enter without input to finish.")
    holes = []
    hole_count = 1
    
    while True:
        try:
            hole_input = input(f"Enter hole #{hole_count} position as 'row,col' (or press Enter to finish): ").strip()
            if not hole_input:
                break
            
            hole_row, hole_col = map(int, hole_input.split(','))
            hole_pos = (hole_row, hole_col)
            
            # Validate hole position
            if not (0 <= hole_pos[0] < nrow and 0 <= hole_pos[1] < ncol):
                print(f"❌ Hole position must be within grid bounds (0-{nrow-1}, 0-{ncol-1}).")
                continue
            
            if hole_pos == start_state:
                print("❌ Hole position cannot be the same as start position.")
                continue
            
            if hole_pos == goal:
                print("❌ Hole position cannot be the same as goal position.")
                continue
            
            if hole_pos in holes:
                print("❌ This hole position already exists.")
                continue
            
            holes.append(hole_pos)
            print(f"✅ Added hole at {hole_pos}")
            hole_count += 1
            
        except ValueError:
            print("❌ Please enter position as 'row,col' (e.g., '2,3').")
    
    # Summary
    print("\n📋 Environment Summary:")
    print(f"  Grid Size: {nrow} x {ncol}")
    print(f"  Start: {start_state}")
    print(f"  Goal: {goal}")
    print(f"  Holes: {holes if holes else 'None'}")
    
    return nrow, ncol, holes, goal, start_state

def make_frozen_lake(nrow=None, ncol=None, holes=None, goal=None, start_state=None, interactive=False):
    """
    Factory function following Gymnasium philosophy with flexible parameters.
    
    Args:
        nrow (int): Number of rows (default: 5)
        ncol (int): Number of columns (default: 5)
        holes (list): List of (row, col) tuples for hole positions
        goal (tuple): (row, col) tuple for goal position
        start_state (tuple): (row, col) tuple for start position (default: (0, 0))
        interactive (bool): If True, get parameters from user input
    
    Usage: 
        env = make_frozen_lake()  # Default 5x5 with goal at (4,4)
        env = make_frozen_lake(nrow=3, ncol=4, holes=[(1,1), (2,2)], goal=(2,3))
        env = make_frozen_lake(interactive=True)  # User input
    """
    if interactive:
        nrow, ncol, holes, goal, start_state = get_user_input_for_environment()
    else:
        # Use defaults if not specified
        if nrow is None:
            nrow = 5
        if ncol is None:
            ncol = 5
        if start_state is None:
            start_state = (0, 0)
        if goal is None:
            goal = (nrow - 1, ncol - 1)
        if holes is None:
            holes = []
    
    return FrozenLakeEnv(nrow=nrow, ncol=ncol, holes=holes, goal=goal, start_state=start_state)

# Example usage following Gymnasium philosophy
if __name__ == "__main__":
    print("🎮 FrozenLake Environment Demo")
    print("Choose an option:")
    print("1. Default 5x5 environment")
    print("2. Custom 3x4 environment with holes")
    print("3. Interactive environment creator")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        # Default environment
        env = make_frozen_lake()
        print("\n🏒 Created default 5x5 FrozenLake")
    elif choice == "2":
        # Custom environment example
        env = make_frozen_lake(nrow=3, ncol=4, holes=[(1, 1), (1, 2)], goal=(2, 3), start_state=(0, 0))
        print("\n🏒 Created custom 3x4 FrozenLake with holes at (1,1) and (1,2)")
    elif choice == "3":
        # Interactive environment
        env = make_frozen_lake(interactive=True)
        print("\n🏒 Created interactive FrozenLake")
    else:
        print("Invalid choice, using default environment")
        env = make_frozen_lake()
    
    # Demo the environment
    print("\n🎲 Running random episode...")
    state, info = env.reset()
    env.render()
    terminated = False
    truncated = False
    step_count = 0
    
    while not (terminated or truncated) and step_count < 50:
        action = np.random.choice(env.action_space)
        state, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        print(f"Step {step_count}: Action {action}, State {state}, Reward {reward}")
        
        if terminated:
            if reward > 0:
                print("🎉 Reached the goal!")
            else:
                print("💀 Fell into a hole!")
        elif step_count >= 50:
            print("⏰ Maximum steps reached")
    
    env.render()
    env.close()
