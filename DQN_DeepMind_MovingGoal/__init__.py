"""
DQN Agent for FrozenLake with Moving Goal

This module implements a Deep Q-Network (DQN) agent adapted to handle
dynamically changing goal positions in the FrozenLake environment.

Key Features:
- State encoding includes both agent position AND goal position
- Learns general navigation policy rather than memorizing a path
- Can generalize to reach any goal position
- Reuses DeepMind DQN neural network architecture

Usage:
    from DQN_DeepMind_MovingGoal import DQNAgentMovingGoal, make_frozen_lake_moving_goal
    
    # Create agent
    agent = DQNAgentMovingGoal()
    
    # Train on moving goal environment
    agent.train(num_episodes=1000)
    
    # Evaluate
    agent.evaluate(num_episodes=100)
    
    # Generate results
    agent.generate_results_folder()
"""

from .dqn_agent_moving_goal import DQNAgentMovingGoal
from .frozenlake_moving_goal_env import FrozenLakeMovingGoalEnv, make_frozen_lake_moving_goal

__all__ = [
    'DQNAgentMovingGoal',
    'FrozenLakeMovingGoalEnv',
    'make_frozen_lake_moving_goal'
]

__version__ = '1.0.0'
__author__ = 'Adapted from DQN_DeepMind'
