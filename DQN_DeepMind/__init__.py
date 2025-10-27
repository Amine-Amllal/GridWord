"""
DeepMind DQN Agent for FrozenLake Environment

This module implements the Deep Q-Network (DQN) algorithm as introduced
by DeepMind in their 2015 Nature paper.

Key Features:
- Experience Replay
- Target Network
- Epsilon-greedy Exploration
- Neural Network Q-value Approximation

Usage:
    from DQN_DeepMind import DQNAgent
    
    agent = DQNAgent()
    agent.train(num_episodes=1000)
    agent.evaluate(num_episodes=100)
    agent.generate_results_folder()
"""

from .dqn_agent import DQNAgent, DeepMindDQN

__all__ = ['DQNAgent', 'DeepMindDQN']
__version__ = '1.0.0'
__author__ = 'Your Name'
