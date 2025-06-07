"""
Base MDP Environment Implementation
==================================

This module provides a foundational MDP environment class that can be used
to implement various Markov Decision Process algorithms. The environment
supports both discrete and continuous state/action spaces and provides
methods for defining transition probabilities and reward functions.

"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns


class BaseMDP(ABC):
    """
    Abstract base class for Markov Decision Process environments.
    
    This class provides the fundamental structure for MDP environments,
    including state and action spaces, transition dynamics, and reward functions.
    Concrete implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, 
                 num_states: int,
                 num_actions: int,
                 discount_factor: float = 0.95,
                 device: str = 'cpu'):
        """
        Initialize the MDP environment.
        
        Args:
            num_states: Number of states in the MDP
            num_actions: Number of actions available in each state
            discount_factor: Discount factor gamma (0 < gamma <= 1)
            device: PyTorch device ('cpu' or 'cuda')
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.device = torch.device(device)
        
        # Initialize transition probabilities P(s'|s,a)
        # Shape: (num_states, num_actions, num_states)
        self.transition_probs = torch.zeros(
            (num_states, num_actions, num_states),
            device=self.device,
            dtype=torch.float32
        )
        
        # Initialize reward function R(s,a,s')
        # Shape: (num_states, num_actions, num_states)
        self.rewards = torch.zeros(
            (num_states, num_actions, num_states),
            device=self.device,
            dtype=torch.float32
        )
        
        # Expected immediate rewards R(s,a)
        # Shape: (num_states, num_actions)
        self.expected_rewards = torch.zeros(
            (num_states, num_actions),
            device=self.device,
            dtype=torch.float32
        )
        
        # Current state
        self.current_state = 0
        
        # State and action names for visualization
        self.state_names = [f"State_{i}" for i in range(num_states)]
        self.action_names = [f"Action_{i}" for i in range(num_actions)]
    
    @abstractmethod
    def setup_environment(self):
        """
        Setup the specific MDP environment by defining transition probabilities
        and reward functions. This method must be implemented by subclasses.
        """
        pass
    
    def set_transition_probability(self, 
                                 state: int, 
                                 action: int, 
                                 next_state: int, 
                                 probability: float):
        """
        Set the transition probability P(next_state | state, action).
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            probability: Transition probability
        """
        self.transition_probs[state, action, next_state] = probability
    
    def set_reward(self, 
                   state: int, 
                   action: int, 
                   next_state: int, 
                   reward: float):
        """
        Set the reward R(state, action, next_state).
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward value
        """
        self.rewards[state, action, next_state] = reward
    
    def compute_expected_rewards(self):
        """
        Compute expected immediate rewards R(s,a) = sum_s' P(s'|s,a) * R(s,a,s').
        """
        self.expected_rewards = torch.sum(
            self.transition_probs * self.rewards, dim=2
        )
    
    def get_transition_probabilities(self, state: int, action: int) -> torch.Tensor:
        """
        Get transition probabilities for a given state-action pair.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Tensor of transition probabilities to all next states
        """
        return self.transition_probs[state, action, :]
    
    def get_expected_reward(self, state: int, action: int) -> float:
        """
        Get expected immediate reward for a given state-action pair.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Expected immediate reward
        """
        return self.expected_rewards[state, action].item()
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take an action in the environment and return the result.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Sample next state according to transition probabilities
        probs = self.get_transition_probabilities(self.current_state, action)
        next_state = torch.multinomial(probs, 1).item()
        
        # Get reward
        reward = self.rewards[self.current_state, action, next_state].item()
        
        # Update current state
        self.current_state = next_state
        
        # Check if episode is done (can be overridden by subclasses)
        done = self.is_terminal(next_state)
        
        return next_state, reward, done
    
    def reset(self, initial_state: Optional[int] = None) -> int:
        """
        Reset the environment to an initial state.
        
        Args:
            initial_state: Specific initial state (random if None)
            
        Returns:
            Initial state
        """
        if initial_state is None:
            self.current_state = np.random.randint(0, self.num_states)
        else:
            self.current_state = initial_state
        return self.current_state
    
    def is_terminal(self, state: int) -> bool:
        """
        Check if a state is terminal. Default implementation returns False.
        Override in subclasses for episodic tasks.
        
        Args:
            state: State to check
            
        Returns:
            True if state is terminal, False otherwise
        """
        return False
    
    def validate_mdp(self) -> bool:
        """
        Validate that the MDP is properly defined (transition probabilities sum to 1).
        
        Returns:
            True if MDP is valid, False otherwise
        """
        # Check that transition probabilities sum to 1 for each (state, action) pair
        prob_sums = torch.sum(self.transition_probs, dim=2)
        tolerance = 1e-6
        
        valid = torch.all(torch.abs(prob_sums - 1.0) < tolerance)
        
        if not valid:
            print("Warning: Transition probabilities do not sum to 1 for all (state, action) pairs")
            print("Probability sums:")
            print(prob_sums)
        
        return valid.item()
    
    def visualize_transition_matrix(self, action: int, save_path: Optional[str] = None):
        """
        Visualize the transition probability matrix for a specific action.
        
        Args:
            action: Action to visualize
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        # Extract transition matrix for the specified action
        transition_matrix = self.transition_probs[:, action, :].cpu().numpy()
        
        # Create heatmap
        sns.heatmap(
            transition_matrix,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=self.state_names,
            yticklabels=self.state_names,
            cbar_kws={'label': 'Transition Probability'}
        )
        
        plt.title(f'Transition Probabilities for {self.action_names[action]}')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_reward_matrix(self, save_path: Optional[str] = None):
        """
        Visualize the expected reward matrix R(s,a).
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap of expected rewards
        sns.heatmap(
            self.expected_rewards.cpu().numpy(),
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.action_names,
            yticklabels=self.state_names,
            cbar_kws={'label': 'Expected Reward'}
        )
        
        plt.title('Expected Rewards R(s,a)')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_state_action_pairs(self) -> List[Tuple[int, int]]:
        """
        Get all valid state-action pairs.
        
        Returns:
            List of (state, action) tuples
        """
        pairs = []
        for s in range(self.num_states):
            for a in range(self.num_actions):
                pairs.append((s, a))
        return pairs
    
    def print_mdp_summary(self):
        """
        Print a summary of the MDP structure.
        """
        print("=" * 50)
        print("MDP Environment Summary")
        print("=" * 50)
        print(f"Number of states: {self.num_states}")
        print(f"Number of actions: {self.num_actions}")
        print(f"Discount factor (γ): {self.gamma}")
        print(f"Device: {self.device}")
        print(f"State names: {self.state_names}")
        print(f"Action names: {self.action_names}")
        print(f"MDP is valid: {self.validate_mdp()}")
        print("=" * 50)


class GridWorldMDP(BaseMDP):
    """
    A simple grid world MDP implementation for testing and demonstration.
    
    The agent moves in a grid and receives rewards for reaching certain states.
    This serves as a concrete example of how to implement the BaseMDP class.
    """
    
    def __init__(self, 
                 grid_size: int = 4,
                 goal_states: List[int] = None,
                 obstacle_states: List[int] = None,
                 discount_factor: float = 0.95,
                 device: str = 'cpu'):
        """
        Initialize a grid world MDP.
        
        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            goal_states: List of goal state indices
            obstacle_states: List of obstacle state indices
            discount_factor: Discount factor gamma
            device: PyTorch device
        """
        num_states = grid_size * grid_size
        num_actions = 4  # Up, Down, Left, Right
        
        super().__init__(num_states, num_actions, discount_factor, device)
        
        self.grid_size = grid_size
        self.goal_states = goal_states or [num_states - 1]  # Default: bottom-right corner
        self.obstacle_states = obstacle_states or []
        
        # Action mappings
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        self.state_names = [f"({i//grid_size},{i%grid_size})" for i in range(num_states)]
        
        self.setup_environment()
    
    def setup_environment(self):
        """
        Setup the grid world environment with transition probabilities and rewards.
        """
        # Define action effects: [row_change, col_change]
        action_effects = {
            0: [-1, 0],  # Up
            1: [1, 0],   # Down
            2: [0, -1],  # Left
            3: [0, 1]    # Right
        }
        
        for state in range(self.num_states):
            row, col = state // self.grid_size, state % self.grid_size
            
            for action in range(self.num_actions):
                # Calculate intended next position
                d_row, d_col = action_effects[action]
                new_row, new_col = row + d_row, col + d_col
                
                # Check boundaries and obstacles
                if (0 <= new_row < self.grid_size and 
                    0 <= new_col < self.grid_size):
                    next_state = new_row * self.grid_size + new_col
                    
                    if next_state not in self.obstacle_states:
                        # Valid move
                        self.set_transition_probability(state, action, next_state, 1.0)
                        
                        # Set rewards
                        if next_state in self.goal_states:
                            self.set_reward(state, action, next_state, 10.0)
                        elif next_state in self.obstacle_states:
                            self.set_reward(state, action, next_state, -10.0)
                        else:
                            self.set_reward(state, action, next_state, -0.1)  # Small step cost
                    else:
                        # Hit obstacle, stay in current state
                        self.set_transition_probability(state, action, state, 1.0)
                        self.set_reward(state, action, state, -1.0)
                else:
                    # Hit boundary, stay in current state
                    self.set_transition_probability(state, action, state, 1.0)
                    self.set_reward(state, action, state, -1.0)
        
        # Compute expected rewards
        self.compute_expected_rewards()
    
    def is_terminal(self, state: int) -> bool:
        """
        Check if a state is terminal (goal state).
        
        Args:
            state: State to check
            
        Returns:
            True if state is a goal state, False otherwise
        """
        return state in self.goal_states
    
    def visualize_grid(self, values: Optional[torch.Tensor] = None, 
                      policy: Optional[torch.Tensor] = None,
                      save_path: Optional[str] = None):
        """
        Visualize the grid world with optional value function or policy.
        
        Args:
            values: State values to display
            policy: Policy to display (action indices)
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create grid
        grid = np.zeros((self.grid_size, self.grid_size))
        
        if values is not None:
            # Display values
            values_np = values.cpu().numpy()
            for state in range(self.num_states):
                row, col = state // self.grid_size, state % self.grid_size
                grid[row, col] = values_np[state]
        
        # Create heatmap
        im = ax.imshow(grid, cmap='RdYlGn', aspect='equal')
        
        # Add text annotations
        for state in range(self.num_states):
            row, col = state // self.grid_size, state % self.grid_size
            
            # State value
            if values is not None:
                ax.text(col, row, f'{values[state]:.2f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Policy arrow
            if policy is not None:
                action = policy[state].item()
                arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
                ax.text(col, row + 0.3, arrow_map[action], 
                       ha='center', va='center', fontsize=16, color='blue')
            
            # Mark special states
            if state in self.goal_states:
                ax.add_patch(plt.Circle((col, row), 0.4, color='gold', alpha=0.7))
                ax.text(col, row - 0.3, 'GOAL', ha='center', va='center', 
                       fontsize=8, fontweight='bold')
            elif state in self.obstacle_states:
                ax.add_patch(plt.Rectangle((col-0.4, row-0.4), 0.8, 0.8, 
                                         color='red', alpha=0.7))
                ax.text(col, row, 'X', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='white')
        
        # Customize plot
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, alpha=0.3)
        ax.set_title('Grid World MDP')
        
        # Add colorbar for values
        if values is not None:
            plt.colorbar(im, ax=ax, label='State Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing Grid World MDP Implementation")
    
    # Create a simple 4x4 grid world
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    # Print summary
    env.print_mdp_summary()
    
    # Visualize the grid
    env.visualize_grid()
    
    # Test environment interaction
    print("\nTesting environment interaction:")
    state = env.reset(0)
    print(f"Initial state: {state}")
    
    for step in range(5):
        action = np.random.randint(0, 4)
        next_state, reward, done = env.step(action)
        print(f"Step {step+1}: Action={env.action_names[action]}, "
              f"Next State={next_state}, Reward={reward:.2f}, Done={done}")
        if done:
            break

