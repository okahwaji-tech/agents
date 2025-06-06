"""
Value Iteration Algorithm Implementation
=======================================

This module implements the Value Iteration algorithm for solving Markov Decision Processes.
Value Iteration is a dynamic programming algorithm that iteratively updates state values
until convergence to the optimal value function.

Author: Manus AI
Date: June 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import time
from base_mdp import BaseMDP


class ValueIteration:
    """
    Value Iteration algorithm for solving MDPs.
    
    Value Iteration iteratively applies the Bellman optimality operator until
    convergence to compute the optimal value function and policy.
    """
    
    def __init__(self, mdp: BaseMDP, tolerance: float = 1e-6, max_iterations: int = 1000):
        """
        Initialize the Value Iteration algorithm.
        
        Args:
            mdp: The MDP environment to solve
            tolerance: Convergence tolerance for value function changes
            max_iterations: Maximum number of iterations
        """
        self.mdp = mdp
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Initialize value function
        self.values = torch.zeros(mdp.num_states, device=mdp.device, dtype=torch.float32)
        
        # Initialize policy (will be computed from values)
        self.policy = torch.zeros(mdp.num_states, device=mdp.device, dtype=torch.long)
        
        # Track convergence history
        self.value_history = []
        self.delta_history = []
        self.converged = False
        self.iterations_to_convergence = 0
    
    def bellman_update(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply the Bellman optimality operator to update values.
        
        V(s) = max_a [R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')]
        
        Args:
            values: Current value function
            
        Returns:
            Updated value function
        """
        new_values = torch.zeros_like(values)
        
        for state in range(self.mdp.num_states):
            action_values = torch.zeros(self.mdp.num_actions, device=self.mdp.device)
            
            for action in range(self.mdp.num_actions):
                # Compute Q(s,a) = R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')
                immediate_reward = self.mdp.expected_rewards[state, action]
                future_value = torch.sum(
                    self.mdp.transition_probs[state, action, :] * values
                )
                action_values[action] = immediate_reward + self.mdp.gamma * future_value
            
            # Take maximum over actions
            new_values[state] = torch.max(action_values)
        
        return new_values
    
    def extract_policy(self, values: torch.Tensor) -> torch.Tensor:
        """
        Extract the greedy policy from the value function.
        
        π(s) = argmax_a [R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')]
        
        Args:
            values: Value function
            
        Returns:
            Policy (action for each state)
        """
        policy = torch.zeros(self.mdp.num_states, device=self.mdp.device, dtype=torch.long)
        
        for state in range(self.mdp.num_states):
            action_values = torch.zeros(self.mdp.num_actions, device=self.mdp.device)
            
            for action in range(self.mdp.num_actions):
                immediate_reward = self.mdp.expected_rewards[state, action]
                future_value = torch.sum(
                    self.mdp.transition_probs[state, action, :] * values
                )
                action_values[action] = immediate_reward + self.mdp.gamma * future_value
            
            # Select action with highest value
            policy[state] = torch.argmax(action_values)
        
        return policy
    
    def solve(self, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the MDP using Value Iteration.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (optimal_values, optimal_policy)
        """
        if verbose:
            print("Starting Value Iteration...")
            print(f"Tolerance: {self.tolerance}")
            print(f"Max iterations: {self.max_iterations}")
            print("-" * 50)
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Store current values for convergence check
            old_values = self.values.clone()
            
            # Apply Bellman update
            self.values = self.bellman_update(self.values)
            
            # Compute change in value function
            delta = torch.max(torch.abs(self.values - old_values)).item()
            
            # Store history
            self.value_history.append(self.values.clone())
            self.delta_history.append(delta)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:4d}: Max value change = {delta:.6f}")
            
            # Check convergence
            if delta < self.tolerance:
                self.converged = True
                self.iterations_to_convergence = iteration + 1
                break
        
        # Extract optimal policy
        self.policy = self.extract_policy(self.values)
        
        end_time = time.time()
        
        if verbose:
            print("-" * 50)
            if self.converged:
                print(f"Converged after {self.iterations_to_convergence} iterations")
            else:
                print(f"Did not converge after {self.max_iterations} iterations")
            print(f"Final max value change: {self.delta_history[-1]:.6f}")
            print(f"Total time: {end_time - start_time:.3f} seconds")
        
        return self.values, self.policy
    
    def compute_q_values(self, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Q-values from the value function.
        
        Q(s,a) = R(s,a) + γ * Σ_s' P(s'|s,a) * V(s')
        
        Args:
            values: Value function (uses self.values if None)
            
        Returns:
            Q-values tensor of shape (num_states, num_actions)
        """
        if values is None:
            values = self.values
        
        q_values = torch.zeros(
            (self.mdp.num_states, self.mdp.num_actions),
            device=self.mdp.device,
            dtype=torch.float32
        )
        
        for state in range(self.mdp.num_states):
            for action in range(self.mdp.num_actions):
                immediate_reward = self.mdp.expected_rewards[state, action]
                future_value = torch.sum(
                    self.mdp.transition_probs[state, action, :] * values
                )
                q_values[state, action] = immediate_reward + self.mdp.gamma * future_value
        
        return q_values
    
    def evaluate_policy(self, policy: torch.Tensor, num_episodes: int = 1000) -> float:
        """
        Evaluate a policy by running episodes in the environment.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to run
            
        Returns:
            Average return over episodes
        """
        total_return = 0.0
        
        for episode in range(num_episodes):
            state = self.mdp.reset()
            episode_return = 0.0
            discount = 1.0
            
            for step in range(100):  # Max episode length
                action = policy[state].item()
                next_state, reward, done = self.mdp.step(action)
                
                episode_return += discount * reward
                discount *= self.mdp.gamma
                
                state = next_state
                
                if done:
                    break
            
            total_return += episode_return
        
        return total_return / num_episodes
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot the convergence of the value iteration algorithm.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.delta_history:
            print("No convergence history available. Run solve() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot value function changes
        iterations = range(1, len(self.delta_history) + 1)
        ax1.semilogy(iterations, self.delta_history, 'b-', linewidth=2)
        ax1.axhline(y=self.tolerance, color='r', linestyle='--', 
                   label=f'Tolerance ({self.tolerance})')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Max Value Change (log scale)')
        ax1.set_title('Value Iteration Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot value function evolution for a few states
        if len(self.value_history) > 0:
            num_states_to_plot = min(5, self.mdp.num_states)
            states_to_plot = np.linspace(0, self.mdp.num_states - 1, 
                                       num_states_to_plot, dtype=int)
            
            for state in states_to_plot:
                values_for_state = [v[state].item() for v in self.value_history]
                ax2.plot(iterations, values_for_state, 
                        label=f'State {state}', linewidth=2)
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('State Value')
            ax2.set_title('Value Function Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self):
        """
        Print a summary of the Value Iteration results.
        """
        print("=" * 60)
        print("Value Iteration Results")
        print("=" * 60)
        print(f"Converged: {self.converged}")
        print(f"Iterations to convergence: {self.iterations_to_convergence}")
        print(f"Final tolerance: {self.delta_history[-1] if self.delta_history else 'N/A':.6f}")
        print()
        
        print("Optimal Value Function:")
        for state in range(self.mdp.num_states):
            print(f"V*({self.mdp.state_names[state]}) = {self.values[state]:.4f}")
        print()
        
        print("Optimal Policy:")
        for state in range(self.mdp.num_states):
            action = self.policy[state].item()
            print(f"π*({self.mdp.state_names[state]}) = {self.mdp.action_names[action]}")
        print("=" * 60)


def demonstrate_value_iteration():
    """
    Demonstrate Value Iteration on a simple grid world problem.
    """
    print("Value Iteration Demonstration")
    print("=" * 50)
    
    # Import GridWorldMDP
    from base_mdp import GridWorldMDP
    
    # Create a 4x4 grid world with goal at bottom-right and obstacle
    env = GridWorldMDP(
        grid_size=4,
        goal_states=[15],
        obstacle_states=[5, 9],
        discount_factor=0.9
    )
    
    print("Environment Setup:")
    env.print_mdp_summary()
    
    # Visualize the environment
    print("\nGrid World Layout:")
    env.visualize_grid()
    
    # Solve using Value Iteration
    vi = ValueIteration(env, tolerance=1e-6, max_iterations=100)
    optimal_values, optimal_policy = vi.solve(verbose=True)
    
    # Print results
    vi.print_results()
    
    # Plot convergence
    vi.plot_convergence()
    
    # Visualize results
    print("\nOptimal Value Function and Policy:")
    env.visualize_grid(values=optimal_values, policy=optimal_policy)
    
    # Evaluate the optimal policy
    avg_return = vi.evaluate_policy(optimal_policy, num_episodes=1000)
    print(f"\nAverage return of optimal policy: {avg_return:.4f}")
    
    # Compare with Q-values
    q_values = vi.compute_q_values()
    print("\nQ-Values for each state-action pair:")
    for state in range(env.num_states):
        print(f"State {state}:")
        for action in range(env.num_actions):
            print(f"  Q({state}, {env.action_names[action]}) = {q_values[state, action]:.4f}")


if __name__ == "__main__":
    demonstrate_value_iteration()

