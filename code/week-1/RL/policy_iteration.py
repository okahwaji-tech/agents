"""
Policy Iteration Algorithm Implementation
========================================

This module implements the Policy Iteration algorithm for solving Markov Decision Processes.
Policy Iteration alternates between policy evaluation (computing the value function for a
fixed policy) and policy improvement (updating the policy to be greedy with respect to
the current value function).

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import time
from base_mdp import BaseMDP


class PolicyIteration:
    """
    Policy Iteration algorithm for solving MDPs.
    
    Policy Iteration alternates between:
    1. Policy Evaluation: Compute V^π for the current policy π
    2. Policy Improvement: Update π to be greedy with respect to V^π
    """
    
    def __init__(self, 
                 mdp: BaseMDP, 
                 eval_tolerance: float = 1e-6,
                 eval_max_iterations: int = 1000,
                 max_policy_iterations: int = 100):
        """
        Initialize the Policy Iteration algorithm.
        
        Args:
            mdp: The MDP environment to solve
            eval_tolerance: Convergence tolerance for policy evaluation
            eval_max_iterations: Maximum iterations for policy evaluation
            max_policy_iterations: Maximum number of policy iterations
        """
        self.mdp = mdp
        self.eval_tolerance = eval_tolerance
        self.eval_max_iterations = eval_max_iterations
        self.max_policy_iterations = max_policy_iterations
        
        # Initialize random policy
        self.policy = torch.randint(
            0, mdp.num_actions, (mdp.num_states,), 
            device=mdp.device, dtype=torch.long
        )
        
        # Initialize value function
        self.values = torch.zeros(mdp.num_states, device=mdp.device, dtype=torch.float32)
        
        # Track convergence history
        self.policy_history = []
        self.value_history = []
        self.policy_stable = False
        self.iterations_to_convergence = 0
    
    def policy_evaluation(self, 
                         policy: torch.Tensor, 
                         verbose: bool = False) -> torch.Tensor:
        """
        Evaluate a policy by computing its value function.
        
        Solves: V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
        
        For deterministic policies, this simplifies to:
        V^π(s) = R(s,π(s)) + γ Σ_s' P(s'|s,π(s)) V^π(s')
        
        Args:
            policy: Policy to evaluate (deterministic)
            verbose: Whether to print evaluation progress
            
        Returns:
            Value function for the given policy
        """
        values = torch.zeros(self.mdp.num_states, device=self.mdp.device, dtype=torch.float32)
        
        if verbose:
            print(f"  Policy Evaluation (tolerance: {self.eval_tolerance})")
        
        for iteration in range(self.eval_max_iterations):
            old_values = values.clone()
            
            # Update values using Bellman expectation equation
            for state in range(self.mdp.num_states):
                action = policy[state].item()
                
                # V^π(s) = R(s,π(s)) + γ Σ_s' P(s'|s,π(s)) V^π(s')
                immediate_reward = self.mdp.expected_rewards[state, action]
                future_value = torch.sum(
                    self.mdp.transition_probs[state, action, :] * values
                )
                values[state] = immediate_reward + self.mdp.gamma * future_value
            
            # Check convergence
            delta = torch.max(torch.abs(values - old_values)).item()
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"    Eval iteration {iteration + 1:4d}: Max change = {delta:.6f}")
            
            if delta < self.eval_tolerance:
                if verbose:
                    print(f"    Policy evaluation converged after {iteration + 1} iterations")
                break
        
        return values
    
    def policy_evaluation_linear_system(self, policy: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a policy by solving the linear system directly.
        
        Solves: (I - γP^π)V^π = R^π
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            Value function for the given policy
        """
        # Build transition matrix for the policy
        P_pi = torch.zeros(
            (self.mdp.num_states, self.mdp.num_states),
            device=self.mdp.device,
            dtype=torch.float32
        )
        
        # Build reward vector for the policy
        R_pi = torch.zeros(self.mdp.num_states, device=self.mdp.device, dtype=torch.float32)
        
        for state in range(self.mdp.num_states):
            action = policy[state].item()
            P_pi[state, :] = self.mdp.transition_probs[state, action, :]
            R_pi[state] = self.mdp.expected_rewards[state, action]
        
        # Solve (I - γP^π)V^π = R^π
        I = torch.eye(self.mdp.num_states, device=self.mdp.device, dtype=torch.float32)
        A = I - self.mdp.gamma * P_pi
        
        try:
            values = torch.linalg.solve(A, R_pi)
        except torch.linalg.LinAlgError:
            print("Warning: Linear system is singular. Using iterative evaluation.")
            values = self.policy_evaluation(policy)
        
        return values
    
    def policy_improvement(self, values: torch.Tensor) -> torch.Tensor:
        """
        Improve the policy by making it greedy with respect to the value function.
        
        π'(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
        
        Args:
            values: Current value function
            
        Returns:
            Improved policy
        """
        new_policy = torch.zeros(
            self.mdp.num_states, 
            device=self.mdp.device, 
            dtype=torch.long
        )
        
        for state in range(self.mdp.num_states):
            action_values = torch.zeros(self.mdp.num_actions, device=self.mdp.device)
            
            for action in range(self.mdp.num_actions):
                immediate_reward = self.mdp.expected_rewards[state, action]
                future_value = torch.sum(
                    self.mdp.transition_probs[state, action, :] * values
                )
                action_values[action] = immediate_reward + self.mdp.gamma * future_value
            
            # Select action with highest value
            new_policy[state] = torch.argmax(action_values)
        
        return new_policy
    
    def solve(self, 
              use_linear_system: bool = False,
              verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the MDP using Policy Iteration.
        
        Args:
            use_linear_system: Whether to use direct linear system solving for evaluation
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (optimal_values, optimal_policy)
        """
        if verbose:
            print("Starting Policy Iteration...")
            print(f"Evaluation tolerance: {self.eval_tolerance}")
            print(f"Max policy iterations: {self.max_policy_iterations}")
            print(f"Using linear system: {use_linear_system}")
            print("-" * 60)
        
        start_time = time.time()
        
        for iteration in range(self.max_policy_iterations):
            if verbose:
                print(f"Policy Iteration {iteration + 1}")
            
            # Store current policy
            old_policy = self.policy.clone()
            self.policy_history.append(old_policy.clone())
            
            # Policy Evaluation
            if use_linear_system:
                self.values = self.policy_evaluation_linear_system(self.policy)
            else:
                self.values = self.policy_evaluation(self.policy, verbose=verbose)
            
            self.value_history.append(self.values.clone())
            
            # Policy Improvement
            self.policy = self.policy_improvement(self.values)
            
            if verbose:
                print(f"  Policy changes: {torch.sum(self.policy != old_policy).item()}")
            
            # Check if policy is stable
            if torch.equal(self.policy, old_policy):
                self.policy_stable = True
                self.iterations_to_convergence = iteration + 1
                if verbose:
                    print(f"  Policy is stable!")
                break
            
            if verbose:
                print()
        
        end_time = time.time()
        
        if verbose:
            print("-" * 60)
            if self.policy_stable:
                print(f"Converged after {self.iterations_to_convergence} policy iterations")
            else:
                print(f"Did not converge after {self.max_policy_iterations} iterations")
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
        Plot the convergence of the policy iteration algorithm.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.value_history:
            print("No convergence history available. Run solve() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot policy changes over iterations
        policy_changes = []
        if len(self.policy_history) > 1:
            for i in range(1, len(self.policy_history)):
                changes = torch.sum(
                    self.policy_history[i] != self.policy_history[i-1]
                ).item()
                policy_changes.append(changes)
        
        if policy_changes:
            iterations = range(1, len(policy_changes) + 1)
            ax1.bar(iterations, policy_changes, alpha=0.7, color='skyblue')
            ax1.set_xlabel('Policy Iteration')
            ax1.set_ylabel('Number of Policy Changes')
            ax1.set_title('Policy Changes per Iteration')
            ax1.grid(True, alpha=0.3)
        
        # Plot value function evolution for a few states
        if len(self.value_history) > 0:
            num_states_to_plot = min(5, self.mdp.num_states)
            states_to_plot = np.linspace(0, self.mdp.num_states - 1, 
                                       num_states_to_plot, dtype=int)
            
            iterations = range(len(self.value_history))
            for state in states_to_plot:
                values_for_state = [v[state].item() for v in self.value_history]
                ax2.plot(iterations, values_for_state, 
                        label=f'State {state}', linewidth=2, marker='o')
            
            ax2.set_xlabel('Policy Iteration')
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
        Print a summary of the Policy Iteration results.
        """
        print("=" * 60)
        print("Policy Iteration Results")
        print("=" * 60)
        print(f"Policy stable: {self.policy_stable}")
        print(f"Iterations to convergence: {self.iterations_to_convergence}")
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


def demonstrate_policy_iteration():
    """
    Demonstrate Policy Iteration on a simple grid world problem.
    """
    print("Policy Iteration Demonstration")
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
    
    # Solve using Policy Iteration (both methods)
    print("\n" + "="*60)
    print("Method 1: Iterative Policy Evaluation")
    print("="*60)
    
    pi_iterative = PolicyIteration(env, eval_tolerance=1e-6)
    optimal_values_iter, optimal_policy_iter = pi_iterative.solve(
        use_linear_system=False, verbose=True
    )
    
    print("\n" + "="*60)
    print("Method 2: Linear System Policy Evaluation")
    print("="*60)
    
    pi_linear = PolicyIteration(env, eval_tolerance=1e-6)
    optimal_values_linear, optimal_policy_linear = pi_linear.solve(
        use_linear_system=True, verbose=True
    )
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)
    
    value_diff = torch.max(torch.abs(optimal_values_iter - optimal_values_linear)).item()
    policy_diff = torch.sum(optimal_policy_iter != optimal_policy_linear).item()
    
    print(f"Max value function difference: {value_diff:.8f}")
    print(f"Policy differences: {policy_diff}")
    
    # Print results
    pi_iterative.print_results()
    
    # Plot convergence
    pi_iterative.plot_convergence()
    
    # Visualize results
    print("\nOptimal Value Function and Policy:")
    env.visualize_grid(values=optimal_values_iter, policy=optimal_policy_iter)
    
    # Evaluate the optimal policy
    avg_return = pi_iterative.evaluate_policy(optimal_policy_iter, num_episodes=1000)
    print(f"\nAverage return of optimal policy: {avg_return:.4f}")


def compare_algorithms():
    """
    Compare Policy Iteration with Value Iteration on the same problem.
    """
    print("\n" + "="*60)
    print("Comparing Policy Iteration vs Value Iteration")
    print("="*60)
    
    from base_mdp import GridWorldMDP
    from value_iteration import ValueIteration
    
    # Create environment
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    # Solve with Policy Iteration
    print("Solving with Policy Iteration...")
    pi = PolicyIteration(env)
    start_time = time.time()
    pi_values, pi_policy = pi.solve(verbose=False)
    pi_time = time.time() - start_time
    
    # Solve with Value Iteration
    print("Solving with Value Iteration...")
    vi = ValueIteration(env)
    start_time = time.time()
    vi_values, vi_policy = vi.solve(verbose=False)
    vi_time = time.time() - start_time
    
    # Compare results
    print("\nComparison Results:")
    print(f"Policy Iteration: {pi.iterations_to_convergence} iterations, {pi_time:.4f}s")
    print(f"Value Iteration: {vi.iterations_to_convergence} iterations, {vi_time:.4f}s")
    
    value_diff = torch.max(torch.abs(pi_values - vi_values)).item()
    policy_diff = torch.sum(pi_policy != vi_policy).item()
    
    print(f"Max value difference: {value_diff:.8f}")
    print(f"Policy differences: {policy_diff}")
    
    if policy_diff == 0:
        print("Both algorithms found the same optimal policy!")
    else:
        print("Algorithms found different policies (may both be optimal)")


if __name__ == "__main__":
    demonstrate_policy_iteration()
    compare_algorithms()

