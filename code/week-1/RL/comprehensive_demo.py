"""
Comprehensive MDP Demonstration
==============================

This script provides a comprehensive demonstration of all MDP algorithms
and examples implemented in this package. It showcases the theoretical
concepts from the enhanced MDP guide through practical PyTorch implementations.

Author: Manus AI
Date: June 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

# Import all our implementations
from base_mdp import BaseMDP, GridWorldMDP
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from q_learning import QLearning
from sarsa import SARSA
from healthcare_examples import DiabetesTreatmentMDP, SepsisManagementMDP


def run_comprehensive_demo():
    """
    Run a comprehensive demonstration of all MDP concepts and algorithms.
    """
    print("=" * 80)
    print("COMPREHENSIVE MDP DEMONSTRATION")
    print("Implementing Concepts from the Enhanced MDP Guide")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Basic MDP Environment Demo
    print("\n" + "="*60)
    print("1. BASIC MDP ENVIRONMENT DEMONSTRATION")
    print("="*60)
    
    basic_demo()
    
    # 2. Dynamic Programming Algorithms
    print("\n" + "="*60)
    print("2. DYNAMIC PROGRAMMING ALGORITHMS")
    print("="*60)
    
    dynamic_programming_demo()
    
    # 3. Reinforcement Learning Algorithms
    print("\n" + "="*60)
    print("3. REINFORCEMENT LEARNING ALGORITHMS")
    print("="*60)
    
    reinforcement_learning_demo()
    
    # 4. Healthcare Applications
    print("\n" + "="*60)
    print("4. HEALTHCARE APPLICATIONS")
    print("="*60)
    
    healthcare_demo()
    
    # 5. Algorithm Comparison
    print("\n" + "="*60)
    print("5. COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*60)
    
    algorithm_comparison_demo()
    
    # 6. Convergence Analysis
    print("\n" + "="*60)
    print("6. CONVERGENCE AND PERFORMANCE ANALYSIS")
    print("="*60)
    
    convergence_analysis_demo()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


def basic_demo():
    """
    Demonstrate basic MDP environment setup and validation.
    """
    print("Creating a 4x4 Grid World MDP...")
    
    # Create grid world environment
    env = GridWorldMDP(
        grid_size=4,
        goal_states=[15],
        obstacle_states=[5, 9],
        discount_factor=0.9
    )
    
    # Print environment summary
    env.print_mdp_summary()
    
    # Validate MDP properties
    print(f"MDP is valid: {env.validate_mdp()}")
    
    # Visualize environment
    print("\nVisualizing Grid World Environment:")
    env.visualize_grid()
    
    # Show transition matrix for one action
    print("\nTransition probabilities for 'Right' action:")
    env.visualize_transition_matrix(action=3)
    
    # Show reward matrix
    print("\nExpected rewards for all state-action pairs:")
    env.visualize_reward_matrix()
    
    # Test environment interaction
    print("\nTesting environment interaction:")
    state = env.reset(0)
    print(f"Initial state: {state} ({env.state_names[state]})")
    
    for step in range(5):
        action = np.random.randint(0, 4)
        next_state, reward, done = env.step(action)
        print(f"Step {step+1}: Action={env.action_names[action]}, "
              f"Next State={next_state} ({env.state_names[next_state]}), "
              f"Reward={reward:.2f}, Done={done}")
        if done:
            break


def dynamic_programming_demo():
    """
    Demonstrate dynamic programming algorithms (Value Iteration and Policy Iteration).
    """
    print("Setting up Grid World for Dynamic Programming...")
    
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    # Value Iteration
    print("\n1. VALUE ITERATION")
    print("-" * 30)
    
    vi = ValueIteration(env, tolerance=1e-6, max_iterations=100)
    start_time = time.time()
    vi_values, vi_policy = vi.solve(verbose=True)
    vi_time = time.time() - start_time
    
    print(f"\nValue Iteration completed in {vi_time:.4f} seconds")
    vi.print_results()
    
    # Policy Iteration
    print("\n2. POLICY ITERATION")
    print("-" * 30)
    
    pi = PolicyIteration(env, eval_tolerance=1e-6, max_policy_iterations=50)
    start_time = time.time()
    pi_values, pi_policy = pi.solve(verbose=True)
    pi_time = time.time() - start_time
    
    print(f"\nPolicy Iteration completed in {pi_time:.4f} seconds")
    pi.print_results()
    
    # Compare results
    print("\n3. COMPARISON OF DYNAMIC PROGRAMMING METHODS")
    print("-" * 50)
    
    value_diff = torch.max(torch.abs(vi_values - pi_values)).item()
    policy_diff = torch.sum(vi_policy != pi_policy).item()
    
    print(f"Maximum value difference: {value_diff:.8f}")
    print(f"Policy differences: {policy_diff} states")
    print(f"Value Iteration time: {vi_time:.4f}s")
    print(f"Policy Iteration time: {pi_time:.4f}s")
    
    if policy_diff == 0:
        print("✓ Both algorithms found the same optimal policy!")
    
    # Visualize results
    print("\nVisualizing optimal policy and values:")
    env.visualize_grid(values=vi_values, policy=vi_policy)
    
    # Plot convergence
    vi.plot_convergence()
    pi.plot_convergence()


def reinforcement_learning_demo():
    """
    Demonstrate reinforcement learning algorithms (Q-Learning and SARSA).
    """
    print("Setting up Grid World for Reinforcement Learning...")
    
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    # Q-Learning
    print("\n1. Q-LEARNING")
    print("-" * 20)
    
    ql_agent = QLearning(
        mdp=env,
        learning_rate=0.1,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        exploration_strategy='epsilon_greedy'
    )
    
    start_time = time.time()
    ql_agent.train(num_episodes=1000, verbose=True, log_interval=200)
    ql_time = time.time() - start_time
    
    ql_policy = ql_agent.extract_policy()
    ql_return, ql_length = ql_agent.evaluate_policy(num_episodes=100)
    
    print(f"\nQ-Learning completed in {ql_time:.4f} seconds")
    print(f"Final performance: Return={ql_return:.4f}, Length={ql_length:.2f}")
    
    # SARSA
    print("\n2. SARSA")
    print("-" * 10)
    
    sarsa_agent = SARSA(
        mdp=env,
        learning_rate=0.1,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        exploration_strategy='epsilon_greedy'
    )
    
    start_time = time.time()
    sarsa_agent.train(num_episodes=1000, verbose=True, log_interval=200)
    sarsa_time = time.time() - start_time
    
    sarsa_policy = sarsa_agent.extract_policy()
    sarsa_return, sarsa_length = sarsa_agent.evaluate_policy(num_episodes=100, use_learned_policy=False)
    
    print(f"\nSARSA completed in {sarsa_time:.4f} seconds")
    print(f"Final performance: Return={sarsa_return:.4f}, Length={sarsa_length:.2f}")
    
    # Compare RL algorithms
    print("\n3. REINFORCEMENT LEARNING COMPARISON")
    print("-" * 40)
    
    ql_sarsa_policy_diff = torch.sum(ql_policy != sarsa_policy).item()
    
    print(f"Q-Learning vs SARSA policy differences: {ql_sarsa_policy_diff} states")
    print(f"Q-Learning performance: {ql_return:.4f}")
    print(f"SARSA performance: {sarsa_return:.4f}")
    
    # Visualize learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Learning curves
    window_size = 50
    ql_ma = np.convolve(ql_agent.episode_returns, np.ones(window_size)/window_size, mode='valid')
    sarsa_ma = np.convolve(sarsa_agent.episode_returns, np.ones(window_size)/window_size, mode='valid')
    
    ax1.plot(range(window_size-1, len(ql_agent.episode_returns)), ql_ma, 'b-', linewidth=2, label='Q-Learning')
    ax1.plot(range(window_size-1, len(sarsa_agent.episode_returns)), sarsa_ma, 'r-', linewidth=2, label='SARSA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Learning Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final performance
    algorithms = ['Q-Learning', 'SARSA']
    returns = [ql_return, sarsa_return]
    
    ax2.bar(algorithms, returns, alpha=0.7, color=['blue', 'red'])
    ax2.set_ylabel('Average Return')
    ax2.set_title('Final Performance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize Q-tables
    ql_agent.visualize_q_table()
    sarsa_agent.visualize_q_table()


def healthcare_demo():
    """
    Demonstrate healthcare-specific MDP applications.
    """
    print("Demonstrating Healthcare MDP Applications...")
    
    # Diabetes Treatment MDP
    print("\n1. DIABETES TREATMENT MDP")
    print("-" * 30)
    
    diabetes_mdp = DiabetesTreatmentMDP(discount_factor=0.95)
    diabetes_mdp.print_mdp_summary()
    
    # Solve with Value Iteration
    vi_diabetes = ValueIteration(diabetes_mdp, tolerance=1e-6)
    diabetes_values, diabetes_policy = vi_diabetes.solve(verbose=False)
    
    print("\nOptimal Diabetes Treatment Policy:")
    clinical_interp = diabetes_mdp.get_clinical_interpretation(diabetes_policy)
    for state_name, interpretation in clinical_interp.items():
        print(f"  {state_name}:")
        print(f"    {interpretation}")
    
    # Sepsis Management MDP
    print("\n2. SEPSIS MANAGEMENT MDP")
    print("-" * 25)
    
    sepsis_mdp = SepsisManagementMDP(discount_factor=0.9)
    sepsis_mdp.print_mdp_summary()
    
    # Solve with Policy Iteration
    pi_sepsis = PolicyIteration(sepsis_mdp, eval_tolerance=1e-6)
    sepsis_values, sepsis_policy = pi_sepsis.solve(verbose=False)
    
    print("\nOptimal Sepsis Management Policy:")
    for state in range(sepsis_mdp.num_states):
        action = sepsis_policy[state].item()
        print(f"  {sepsis_mdp.state_names[state]}: {sepsis_mdp.action_names[action]}")
    
    # Train RL agents on healthcare MDPs
    print("\n3. REINFORCEMENT LEARNING FOR HEALTHCARE")
    print("-" * 40)
    
    print("Training Q-Learning on Diabetes MDP...")
    ql_diabetes = QLearning(diabetes_mdp, learning_rate=0.1, epsilon=0.9)
    ql_diabetes.train(num_episodes=1500, verbose=False)
    
    print("Training SARSA on Sepsis MDP...")
    sarsa_sepsis = SARSA(sepsis_mdp, learning_rate=0.1, epsilon=0.9)
    sarsa_sepsis.train(num_episodes=1500, verbose=False)
    
    # Evaluate healthcare policies
    diabetes_vi_return = vi_diabetes.evaluate_policy(diabetes_policy, num_episodes=500)
    diabetes_ql_return, _ = ql_diabetes.evaluate_policy(num_episodes=500)
    
    sepsis_pi_return = pi_sepsis.evaluate_policy(sepsis_policy, num_episodes=500)
    sepsis_sarsa_return, _ = sarsa_sepsis.evaluate_policy(num_episodes=500, use_learned_policy=False)
    
    print(f"\nHealthcare MDP Performance:")
    print(f"Diabetes - Value Iteration: {diabetes_vi_return:.4f}")
    print(f"Diabetes - Q-Learning: {diabetes_ql_return:.4f}")
    print(f"Sepsis - Policy Iteration: {sepsis_pi_return:.4f}")
    print(f"Sepsis - SARSA: {sepsis_sarsa_return:.4f}")


def algorithm_comparison_demo():
    """
    Comprehensive comparison of all algorithms on the same problem.
    """
    print("Comprehensive Algorithm Comparison on Grid World...")
    
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    results = {}
    
    # Value Iteration
    print("\nRunning Value Iteration...")
    vi = ValueIteration(env, tolerance=1e-6)
    start_time = time.time()
    vi_values, vi_policy = vi.solve(verbose=False)
    vi_time = time.time() - start_time
    vi_return = vi.evaluate_policy(vi_policy, num_episodes=1000)
    
    results['Value Iteration'] = {
        'time': vi_time,
        'return': vi_return,
        'iterations': vi.iterations_to_convergence,
        'policy': vi_policy
    }
    
    # Policy Iteration
    print("Running Policy Iteration...")
    pi = PolicyIteration(env, eval_tolerance=1e-6)
    start_time = time.time()
    pi_values, pi_policy = pi.solve(verbose=False)
    pi_time = time.time() - start_time
    pi_return = pi.evaluate_policy(pi_policy, num_episodes=1000)
    
    results['Policy Iteration'] = {
        'time': pi_time,
        'return': pi_return,
        'iterations': pi.iterations_to_convergence,
        'policy': pi_policy
    }
    
    # Q-Learning
    print("Running Q-Learning...")
    ql = QLearning(env, learning_rate=0.1, epsilon=0.9, epsilon_decay=0.995)
    start_time = time.time()
    ql.train(num_episodes=1000, verbose=False)
    ql_time = time.time() - start_time
    ql_policy = ql.extract_policy()
    ql_return, _ = ql.evaluate_policy(num_episodes=1000)
    
    results['Q-Learning'] = {
        'time': ql_time,
        'return': ql_return,
        'iterations': len(ql.episode_returns),
        'policy': ql_policy
    }
    
    # SARSA
    print("Running SARSA...")
    sarsa = SARSA(env, learning_rate=0.1, epsilon=0.9, epsilon_decay=0.995)
    start_time = time.time()
    sarsa.train(num_episodes=1000, verbose=False)
    sarsa_time = time.time() - start_time
    sarsa_policy = sarsa.extract_policy()
    sarsa_return, _ = sarsa.evaluate_policy(num_episodes=1000, use_learned_policy=False)
    
    results['SARSA'] = {
        'time': sarsa_time,
        'return': sarsa_return,
        'iterations': len(sarsa.episode_returns),
        'policy': sarsa_policy
    }
    
    # Print comparison table
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*80)
    print(f"{'Algorithm':<18} {'Time (s)':<10} {'Return':<10} {'Iterations':<12} {'Policy Match':<12}")
    print("-"*80)
    
    baseline_policy = results['Value Iteration']['policy']
    
    for alg_name, result in results.items():
        policy_match = torch.sum(result['policy'] == baseline_policy).item()
        match_pct = (policy_match / len(baseline_policy)) * 100
        
        print(f"{alg_name:<18} {result['time']:<10.4f} {result['return']:<10.4f} "
              f"{result['iterations']:<12} {match_pct:<12.1f}%")
    
    # Visualize comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    algorithms = list(results.keys())
    times = [results[alg]['time'] for alg in algorithms]
    returns = [results[alg]['return'] for alg in algorithms]
    iterations = [results[alg]['iterations'] for alg in algorithms]
    
    # Time comparison
    ax1.bar(algorithms, times, alpha=0.7, color=['green', 'blue', 'red', 'orange'])
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time')
    ax1.tick_params(axis='x', rotation=45)
    
    # Return comparison
    ax2.bar(algorithms, returns, alpha=0.7, color=['green', 'blue', 'red', 'orange'])
    ax2.set_ylabel('Average Return')
    ax2.set_title('Policy Performance')
    ax2.tick_params(axis='x', rotation=45)
    
    # Iterations comparison
    ax3.bar(algorithms, iterations, alpha=0.7, color=['green', 'blue', 'red', 'orange'])
    ax3.set_ylabel('Iterations/Episodes')
    ax3.set_title('Convergence Speed')
    ax3.tick_params(axis='x', rotation=45)
    
    # Learning curves for RL algorithms
    if len(ql.episode_returns) > 0 and len(sarsa.episode_returns) > 0:
        window_size = 50
        ql_ma = np.convolve(ql.episode_returns, np.ones(window_size)/window_size, mode='valid')
        sarsa_ma = np.convolve(sarsa.episode_returns, np.ones(window_size)/window_size, mode='valid')
        
        ax4.plot(range(window_size-1, len(ql.episode_returns)), ql_ma, 'r-', linewidth=2, label='Q-Learning')
        ax4.plot(range(window_size-1, len(sarsa.episode_returns)), sarsa_ma, 'orange', linewidth=2, label='SARSA')
        ax4.axhline(y=vi_return, color='g', linestyle='--', label=f'Value Iteration ({vi_return:.2f})')
        ax4.axhline(y=pi_return, color='b', linestyle='--', label=f'Policy Iteration ({pi_return:.2f})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Return')
        ax4.set_title('RL Learning Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def convergence_analysis_demo():
    """
    Analyze convergence properties and performance characteristics.
    """
    print("Convergence and Performance Analysis...")
    
    # Test different environment sizes
    print("\n1. SCALABILITY ANALYSIS")
    print("-" * 25)
    
    grid_sizes = [3, 4, 5, 6]
    scalability_results = {}
    
    for size in grid_sizes:
        print(f"Testing {size}x{size} grid...")
        env = GridWorldMDP(grid_size=size, goal_states=[size*size-1])
        
        # Value Iteration
        vi = ValueIteration(env, tolerance=1e-6, max_iterations=1000)
        start_time = time.time()
        vi.solve(verbose=False)
        vi_time = time.time() - start_time
        
        # Q-Learning
        ql = QLearning(env, learning_rate=0.1, epsilon=0.9)
        start_time = time.time()
        ql.train(num_episodes=500, verbose=False)
        ql_time = time.time() - start_time
        
        scalability_results[size] = {
            'states': size * size,
            'vi_time': vi_time,
            'vi_iterations': vi.iterations_to_convergence,
            'ql_time': ql_time,
            'ql_episodes': len(ql.episode_returns)
        }
    
    # Print scalability results
    print("\nScalability Results:")
    print(f"{'Size':<6} {'States':<8} {'VI Time':<10} {'VI Iters':<10} {'QL Time':<10} {'QL Episodes':<12}")
    print("-" * 70)
    
    for size, result in scalability_results.items():
        print(f"{size}x{size:<4} {result['states']:<8} {result['vi_time']:<10.4f} "
              f"{result['vi_iterations']:<10} {result['ql_time']:<10.4f} {result['ql_episodes']:<12}")
    
    # Test different tolerance levels
    print("\n2. TOLERANCE ANALYSIS")
    print("-" * 20)
    
    env = GridWorldMDP(grid_size=4, goal_states=[15])
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    tolerance_results = {}
    
    for tol in tolerances:
        vi = ValueIteration(env, tolerance=tol, max_iterations=1000)
        start_time = time.time()
        values, policy = vi.solve(verbose=False)
        solve_time = time.time() - start_time
        
        tolerance_results[tol] = {
            'time': solve_time,
            'iterations': vi.iterations_to_convergence,
            'final_delta': vi.delta_history[-1] if vi.delta_history else 0,
            'policy': policy
        }
    
    print("\nTolerance Analysis Results:")
    print(f"{'Tolerance':<12} {'Time (s)':<10} {'Iterations':<12} {'Final Delta':<15}")
    print("-" * 55)
    
    for tol, result in tolerance_results.items():
        print(f"{tol:<12.0e} {result['time']:<10.4f} {result['iterations']:<12} {result['final_delta']:<15.2e}")
    
    # Visualize scalability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scalability plot
    sizes = list(scalability_results.keys())
    states = [scalability_results[s]['states'] for s in sizes]
    vi_times = [scalability_results[s]['vi_time'] for s in sizes]
    ql_times = [scalability_results[s]['ql_time'] for s in sizes]
    
    ax1.plot(states, vi_times, 'g-o', linewidth=2, label='Value Iteration')
    ax1.plot(states, ql_times, 'r-o', linewidth=2, label='Q-Learning')
    ax1.set_xlabel('Number of States')
    ax1.set_ylabel('Computation Time (s)')
    ax1.set_title('Scalability Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tolerance plot
    tols = list(tolerance_results.keys())
    tol_times = [tolerance_results[t]['time'] for t in tols]
    tol_iters = [tolerance_results[t]['iterations'] for t in tols]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.semilogx(tols, tol_times, 'b-o', linewidth=2, label='Time')
    line2 = ax2_twin.semilogx(tols, tol_iters, 'r-s', linewidth=2, label='Iterations')
    
    ax2.set_xlabel('Tolerance')
    ax2.set_ylabel('Time (s)', color='b')
    ax2_twin.set_ylabel('Iterations', color='r')
    ax2.set_title('Tolerance vs Performance')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the comprehensive demonstration
    run_comprehensive_demo()
    
    print("\n" + "="*80)
    print("ADDITIONAL RESOURCES")
    print("="*80)
    print("For more information about the theoretical foundations,")
    print("please refer to the enhanced MDP guide: 'mdp_guide_enhanced.md'")
    print()
    print("Individual algorithm demonstrations can be run separately:")
    print("- python base_mdp.py")
    print("- python value_iteration.py") 
    print("- python policy_iteration.py")
    print("- python q_learning.py")
    print("- python sarsa.py")
    print("- python healthcare_examples.py")
    print("="*80)

