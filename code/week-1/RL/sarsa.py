"""
SARSA Algorithm Implementation
=============================

This module implements the SARSA (State-Action-Reward-State-Action) algorithm,
an on-policy temporal difference learning method that learns the value of the
policy being followed rather than the optimal policy.

Author: Manus AI
Date: June 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import time
import random
from base_mdp import BaseMDP


class SARSA:
    """
    SARSA algorithm for on-policy reinforcement learning.
    
    SARSA learns the action-value function Q^π(s,a) for the policy being followed
    using the update rule:
    Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    where a' is the action actually taken in state s' according to the current policy.
    """
    
    def __init__(self, 
                 mdp: BaseMDP,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 exploration_strategy: str = 'epsilon_greedy'):
        """
        Initialize the SARSA algorithm.
        
        Args:
            mdp: The MDP environment
            learning_rate: Learning rate α for Q-value updates
            epsilon: Initial exploration rate for ε-greedy policy
            epsilon_decay: Decay factor for epsilon
            epsilon_min: Minimum epsilon value
            exploration_strategy: Exploration strategy ('epsilon_greedy', 'boltzmann')
        """
        self.mdp = mdp
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_strategy = exploration_strategy
        
        # Initialize Q-table
        self.q_table = torch.zeros(
            (mdp.num_states, mdp.num_actions),
            device=mdp.device,
            dtype=torch.float32
        )
        
        # Track learning progress
        self.episode_returns = []
        self.episode_lengths = []
        self.q_value_history = []
        self.epsilon_history = []
        self.policy_updates = []
        
        # Track state-action visitation for analysis
        self.visitation_counts = torch.zeros(
            (mdp.num_states, mdp.num_actions),
            device=mdp.device,
            dtype=torch.long
        )
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action using the current exploration strategy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if not training:
            # Greedy action selection during evaluation
            return torch.argmax(self.q_table[state]).item()
        
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_action(state)
        elif self.exploration_strategy == 'boltzmann':
            return self._boltzmann_action(state)
        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration_strategy}")
    
    def _epsilon_greedy_action(self, state: int) -> int:
        """
        Select action using ε-greedy exploration.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.mdp.num_actions - 1)
        else:
            # Exploit: greedy action
            return torch.argmax(self.q_table[state]).item()
    
    def _boltzmann_action(self, state: int, temperature: float = 1.0) -> int:
        """
        Select action using Boltzmann (softmax) exploration.
        
        Args:
            state: Current state
            temperature: Temperature parameter for softmax
            
        Returns:
            Selected action
        """
        q_values = self.q_table[state] / temperature
        probabilities = torch.softmax(q_values, dim=0)
        return torch.multinomial(probabilities, 1).item()
    
    def update_q_value(self, 
                      state: int, 
                      action: int, 
                      reward: float, 
                      next_state: int, 
                      next_action: int):
        """
        Update Q-value using the SARSA update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (actually taken)
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Q-value for next state-action pair
        next_q = self.q_table[next_state, next_action]
        
        # SARSA target
        target = reward + self.mdp.gamma * next_q
        
        # SARSA update
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)
        
        # Update visitation count
        self.visitation_counts[state, action] += 1
    
    def train(self, 
              num_episodes: int = 1000,
              max_steps_per_episode: int = 100,
              verbose: bool = True,
              log_interval: int = 100) -> Tuple[torch.Tensor, List[float]]:
        """
        Train the SARSA agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            log_interval: Interval for logging progress
            
        Returns:
            Tuple of (final_q_table, episode_returns)
        """
        if verbose:
            print("Starting SARSA Training...")
            print(f"Episodes: {num_episodes}")
            print(f"Learning rate: {self.alpha}")
            print(f"Initial epsilon: {self.epsilon}")
            print(f"Exploration strategy: {self.exploration_strategy}")
            print("-" * 50)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment and select initial action
            state = self.mdp.reset()
            action = self.select_action(state, training=True)
            
            episode_return = 0.0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # Take action
                next_state, reward, done = self.mdp.step(action)
                
                # Select next action
                next_action = self.select_action(next_state, training=True)
                
                # Update Q-value using SARSA rule
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # Update episode statistics
                episode_return += reward
                episode_length += 1
                
                # Move to next state-action pair
                state = next_state
                action = next_action
                
                if done:
                    break
            
            # Store episode statistics
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            self.epsilon_history.append(self.epsilon)
            
            # Store Q-value snapshot (for convergence analysis)
            if episode % 10 == 0:
                self.q_value_history.append(self.q_table.clone())
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Log progress
            if verbose and (episode + 1) % log_interval == 0:
                avg_return = np.mean(self.episode_returns[-log_interval:])
                avg_length = np.mean(self.episode_lengths[-log_interval:])
                print(f"Episode {episode + 1:4d}: "
                      f"Avg Return = {avg_return:6.2f}, "
                      f"Avg Length = {avg_length:5.1f}, "
                      f"Epsilon = {self.epsilon:.3f}")
        
        end_time = time.time()
        
        if verbose:
            print("-" * 50)
            print(f"Training completed in {end_time - start_time:.2f} seconds")
            print(f"Final epsilon: {self.epsilon:.4f}")
        
        return self.q_table, self.episode_returns
    
    def extract_policy(self) -> torch.Tensor:
        """
        Extract the greedy policy from the Q-table.
        
        Returns:
            Policy tensor (action for each state)
        """
        return torch.argmax(self.q_table, dim=1)
    
    def get_policy_probabilities(self, state: int) -> torch.Tensor:
        """
        Get the action probabilities for the current policy in a given state.
        
        Args:
            state: State to get probabilities for
            
        Returns:
            Probability distribution over actions
        """
        if self.exploration_strategy == 'epsilon_greedy':
            probs = torch.full((self.mdp.num_actions,), 
                             self.epsilon / self.mdp.num_actions,
                             device=self.mdp.device)
            best_action = torch.argmax(self.q_table[state])
            probs[best_action] += 1.0 - self.epsilon
            return probs
        elif self.exploration_strategy == 'boltzmann':
            return torch.softmax(self.q_table[state], dim=0)
        else:
            # Default to greedy
            probs = torch.zeros(self.mdp.num_actions, device=self.mdp.device)
            probs[torch.argmax(self.q_table[state])] = 1.0
            return probs
    
    def evaluate_policy(self, 
                       num_episodes: int = 100,
                       max_steps_per_episode: int = 100,
                       use_learned_policy: bool = True) -> Tuple[float, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            use_learned_policy: Whether to use the learned policy or greedy policy
            
        Returns:
            Tuple of (average_return, average_length)
        """
        returns = []
        lengths = []
        
        for episode in range(num_episodes):
            state = self.mdp.reset()
            episode_return = 0.0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                if use_learned_policy:
                    # Use the same policy as during training
                    action = self.select_action(state, training=True)
                else:
                    # Use greedy policy
                    action = self.select_action(state, training=False)
                
                next_state, reward, done = self.mdp.step(action)
                
                episode_return += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        return np.mean(returns), np.mean(lengths)
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """
        Plot learning curves showing training progress.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.episode_returns:
            print("No training data available. Run train() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(self.episode_returns))
        
        # Plot episode returns
        ax1.plot(episodes, self.episode_returns, alpha=0.6, linewidth=0.5)
        # Add moving average
        window_size = min(50, len(self.episode_returns) // 10)
        if window_size > 1:
            moving_avg = np.convolve(self.episode_returns, 
                                   np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(self.episode_returns)), 
                    moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            ax1.legend()
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Return')
        ax1.set_title('SARSA Learning Progress: Episode Returns')
        ax1.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax2.plot(episodes, self.episode_lengths, alpha=0.6, linewidth=0.5)
        if window_size > 1:
            moving_avg_length = np.convolve(self.episode_lengths, 
                                          np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(self.episode_lengths)), 
                    moving_avg_length, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            ax2.legend()
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('SARSA Learning Progress: Episode Lengths')
        ax2.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        ax3.plot(episodes, self.epsilon_history, 'g-', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate (Epsilon) Decay')
        ax3.grid(True, alpha=0.3)
        
        # Plot Q-value convergence for a few state-action pairs
        if self.q_value_history:
            num_pairs_to_plot = min(5, self.mdp.num_states * self.mdp.num_actions)
            pairs_to_plot = [(i // self.mdp.num_actions, i % self.mdp.num_actions) 
                           for i in range(0, self.mdp.num_states * self.mdp.num_actions, 
                                        max(1, self.mdp.num_states * self.mdp.num_actions // num_pairs_to_plot))]
            
            q_episodes = range(0, len(self.episode_returns), 10)
            for s, a in pairs_to_plot[:5]:
                q_values_for_pair = [q[s, a].item() for q in self.q_value_history]
                ax4.plot(q_episodes[:len(q_values_for_pair)], q_values_for_pair, 
                        label=f'Q({s},{a})', linewidth=2)
            
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Q-Value')
            ax4.set_title('Q-Value Convergence')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_q_table(self, save_path: Optional[str] = None):
        """
        Visualize the Q-table as a heatmap.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        import seaborn as sns
        sns.heatmap(
            self.q_table.cpu().numpy(),
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=self.mdp.action_names,
            yticklabels=self.mdp.state_names,
            cbar_kws={'label': 'Q-Value'}
        )
        
        plt.title('SARSA Q-Table: Learned Action Values')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_visitation_counts(self, save_path: Optional[str] = None):
        """
        Visualize state-action visitation counts.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        import seaborn as sns
        sns.heatmap(
            self.visitation_counts.cpu().numpy(),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.mdp.action_names,
            yticklabels=self.mdp.state_names,
            cbar_kws={'label': 'Visitation Count'}
        )
        
        plt.title('State-Action Visitation Counts')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self):
        """
        Print a summary of the SARSA results.
        """
        print("=" * 60)
        print("SARSA Results")
        print("=" * 60)
        print(f"Total episodes trained: {len(self.episode_returns)}")
        print(f"Final epsilon: {self.epsilon:.4f}")
        print(f"Total state-action visits: {torch.sum(self.visitation_counts).item()}")
        print()
        
        if self.episode_returns:
            print("Training Performance:")
            print(f"  Average return (last 100 episodes): {np.mean(self.episode_returns[-100:]):.4f}")
            print(f"  Best episode return: {max(self.episode_returns):.4f}")
            print(f"  Average episode length: {np.mean(self.episode_lengths):.2f}")
        
        print()
        print("Learned Policy:")
        policy = self.extract_policy()
        for state in range(self.mdp.num_states):
            action = policy[state].item()
            print(f"π({self.mdp.state_names[state]}) = {self.mdp.action_names[action]}")
        print("=" * 60)


def demonstrate_sarsa():
    """
    Demonstrate SARSA on a simple grid world problem.
    """
    print("SARSA Demonstration")
    print("=" * 50)
    
    # Import GridWorldMDP
    from base_mdp import GridWorldMDP
    
    # Create a 4x4 grid world
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
    
    # Train SARSA agent
    agent = SARSA(
        mdp=env,
        learning_rate=0.1,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        exploration_strategy='epsilon_greedy'
    )
    
    # Train the agent
    q_table, returns = agent.train(num_episodes=1000, verbose=True)
    
    # Print results
    agent.print_results()
    
    # Evaluate the learned policy
    avg_return_learned, avg_length_learned = agent.evaluate_policy(
        num_episodes=100, use_learned_policy=True
    )
    avg_return_greedy, avg_length_greedy = agent.evaluate_policy(
        num_episodes=100, use_learned_policy=False
    )
    
    print(f"\nEvaluation Results:")
    print(f"Learned policy - Avg return: {avg_return_learned:.4f}, Avg length: {avg_length_learned:.2f}")
    print(f"Greedy policy - Avg return: {avg_return_greedy:.4f}, Avg length: {avg_length_greedy:.2f}")
    
    # Plot learning curves
    agent.plot_learning_curves()
    
    # Visualize Q-table and visitation counts
    agent.visualize_q_table()
    agent.visualize_visitation_counts()
    
    # Visualize learned policy
    policy = agent.extract_policy()
    print("\nLearned Policy Visualization:")
    env.visualize_grid(policy=policy)


def compare_sarsa_vs_qlearning():
    """
    Compare SARSA with Q-Learning on the same problem.
    """
    print("\n" + "="*60)
    print("Comparing SARSA vs Q-Learning")
    print("="*60)
    
    from base_mdp import GridWorldMDP
    from q_learning import QLearning
    
    # Create environment
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    # Train SARSA
    print("Training SARSA...")
    sarsa_agent = SARSA(
        mdp=env,
        learning_rate=0.1,
        epsilon=0.9,
        epsilon_decay=0.995,
        exploration_strategy='epsilon_greedy'
    )
    
    start_time = time.time()
    sarsa_agent.train(num_episodes=1000, verbose=False)
    sarsa_time = time.time() - start_time
    
    # Train Q-Learning
    print("Training Q-Learning...")
    qlearning_agent = QLearning(
        mdp=env,
        learning_rate=0.1,
        epsilon=0.9,
        epsilon_decay=0.995,
        exploration_strategy='epsilon_greedy'
    )
    
    start_time = time.time()
    qlearning_agent.train(num_episodes=1000, verbose=False)
    qlearning_time = time.time() - start_time
    
    # Compare results
    print("\nComparison Results:")
    print("-" * 60)
    
    # Evaluate policies
    sarsa_return, sarsa_length = sarsa_agent.evaluate_policy(100, use_learned_policy=False)
    qlearning_return, qlearning_length = qlearning_agent.evaluate_policy(100)
    
    print(f"{'Algorithm':<12} {'Avg Return':<12} {'Avg Length':<12} {'Time (s)':<10}")
    print("-" * 60)
    print(f"{'SARSA':<12} {sarsa_return:<12.4f} {sarsa_length:<12.2f} {sarsa_time:<10.3f}")
    print(f"{'Q-Learning':<12} {qlearning_return:<12.4f} {qlearning_length:<12.2f} {qlearning_time:<10.3f}")
    
    # Compare Q-tables
    q_diff = torch.max(torch.abs(sarsa_agent.q_table - qlearning_agent.q_table)).item()
    policy_diff = torch.sum(
        sarsa_agent.extract_policy() != qlearning_agent.extract_policy()
    ).item()
    
    print(f"\nQ-table differences:")
    print(f"Max Q-value difference: {q_diff:.6f}")
    print(f"Policy differences: {policy_diff} states")
    
    # Plot learning curves comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Compare learning curves
    episodes = range(len(sarsa_agent.episode_returns))
    
    # Moving averages
    window_size = 50
    sarsa_ma = np.convolve(sarsa_agent.episode_returns, 
                          np.ones(window_size)/window_size, mode='valid')
    qlearning_ma = np.convolve(qlearning_agent.episode_returns, 
                              np.ones(window_size)/window_size, mode='valid')
    
    ax1.plot(range(window_size-1, len(sarsa_agent.episode_returns)), 
            sarsa_ma, 'b-', linewidth=2, label='SARSA')
    ax1.plot(range(window_size-1, len(qlearning_agent.episode_returns)), 
            qlearning_ma, 'r-', linewidth=2, label='Q-Learning')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Learning Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Compare final performance
    algorithms = ['SARSA', 'Q-Learning']
    returns = [sarsa_return, qlearning_return]
    
    ax2.bar(algorithms, returns, alpha=0.7, color=['skyblue', 'salmon'])
    ax2.set_ylabel('Average Return')
    ax2.set_title('Final Performance Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demonstrate_sarsa()
    compare_sarsa_vs_qlearning()

