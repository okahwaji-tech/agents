"""
Q-Learning Algorithm Implementation
==================================

This module implements the Q-Learning algorithm, a model-free reinforcement learning
method that learns optimal action-value functions directly from experience without
requiring knowledge of the environment's transition probabilities or reward function.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Callable
import time
import random
from base_mdp import BaseMDP


class QLearning:
    """
    Q-Learning algorithm for model-free reinforcement learning.
    
    Q-Learning learns the optimal action-value function Q*(s,a) using the update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, 
                 mdp: BaseMDP,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 exploration_strategy: str = 'epsilon_greedy'):
        """
        Initialize the Q-Learning algorithm.
        
        Args:
            mdp: The MDP environment
            learning_rate: Learning rate α for Q-value updates
            epsilon: Initial exploration rate for ε-greedy policy
            epsilon_decay: Decay factor for epsilon
            epsilon_min: Minimum epsilon value
            exploration_strategy: Exploration strategy ('epsilon_greedy', 'boltzmann', 'ucb')
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
        self.action_counts = torch.zeros(
            (mdp.num_states, mdp.num_actions),
            device=mdp.device,
            dtype=torch.long
        )
        
        # For UCB exploration
        self.total_steps = 0
    
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
        elif self.exploration_strategy == 'ucb':
            return self._ucb_action(state)
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
    
    def _ucb_action(self, state: int, c: float = 2.0) -> int:
        """
        Select action using Upper Confidence Bound (UCB) exploration.
        
        Args:
            state: Current state
            c: Exploration parameter
            
        Returns:
            Selected action
        """
        if self.total_steps == 0:
            return random.randint(0, self.mdp.num_actions - 1)
        
        ucb_values = torch.zeros(self.mdp.num_actions, device=self.mdp.device)
        
        for action in range(self.mdp.num_actions):
            count = self.action_counts[state, action].item()
            if count == 0:
                ucb_values[action] = float('inf')  # Unvisited actions have highest priority
            else:
                confidence = c * np.sqrt(np.log(self.total_steps) / count)
                ucb_values[action] = self.q_table[state, action] + confidence
        
        return torch.argmax(ucb_values).item()
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-value using the Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for next state
        max_next_q = torch.max(self.q_table[next_state])
        
        # Q-learning target
        target = reward + self.mdp.gamma * max_next_q
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)
        
        # Update action count for UCB
        self.action_counts[state, action] += 1
        self.total_steps += 1
    
    def train(self, 
              num_episodes: int = 1000,
              max_steps_per_episode: int = 100,
              verbose: bool = True,
              log_interval: int = 100) -> Tuple[torch.Tensor, List[float]]:
        """
        Train the Q-learning agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            log_interval: Interval for logging progress
            
        Returns:
            Tuple of (final_q_table, episode_returns)
        """
        if verbose:
            print("Starting Q-Learning Training...")
            print(f"Episodes: {num_episodes}")
            print(f"Learning rate: {self.alpha}")
            print(f"Initial epsilon: {self.epsilon}")
            print(f"Exploration strategy: {self.exploration_strategy}")
            print("-" * 50)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.mdp.reset()
            episode_return = 0.0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(state, training=True)
                
                # Take action
                next_state, reward, done = self.mdp.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Update episode statistics
                episode_return += reward
                episode_length += 1
                
                # Move to next state
                state = next_state
                
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
    
    def evaluate_policy(self, 
                       num_episodes: int = 100,
                       max_steps_per_episode: int = 100) -> Tuple[float, float]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            
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
                # Greedy action selection
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
        ax1.set_title('Learning Progress: Episode Returns')
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
        ax2.set_title('Learning Progress: Episode Lengths')
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
        
        plt.title('Q-Table: Learned Action Values')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self):
        """
        Print a summary of the Q-learning results.
        """
        print("=" * 60)
        print("Q-Learning Results")
        print("=" * 60)
        print(f"Total episodes trained: {len(self.episode_returns)}")
        print(f"Final epsilon: {self.epsilon:.4f}")
        print(f"Total steps taken: {self.total_steps}")
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


def demonstrate_q_learning():
    """
    Demonstrate Q-Learning on a simple grid world problem.
    """
    print("Q-Learning Demonstration")
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
    
    # Train Q-Learning agent
    agent = QLearning(
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
    avg_return, avg_length = agent.evaluate_policy(num_episodes=100)
    print(f"\nEvaluation Results:")
    print(f"Average return: {avg_return:.4f}")
    print(f"Average episode length: {avg_length:.2f}")
    
    # Plot learning curves
    agent.plot_learning_curves()
    
    # Visualize Q-table
    agent.visualize_q_table()
    
    # Visualize learned policy
    policy = agent.extract_policy()
    print("\nLearned Policy Visualization:")
    env.visualize_grid(policy=policy)


def compare_exploration_strategies():
    """
    Compare different exploration strategies in Q-Learning.
    """
    print("\n" + "="*60)
    print("Comparing Q-Learning Exploration Strategies")
    print("="*60)
    
    from base_mdp import GridWorldMDP
    
    # Create environment
    env = GridWorldMDP(grid_size=4, goal_states=[15], obstacle_states=[5, 9])
    
    strategies = ['epsilon_greedy', 'boltzmann', 'ucb']
    results = {}
    
    for strategy in strategies:
        print(f"\nTraining with {strategy} exploration...")
        
        agent = QLearning(
            mdp=env,
            learning_rate=0.1,
            epsilon=0.9,
            exploration_strategy=strategy
        )
        
        # Train
        start_time = time.time()
        agent.train(num_episodes=500, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        avg_return, avg_length = agent.evaluate_policy(num_episodes=100)
        
        results[strategy] = {
            'avg_return': avg_return,
            'avg_length': avg_length,
            'training_time': training_time,
            'final_returns': agent.episode_returns[-100:]
        }
    
    # Print comparison
    print("\nComparison Results:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Avg Return':<12} {'Avg Length':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for strategy, result in results.items():
        print(f"{strategy:<15} {result['avg_return']:<12.4f} "
              f"{result['avg_length']:<12.2f} {result['training_time']:<10.3f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Compare final performance
    strategies_list = list(results.keys())
    returns = [results[s]['avg_return'] for s in strategies_list]
    lengths = [results[s]['avg_length'] for s in strategies_list]
    
    ax1.bar(strategies_list, returns, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_ylabel('Average Return')
    ax1.set_title('Final Performance Comparison')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(strategies_list, lengths, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_ylabel('Average Episode Length')
    ax2.set_title('Episode Length Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demonstrate_q_learning()
    compare_exploration_strategies()

