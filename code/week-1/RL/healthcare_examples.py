"""
Healthcare MDP Examples
======================

This module implements healthcare-specific MDP examples to demonstrate
the application of Markov Decision Processes in medical decision-making.
These examples illustrate how MDPs can be used for treatment planning,
medication dosing, and patient care optimization.

Author: Manus AI
Date: June 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from base_mdp import BaseMDP
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from q_learning import QLearning
from sarsa import SARSA


class DiabetesTreatmentMDP(BaseMDP):
    """
    MDP for diabetes treatment decision-making.
    
    This example models the decision process for managing a diabetic patient's
    blood glucose levels through medication adjustments and lifestyle interventions.
    
    States represent glucose level ranges and patient condition.
    Actions represent different treatment intensities.
    Rewards balance glucose control with treatment burden and side effects.
    """
    
    def __init__(self, discount_factor: float = 0.95, device: str = 'cpu'):
        """
        Initialize the diabetes treatment MDP.
        
        Args:
            discount_factor: Discount factor for future rewards
            device: PyTorch device
        """
        # Define states: glucose levels and patient condition
        # States 0-2: Low glucose (hypoglycemic)
        # States 3-5: Normal glucose (target range)
        # States 6-8: High glucose (hyperglycemic)
        # Each glucose level has 3 sub-states for patient condition (good, fair, poor)
        num_states = 9
        
        # Define actions: treatment intensity levels
        # 0: Reduce medication
        # 1: Maintain current treatment
        # 2: Increase medication moderately
        # 3: Increase medication significantly
        num_actions = 4
        
        super().__init__(num_states, num_actions, discount_factor, device)
        
        # Define meaningful state and action names
        glucose_levels = ['Low', 'Normal', 'High']
        conditions = ['Good', 'Fair', 'Poor']
        
        self.state_names = [f"State_{i}" for i in range(num_states)]  # Initialize first
        for i, glucose in enumerate(glucose_levels):
            for j, condition in enumerate(conditions):
                state_idx = i * 3 + j
                self.state_names[state_idx] = f"{glucose}_Glucose_{condition}_Condition"
        
        self.action_names = [
            'Reduce_Medication',
            'Maintain_Treatment', 
            'Increase_Moderate',
            'Increase_Significant'
        ]
        
        # Define glucose level and condition mappings
        self.glucose_states = {
            'low': [0, 1, 2],      # Low glucose states
            'normal': [3, 4, 5],   # Normal glucose states  
            'high': [6, 7, 8]      # High glucose states
        }
        
        self.condition_states = {
            'good': [0, 3, 6],     # Good condition states
            'fair': [1, 4, 7],     # Fair condition states
            'poor': [2, 5, 8]      # Poor condition states
        }
        
        self.setup_environment()
    
    def setup_environment(self):
        """
        Setup the diabetes treatment MDP with realistic transition probabilities and rewards.
        """
        # Setup transition probabilities
        self._setup_transitions()
        
        # Setup reward function
        self._setup_rewards()
        
        # Compute expected rewards
        self.compute_expected_rewards()
    
    def _setup_transitions(self):
        """
        Setup transition probabilities based on medical knowledge.
        """
        for state in range(self.num_states):
            glucose_level = self._get_glucose_level(state)
            condition = self._get_condition_level(state)
            
            for action in range(self.num_actions):
                # Define transition probabilities based on current state and action
                if glucose_level == 'low':
                    self._set_low_glucose_transitions(state, action, condition)
                elif glucose_level == 'normal':
                    self._set_normal_glucose_transitions(state, action, condition)
                else:  # high glucose
                    self._set_high_glucose_transitions(state, action, condition)
    
    def _get_glucose_level(self, state: int) -> str:
        """Get glucose level for a given state."""
        if state in self.glucose_states['low']:
            return 'low'
        elif state in self.glucose_states['normal']:
            return 'normal'
        else:
            return 'high'
    
    def _get_condition_level(self, state: int) -> str:
        """Get condition level for a given state."""
        if state in self.condition_states['good']:
            return 'good'
        elif state in self.condition_states['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _set_low_glucose_transitions(self, state: int, action: int, condition: str):
        """Set transitions for low glucose states."""
        condition_idx = ['good', 'fair', 'poor'].index(condition)
        
        if action == 0:  # Reduce medication - helps with hypoglycemia
            # High probability of moving to normal glucose
            self.set_transition_probability(state, action, 3 + condition_idx, 0.7)
            self.set_transition_probability(state, action, state, 0.2)
            self.set_transition_probability(state, action, 6 + condition_idx, 0.1)
        elif action == 1:  # Maintain treatment
            self.set_transition_probability(state, action, 3 + condition_idx, 0.5)
            self.set_transition_probability(state, action, state, 0.4)
            self.set_transition_probability(state, action, 6 + condition_idx, 0.1)
        else:  # Increase medication - worsens hypoglycemia
            self.set_transition_probability(state, action, state, 0.8)
            self.set_transition_probability(state, action, 3 + condition_idx, 0.2)
    
    def _set_normal_glucose_transitions(self, state: int, action: int, condition: str):
        """Set transitions for normal glucose states."""
        condition_idx = ['good', 'fair', 'poor'].index(condition)
        
        if action == 0:  # Reduce medication - risk of hyperglycemia
            self.set_transition_probability(state, action, state, 0.4)
            self.set_transition_probability(state, action, 6 + condition_idx, 0.5)
            self.set_transition_probability(state, action, 0 + condition_idx, 0.1)
        elif action == 1:  # Maintain treatment - stable
            self.set_transition_probability(state, action, state, 0.8)
            self.set_transition_probability(state, action, 6 + condition_idx, 0.1)
            self.set_transition_probability(state, action, 0 + condition_idx, 0.1)
        elif action == 2:  # Moderate increase
            self.set_transition_probability(state, action, state, 0.7)
            self.set_transition_probability(state, action, 0 + condition_idx, 0.2)
            self.set_transition_probability(state, action, 6 + condition_idx, 0.1)
        else:  # Significant increase - risk of hypoglycemia
            self.set_transition_probability(state, action, 0 + condition_idx, 0.6)
            self.set_transition_probability(state, action, state, 0.4)
    
    def _set_high_glucose_transitions(self, state: int, action: int, condition: str):
        """Set transitions for high glucose states."""
        condition_idx = ['good', 'fair', 'poor'].index(condition)
        
        if action == 0:  # Reduce medication - worsens hyperglycemia
            self.set_transition_probability(state, action, state, 0.9)
            self.set_transition_probability(state, action, 3 + condition_idx, 0.1)
        elif action == 1:  # Maintain treatment
            self.set_transition_probability(state, action, state, 0.6)
            self.set_transition_probability(state, action, 3 + condition_idx, 0.4)
        elif action == 2:  # Moderate increase - helps hyperglycemia
            self.set_transition_probability(state, action, 3 + condition_idx, 0.6)
            self.set_transition_probability(state, action, state, 0.3)
            self.set_transition_probability(state, action, 0 + condition_idx, 0.1)
        else:  # Significant increase - strong effect
            self.set_transition_probability(state, action, 3 + condition_idx, 0.5)
            self.set_transition_probability(state, action, 0 + condition_idx, 0.4)
            self.set_transition_probability(state, action, state, 0.1)
    
    def _setup_rewards(self):
        """
        Setup reward function balancing glucose control, treatment burden, and side effects.
        """
        for state in range(self.num_states):
            glucose_level = self._get_glucose_level(state)
            condition = self._get_condition_level(state)
            
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    reward = self._calculate_reward(state, action, next_state, 
                                                  glucose_level, condition)
                    self.set_reward(state, action, next_state, reward)
    
    def _calculate_reward(self, state: int, action: int, next_state: int, 
                         glucose_level: str, condition: str) -> float:
        """
        Calculate reward based on glucose control, treatment burden, and outcomes.
        """
        reward = 0.0
        
        next_glucose = self._get_glucose_level(next_state)
        next_condition = self._get_condition_level(next_state)
        
        # Reward for glucose control
        if next_glucose == 'normal':
            reward += 10.0  # High reward for target glucose
        elif next_glucose == 'low':
            reward -= 15.0  # Severe penalty for hypoglycemia (dangerous)
        else:  # high glucose
            reward -= 8.0   # Moderate penalty for hyperglycemia
        
        # Reward for patient condition
        condition_rewards = {'good': 5.0, 'fair': 0.0, 'poor': -5.0}
        reward += condition_rewards[next_condition]
        
        # Penalty for treatment burden (medication intensity)
        treatment_penalties = {0: 0.0, 1: -1.0, 2: -2.0, 3: -4.0}
        reward += treatment_penalties[action]
        
        # Bonus for maintaining stable glucose
        if glucose_level == 'normal' and next_glucose == 'normal':
            reward += 3.0
        
        # Penalty for glucose swings
        if ((glucose_level == 'low' and next_glucose == 'high') or 
            (glucose_level == 'high' and next_glucose == 'low')):
            reward -= 5.0
        
        return reward
    
    def is_terminal(self, state: int) -> bool:
        """
        No terminal states in this continuous care model.
        """
        return False
    
    def get_clinical_interpretation(self, policy: torch.Tensor) -> Dict[str, str]:
        """
        Provide clinical interpretation of the optimal policy.
        
        Args:
            policy: Optimal policy tensor
            
        Returns:
            Dictionary mapping states to clinical recommendations
        """
        interpretations = {}
        
        for state in range(self.num_states):
            action = policy[state].item()
            glucose_level = self._get_glucose_level(state)
            condition = self._get_condition_level(state)
            
            action_desc = self.action_names[action]
            
            # Clinical interpretation
            if glucose_level == 'low':
                if action == 0:
                    interpretation = "APPROPRIATE: Reduce medication to prevent hypoglycemia"
                else:
                    interpretation = "CAUTION: Consider reducing medication for hypoglycemia"
            elif glucose_level == 'normal':
                if action == 1:
                    interpretation = "APPROPRIATE: Maintain current stable treatment"
                else:
                    interpretation = "REVIEW: Consider if treatment change is necessary"
            else:  # high glucose
                if action in [2, 3]:
                    interpretation = "APPROPRIATE: Increase medication for hyperglycemia"
                else:
                    interpretation = "SUBOPTIMAL: Consider increasing treatment intensity"
            
            interpretations[self.state_names[state]] = f"{action_desc} - {interpretation}"
        
        return interpretations


class SepsisManagementMDP(BaseMDP):
    """
    MDP for sepsis management in intensive care.
    
    This example models the critical decision-making process for managing
    sepsis patients, including antibiotic therapy, fluid management, and
    supportive care decisions.
    """
    
    def __init__(self, discount_factor: float = 0.9, device: str = 'cpu'):
        """
        Initialize the sepsis management MDP.
        
        Args:
            discount_factor: Discount factor (lower due to urgency)
            device: PyTorch device
        """
        # States represent severity levels and organ function
        # 0-2: Mild sepsis (stable, declining, improving)
        # 3-5: Severe sepsis (stable, declining, improving) 
        # 6-8: Septic shock (stable, declining, improving)
        num_states = 9
        
        # Actions represent treatment intensity
        # 0: Conservative management
        # 1: Standard protocol
        # 2: Aggressive treatment
        # 3: Maximum intervention
        num_actions = 4
        
        super().__init__(num_states, num_actions, discount_factor, device)
        
        # Define state names
        severity_levels = ['Mild_Sepsis', 'Severe_Sepsis', 'Septic_Shock']
        trends = ['Stable', 'Declining', 'Improving']
        
        self.state_names = [f"State_{i}" for i in range(num_states)]  # Initialize first
        for i, severity in enumerate(severity_levels):
            for j, trend in enumerate(trends):
                state_idx = i * 3 + j
                self.state_names[state_idx] = f"{severity}_{trend}"
        
        self.action_names = [
            'Conservative_Management',
            'Standard_Protocol',
            'Aggressive_Treatment',
            'Maximum_Intervention'
        ]
        
        self.setup_environment()
    
    def setup_environment(self):
        """
        Setup the sepsis management MDP.
        """
        self._setup_sepsis_transitions()
        self._setup_sepsis_rewards()
        self.compute_expected_rewards()
    
    def _setup_sepsis_transitions(self):
        """
        Setup transition probabilities for sepsis management.
        """
        # Transition probabilities based on sepsis progression and treatment response
        for state in range(self.num_states):
            severity_level = state // 3  # 0: mild, 1: severe, 2: shock
            trend = state % 3           # 0: stable, 1: declining, 2: improving
            
            for action in range(self.num_actions):
                self._set_sepsis_state_transitions(state, action, severity_level, trend)
    
    def _set_sepsis_state_transitions(self, state: int, action: int, 
                                    severity_level: int, trend: int):
        """
        Set transitions for sepsis states based on treatment intensity.
        """
        # Base transition probabilities
        if severity_level == 0:  # Mild sepsis
            if trend == 0:  # Stable
                if action >= 1:  # Adequate treatment
                    self.set_transition_probability(state, action, 2, 0.6)  # Improve
                    self.set_transition_probability(state, action, 0, 0.3)  # Stay stable
                    self.set_transition_probability(state, action, 1, 0.1)  # Decline
                else:  # Conservative
                    self.set_transition_probability(state, action, 0, 0.4)
                    self.set_transition_probability(state, action, 1, 0.4)
                    self.set_transition_probability(state, action, 3, 0.2)  # Progress to severe
            elif trend == 1:  # Declining
                if action >= 2:  # Aggressive treatment
                    self.set_transition_probability(state, action, 0, 0.5)  # Stabilize
                    self.set_transition_probability(state, action, 1, 0.3)  # Continue declining
                    self.set_transition_probability(state, action, 4, 0.2)  # Progress to severe
                else:
                    self.set_transition_probability(state, action, 4, 0.6)  # Progress to severe
                    self.set_transition_probability(state, action, 1, 0.4)  # Stay declining
            else:  # Improving
                self.set_transition_probability(state, action, 2, 0.8)  # Continue improving
                self.set_transition_probability(state, action, 0, 0.2)  # Stabilize
        
        elif severity_level == 1:  # Severe sepsis
            if trend == 0:  # Stable
                if action >= 2:  # Aggressive treatment
                    self.set_transition_probability(state, action, 5, 0.5)  # Improve
                    self.set_transition_probability(state, action, 3, 0.4)  # Stay stable
                    self.set_transition_probability(state, action, 4, 0.1)  # Decline
                else:
                    self.set_transition_probability(state, action, 4, 0.5)  # Decline
                    self.set_transition_probability(state, action, 3, 0.3)  # Stay stable
                    self.set_transition_probability(state, action, 6, 0.2)  # Progress to shock
            elif trend == 1:  # Declining
                if action == 3:  # Maximum intervention
                    self.set_transition_probability(state, action, 3, 0.4)  # Stabilize
                    self.set_transition_probability(state, action, 4, 0.3)  # Continue declining
                    self.set_transition_probability(state, action, 7, 0.3)  # Progress to shock
                else:
                    self.set_transition_probability(state, action, 7, 0.7)  # Progress to shock
                    self.set_transition_probability(state, action, 4, 0.3)  # Stay declining
            else:  # Improving
                self.set_transition_probability(state, action, 5, 0.6)  # Continue improving
                self.set_transition_probability(state, action, 2, 0.3)  # Improve to mild
                self.set_transition_probability(state, action, 3, 0.1)  # Stabilize
        
        else:  # Septic shock (severity_level == 2)
            if trend == 0:  # Stable
                if action == 3:  # Maximum intervention
                    self.set_transition_probability(state, action, 8, 0.4)  # Improve
                    self.set_transition_probability(state, action, 6, 0.5)  # Stay stable
                    self.set_transition_probability(state, action, 7, 0.1)  # Decline
                else:
                    self.set_transition_probability(state, action, 7, 0.6)  # Decline
                    self.set_transition_probability(state, action, 6, 0.4)  # Stay stable
            elif trend == 1:  # Declining - critical
                if action == 3:  # Maximum intervention
                    self.set_transition_probability(state, action, 6, 0.3)  # Stabilize
                    self.set_transition_probability(state, action, 7, 0.7)  # Continue declining
                else:
                    self.set_transition_probability(state, action, 7, 1.0)  # Continue declining
            else:  # Improving
                self.set_transition_probability(state, action, 8, 0.5)  # Continue improving
                self.set_transition_probability(state, action, 5, 0.3)  # Improve to severe
                self.set_transition_probability(state, action, 6, 0.2)  # Stabilize
    
    def _setup_sepsis_rewards(self):
        """
        Setup reward function for sepsis management.
        """
        for state in range(self.num_states):
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    reward = self._calculate_sepsis_reward(state, action, next_state)
                    self.set_reward(state, action, next_state, reward)
    
    def _calculate_sepsis_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Calculate reward for sepsis management decisions.
        """
        reward = 0.0
        
        severity_level = state // 3
        next_severity = next_state // 3
        next_trend = next_state % 3
        
        # Reward for patient outcomes
        if next_severity < severity_level:  # Improvement in severity
            reward += 20.0
        elif next_severity > severity_level:  # Worsening
            reward -= 25.0
        
        # Reward for trend
        if next_trend == 2:  # Improving
            reward += 15.0
        elif next_trend == 1:  # Declining
            reward -= 20.0
        else:  # Stable
            reward += 5.0
        
        # Penalty for treatment intensity (resource usage, side effects)
        treatment_costs = {0: 0, 1: -2, 2: -5, 3: -10}
        reward += treatment_costs[action]
        
        # Severe penalty for septic shock
        if next_severity == 2:
            reward -= 15.0
        
        # Bonus for recovery to mild sepsis
        if next_severity == 0 and next_trend == 2:
            reward += 10.0
        
        return reward


def demonstrate_healthcare_mdps():
    """
    Demonstrate healthcare MDP applications.
    """
    print("Healthcare MDP Demonstrations")
    print("=" * 60)
    
    # Diabetes Treatment MDP
    print("\n1. DIABETES TREATMENT MDP")
    print("-" * 40)
    
    diabetes_mdp = DiabetesTreatmentMDP(discount_factor=0.95)
    diabetes_mdp.print_mdp_summary()
    
    # Solve using Value Iteration
    print("\nSolving with Value Iteration...")
    vi_diabetes = ValueIteration(diabetes_mdp, tolerance=1e-6)
    optimal_values, optimal_policy = vi_diabetes.solve(verbose=False)
    
    print("\nOptimal Diabetes Treatment Policy:")
    clinical_interp = diabetes_mdp.get_clinical_interpretation(optimal_policy)
    for state_name, interpretation in clinical_interp.items():
        print(f"{state_name}: {interpretation}")
    
    # Visualize results
    diabetes_mdp.visualize_reward_matrix()
    vi_diabetes.plot_convergence()
    
    # Sepsis Management MDP
    print("\n\n2. SEPSIS MANAGEMENT MDP")
    print("-" * 40)
    
    sepsis_mdp = SepsisManagementMDP(discount_factor=0.9)
    sepsis_mdp.print_mdp_summary()
    
    # Solve using Policy Iteration
    print("\nSolving with Policy Iteration...")
    pi_sepsis = PolicyIteration(sepsis_mdp, eval_tolerance=1e-6)
    sepsis_values, sepsis_policy = pi_sepsis.solve(verbose=False)
    
    print("\nOptimal Sepsis Management Policy:")
    for state in range(sepsis_mdp.num_states):
        action = sepsis_policy[state].item()
        print(f"{sepsis_mdp.state_names[state]}: {sepsis_mdp.action_names[action]}")
    
    # Visualize results
    sepsis_mdp.visualize_reward_matrix()
    pi_sepsis.plot_convergence()
    
    # Compare with Q-Learning
    print("\n\n3. REINFORCEMENT LEARNING COMPARISON")
    print("-" * 40)
    
    print("Training Q-Learning agent on diabetes MDP...")
    ql_agent = QLearning(diabetes_mdp, learning_rate=0.1, epsilon=0.9)
    ql_agent.train(num_episodes=2000, verbose=False)
    
    print("Training SARSA agent on diabetes MDP...")
    sarsa_agent = SARSA(diabetes_mdp, learning_rate=0.1, epsilon=0.9)
    sarsa_agent.train(num_episodes=2000, verbose=False)
    
    # Compare policies
    ql_policy = ql_agent.extract_policy()
    sarsa_policy = sarsa_agent.extract_policy()
    
    print("\nPolicy Comparison:")
    print(f"{'State':<25} {'Value Iter':<20} {'Q-Learning':<20} {'SARSA':<20}")
    print("-" * 85)
    
    for state in range(diabetes_mdp.num_states):
        vi_action = diabetes_mdp.action_names[optimal_policy[state].item()]
        ql_action = diabetes_mdp.action_names[ql_policy[state].item()]
        sarsa_action = diabetes_mdp.action_names[sarsa_policy[state].item()]
        
        print(f"{diabetes_mdp.state_names[state]:<25} {vi_action:<20} {ql_action:<20} {sarsa_action:<20}")
    
    # Evaluate policies
    vi_return = vi_diabetes.evaluate_policy(optimal_policy, num_episodes=1000)
    ql_return, _ = ql_agent.evaluate_policy(num_episodes=1000)
    sarsa_return, _ = sarsa_agent.evaluate_policy(num_episodes=1000, use_learned_policy=False)
    
    print(f"\nPolicy Performance Comparison:")
    print(f"Value Iteration: {vi_return:.4f}")
    print(f"Q-Learning: {ql_return:.4f}")
    print(f"SARSA: {sarsa_return:.4f}")
    
    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Q-Learning learning curve
    window_size = 100
    ql_ma = np.convolve(ql_agent.episode_returns, 
                       np.ones(window_size)/window_size, mode='valid')
    ax1.plot(range(window_size-1, len(ql_agent.episode_returns)), 
            ql_ma, 'b-', linewidth=2, label='Q-Learning')
    
    # SARSA learning curve
    sarsa_ma = np.convolve(sarsa_agent.episode_returns, 
                          np.ones(window_size)/window_size, mode='valid')
    ax1.plot(range(window_size-1, len(sarsa_agent.episode_returns)), 
            sarsa_ma, 'r-', linewidth=2, label='SARSA')
    
    ax1.axhline(y=vi_return, color='g', linestyle='--', 
               label=f'Value Iteration ({vi_return:.2f})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Learning Curves: Diabetes Treatment MDP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final performance comparison
    algorithms = ['Value Iteration', 'Q-Learning', 'SARSA']
    returns = [vi_return, ql_return, sarsa_return]
    
    ax2.bar(algorithms, returns, alpha=0.7, color=['green', 'blue', 'red'])
    ax2.set_ylabel('Average Return')
    ax2.set_title('Final Performance Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_healthcare_mdp_properties():
    """
    Analyze specific properties of healthcare MDPs.
    """
    print("\n" + "="*60)
    print("Healthcare MDP Properties Analysis")
    print("="*60)
    
    diabetes_mdp = DiabetesTreatmentMDP()
    sepsis_mdp = SepsisManagementMDP()
    
    # Analyze state transition structures
    print("\n1. State Transition Analysis")
    print("-" * 30)
    
    # Check for absorbing states
    print("Checking for absorbing states...")
    for mdp, name in [(diabetes_mdp, "Diabetes"), (sepsis_mdp, "Sepsis")]:
        absorbing_states = []
        for state in range(mdp.num_states):
            is_absorbing = True
            for action in range(mdp.num_actions):
                if mdp.transition_probs[state, action, state].item() < 0.99:
                    is_absorbing = False
                    break
            if is_absorbing:
                absorbing_states.append(state)
        
        print(f"{name} MDP absorbing states: {absorbing_states}")
    
    # Analyze reward structures
    print("\n2. Reward Structure Analysis")
    print("-" * 30)
    
    for mdp, name in [(diabetes_mdp, "Diabetes"), (sepsis_mdp, "Sepsis")]:
        rewards = mdp.expected_rewards
        print(f"\n{name} MDP Reward Statistics:")
        print(f"  Min reward: {torch.min(rewards).item():.2f}")
        print(f"  Max reward: {torch.max(rewards).item():.2f}")
        print(f"  Mean reward: {torch.mean(rewards).item():.2f}")
        print(f"  Std reward: {torch.std(rewards).item():.2f}")
    
    # Analyze convergence properties
    print("\n3. Convergence Analysis")
    print("-" * 30)
    
    for mdp, name in [(diabetes_mdp, "Diabetes"), (sepsis_mdp, "Sepsis")]:
        vi = ValueIteration(mdp, tolerance=1e-8, max_iterations=1000)
        values, policy = vi.solve(verbose=False)
        
        print(f"\n{name} MDP Convergence:")
        print(f"  Iterations to convergence: {vi.iterations_to_convergence}")
        print(f"  Final tolerance: {vi.delta_history[-1]:.8f}")
        print(f"  Converged: {vi.converged}")


if __name__ == "__main__":
    demonstrate_healthcare_mdps()
    analyze_healthcare_mdp_properties()

