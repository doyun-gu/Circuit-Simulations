"""
Reinforcement Learning-Based Hybrid Phasor Method Switching

This module implements RL agents for intelligent method selection between
Instantaneous Dynamic Phasor (IDP) and Generalized Averaging.

Key Advantages over Classification:
1. Sequential decision-making (considers simulation history)
2. Multi-objective optimization (accuracy + computational cost)
3. Handles class imbalance naturally through reward shaping
4. Can learn complex switching policies

Author: Doyun Gu
Reference: APEC 2026 Submission
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
from collections import deque
import random


class Method(Enum):
    """Available simulation methods"""
    INSTANTANEOUS = 0
    AVERAGED = 1


@dataclass
class State:
    """
    RL State representation for method switching.
    
    Encodes circuit condition at current timestep for decision-making.
    """
    # Time features
    time: float                    # Current simulation time
    time_normalized: float         # t / t_end
    
    # Signal magnitudes
    i_s_mag: float                 # |i_s| current magnitude
    v_o_mag: float                 # |v_o| voltage magnitude
    
    # Derivatives (transient indicators)
    di_dt: float                   # First derivative of current
    dv_dt: float                   # First derivative of voltage
    d2i_dt2: float                 # Second derivative (acceleration)
    d2v_dt2: float                 # Second derivative
    
    # Envelope features
    envelope_variation: float      # Rate of envelope change
    envelope_bandwidth: float      # Bandwidth relative to carrier
    
    # Circuit parameters
    freq_ratio: float              # ω_s / ω_r (operating vs resonant)
    Q_factor: float                # Quality factor
    
    # History features
    time_since_switch: float       # Time since last method switch
    time_since_transient: float    # Time since last detected transient
    current_method: int            # Currently active method (0 or 1)
    
    # Recent errors (rolling window)
    recent_error_inst: float       # Recent error with instantaneous
    recent_error_avg: float        # Recent error with averaging
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input"""
        return np.array([
            self.time_normalized,
            self.i_s_mag,
            self.v_o_mag,
            self.di_dt,
            self.dv_dt,
            self.d2i_dt2,
            self.d2v_dt2,
            self.envelope_variation,
            self.envelope_bandwidth,
            self.freq_ratio,
            self.Q_factor,
            self.time_since_switch,
            self.time_since_transient,
            self.current_method,
            self.recent_error_inst,
            self.recent_error_avg
        ], dtype=np.float32)
    
    @staticmethod
    def dim() -> int:
        """State dimension"""
        return 16


@dataclass
class RewardConfig:
    """
    Configuration for reward function.
    
    Allows tuning the balance between accuracy and computational cost.
    """
    # Accuracy rewards
    accuracy_weight: float = 1.0          # Weight for accuracy term
    error_penalty_scale: float = 100.0    # Multiplier for error penalty
    
    # Computational cost
    cost_weight: float = 0.1              # Weight for computational cost
    inst_cost: float = 1.0                # Relative cost of instantaneous
    avg_cost: float = 0.3                 # Relative cost of averaging
    
    # Switching penalty (encourage stability)
    switch_penalty: float = 0.05          # Penalty for switching methods
    
    # Bonus for correct difficult decisions
    minority_class_bonus: float = 2.0     # Extra reward for correct averaging selection


class HybridSwitchingEnv:
    """
    RL Environment for hybrid phasor method switching.
    
    Simulates circuit and provides rewards based on simulation accuracy
    and computational efficiency.
    
    Action Space: {0: Use Instantaneous, 1: Use Averaging}
    
    Observation Space: State vector (16 dimensions)
    
    Reward: Combination of:
        - Negative error (accuracy)
        - Negative computational cost
        - Switching penalty
    """
    
    def __init__(self, 
                 circuit,
                 omega_s: float,
                 t_span: Tuple[float, float],
                 dt: float = 1e-6,
                 reward_config: RewardConfig = None):
        """
        Initialize environment.
        
        Args:
            circuit: RLCCircuit instance
            omega_s: Operating frequency
            t_span: (t_start, t_end) simulation span
            dt: Timestep for decisions
            reward_config: Reward function configuration
        """
        self.circuit = circuit
        self.omega_s = omega_s
        self.t_start, self.t_end = t_span
        self.dt = dt
        self.reward_config = reward_config or RewardConfig()
        
        # State tracking
        self.current_time = self.t_start
        self.current_method = Method.INSTANTANEOUS
        self.last_switch_time = self.t_start
        self.last_transient_time = self.t_start
        
        # History buffers
        self.state_history = deque(maxlen=100)
        self.error_history_inst = deque(maxlen=50)
        self.error_history_avg = deque(maxlen=50)
        
        # Ground truth (computed once with high-fidelity method)
        self.ground_truth = None
        
    def reset(self) -> State:
        """Reset environment for new episode"""
        self.current_time = self.t_start
        self.current_method = Method.INSTANTANEOUS
        self.last_switch_time = self.t_start
        self.last_transient_time = self.t_start
        
        self.state_history.clear()
        self.error_history_inst.clear()
        self.error_history_avg.clear()
        
        # Compute ground truth for this episode
        self._compute_ground_truth()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Take action and advance simulation.
        
        Args:
            action: 0 = Instantaneous, 1 = Averaging
            
        Returns:
            next_state: New state after action
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        method = Method(action)
        
        # Check if switching
        switched = (method != self.current_method)
        if switched:
            self.last_switch_time = self.current_time
        self.current_method = method
        
        # Simulate one timestep with chosen method
        result, error = self._simulate_step(method)
        
        # Update histories
        if method == Method.INSTANTANEOUS:
            self.error_history_inst.append(error)
        else:
            self.error_history_avg.append(error)
        
        # Compute reward
        reward = self._compute_reward(method, error, switched)
        
        # Advance time
        self.current_time += self.dt
        done = self.current_time >= self.t_end
        
        # Get new state
        next_state = self._get_state()
        
        info = {
            'error': error,
            'method': method.name,
            'switched': switched,
            'time': self.current_time
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> State:
        """Construct current state observation"""
        # Get current circuit values (simplified - would use actual simulation)
        i_s, v_o = self._get_current_values()
        
        # Compute derivatives from history
        di_dt, dv_dt = self._compute_derivatives()
        d2i_dt2, d2v_dt2 = self._compute_second_derivatives()
        
        # Compute envelope features
        env_var, env_bw = self._compute_envelope_features()
        
        # Circuit parameters
        omega_r = self.circuit.params.omega_r
        Q = self.circuit.params.Q
        
        # Recent errors
        recent_err_inst = np.mean(self.error_history_inst) if self.error_history_inst else 0.0
        recent_err_avg = np.mean(self.error_history_avg) if self.error_history_avg else 0.0
        
        state = State(
            time=self.current_time,
            time_normalized=self.current_time / self.t_end,
            i_s_mag=np.abs(i_s),
            v_o_mag=np.abs(v_o),
            di_dt=di_dt,
            dv_dt=dv_dt,
            d2i_dt2=d2i_dt2,
            d2v_dt2=d2v_dt2,
            envelope_variation=env_var,
            envelope_bandwidth=env_bw,
            freq_ratio=self.omega_s / omega_r,
            Q_factor=Q,
            time_since_switch=self.current_time - self.last_switch_time,
            time_since_transient=self.current_time - self.last_transient_time,
            current_method=self.current_method.value,
            recent_error_inst=recent_err_inst,
            recent_error_avg=recent_err_avg
        )
        
        self.state_history.append(state)
        return state
    
    def _compute_reward(self, method: Method, error: float, switched: bool) -> float:
        """
        Compute reward for this step.
        
        Reward = accuracy_term + cost_term + switch_penalty
        """
        cfg = self.reward_config
        
        # Accuracy term (negative error)
        accuracy_reward = -cfg.accuracy_weight * cfg.error_penalty_scale * error
        
        # Computational cost term
        cost = cfg.inst_cost if method == Method.INSTANTANEOUS else cfg.avg_cost
        cost_reward = -cfg.cost_weight * cost
        
        # Switching penalty
        switch_reward = -cfg.switch_penalty if switched else 0.0
        
        # Bonus for correctly using averaging (minority case)
        # This helps with class imbalance
        if method == Method.AVERAGED and error < 0.01:  # Low error with averaging
            bonus = cfg.minority_class_bonus
        else:
            bonus = 0.0
        
        total_reward = accuracy_reward + cost_reward + switch_reward + bonus
        
        return total_reward
    
    def _compute_ground_truth(self):
        """Compute high-fidelity ground truth for error calculation"""
        # Use very fine time-domain simulation as ground truth
        pass  # Implementation depends on circuit solver
    
    def _simulate_step(self, method: Method) -> Tuple[Dict, float]:
        """Simulate one step and return result + error"""
        # Simplified - actual implementation would call circuit solvers
        result = {}
        error = np.random.exponential(0.01)  # Placeholder
        return result, error
    
    def _get_current_values(self) -> Tuple[complex, complex]:
        """Get current i_s and v_o values"""
        return 0.1 + 0j, 5.0 + 0j  # Placeholder
    
    def _compute_derivatives(self) -> Tuple[float, float]:
        """Compute first derivatives from history"""
        if len(self.state_history) < 2:
            return 0.0, 0.0
        s1, s2 = self.state_history[-2], self.state_history[-1]
        di_dt = (s2.i_s_mag - s1.i_s_mag) / self.dt
        dv_dt = (s2.v_o_mag - s1.v_o_mag) / self.dt
        return di_dt, dv_dt
    
    def _compute_second_derivatives(self) -> Tuple[float, float]:
        """Compute second derivatives"""
        if len(self.state_history) < 3:
            return 0.0, 0.0
        # Central difference approximation
        s1, s2, s3 = list(self.state_history)[-3:]
        d2i = (s3.i_s_mag - 2*s2.i_s_mag + s1.i_s_mag) / (self.dt ** 2)
        d2v = (s3.v_o_mag - 2*s2.v_o_mag + s1.v_o_mag) / (self.dt ** 2)
        return d2i, d2v
    
    def _compute_envelope_features(self) -> Tuple[float, float]:
        """Compute envelope variation and bandwidth"""
        if len(self.state_history) < 10:
            return 0.0, 0.0
        recent = list(self.state_history)[-10:]
        i_mags = [s.i_s_mag for s in recent]
        env_var = np.std(i_mags) / (np.mean(i_mags) + 1e-10)
        env_bw = env_var * self.omega_s  # Approximate bandwidth
        return env_var, env_bw


class DQNAgent:
    """
    Deep Q-Network Agent for method switching.
    
    Uses experience replay and target network for stable learning.
    """
    
    def __init__(self,
                 state_dim: int = 16,
                 action_dim: int = 2,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (2: inst/avg)
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_*: Exploration parameters
            buffer_size: Replay buffer size
            batch_size: Training batch size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Networks (would use PyTorch/TensorFlow in production)
        self.q_network = self._build_network(hidden_dims)
        self.target_network = self._build_network(hidden_dims)
        self._update_target_network()
        
        # Training tracking
        self.training_step = 0
        self.update_target_every = 100
    
    def _build_network(self, hidden_dims: List[int]) -> Dict:
        """Build Q-network (simplified representation)"""
        # In production, this would be a PyTorch/TF neural network
        return {
            'hidden_dims': hidden_dims,
            'weights': [np.random.randn(self.state_dim, hidden_dims[0]) * 0.1]
        }
    
    def _update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network = self.q_network.copy()
    
    def select_action(self, state: State, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
            
        Returns:
            action: 0 (Instantaneous) or 1 (Averaging)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Get Q-values (simplified)
        state_array = state.to_array()
        q_values = self._compute_q_values(state_array)
        return int(np.argmax(q_values))
    
    def _compute_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for state (simplified)"""
        # In production, forward pass through neural network
        # Here using simple linear approximation
        return np.random.randn(self.action_dim)
    
    def store_transition(self, state: State, action: int, 
                        reward: float, next_state: State, done: bool):
        """Store transition in replay buffer"""
        self.memory.append((
            state.to_array(),
            action,
            reward,
            next_state.to_array(),
            done
        ))
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            loss: Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Compute target Q-values (simplified)
        # In production: target = r + gamma * max(Q_target(s'))
        loss = 0.0  # Placeholder
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self._update_target_network()
        
        return loss
    
    def save(self, path: str):
        """Save model weights"""
        np.save(path, {'q_network': self.q_network, 
                       'epsilon': self.epsilon})
    
    def load(self, path: str):
        """Load model weights"""
        data = np.load(path, allow_pickle=True).item()
        self.q_network = data['q_network']
        self.epsilon = data['epsilon']


@dataclass
class SwitchingStrategy:
    """
    Complete switching strategy configuration.
    
    Defines when and how to switch between methods.
    """
    name: str
    description: str
    
    # Primary trigger conditions
    use_ml: bool = True                    # Use ML/RL for decisions
    use_rules: bool = False                # Use rule-based backup
    
    # Rule-based thresholds (backup)
    derivative_threshold: float = 1e6      # |di/dt| or |dv/dt| threshold
    envelope_variation_threshold: float = 0.1
    early_time_threshold: float = 50e-6    # Use inst for t < threshold
    
    # Computational budget
    max_switches_per_ms: int = 10          # Limit switching frequency
    min_time_between_switches: float = 1e-6
    
    # Hybrid settings
    blend_window: float = 5e-6             # Smooth transition window
    
    def should_use_instantaneous(self, state: State, 
                                 ml_prediction: Optional[int] = None) -> bool:
        """
        Determine if instantaneous method should be used.
        
        Args:
            state: Current circuit state
            ml_prediction: ML model prediction (0=inst, 1=avg)
            
        Returns:
            True if instantaneous should be used
        """
        # ML prediction takes priority
        if self.use_ml and ml_prediction is not None:
            return ml_prediction == 0
        
        # Rule-based backup
        if self.use_rules:
            # Early in simulation
            if state.time < self.early_time_threshold:
                return True
            
            # High derivatives (transient)
            if abs(state.di_dt) > self.derivative_threshold:
                return True
            if abs(state.dv_dt) > self.derivative_threshold:
                return True
            
            # High envelope variation
            if state.envelope_variation > self.envelope_variation_threshold:
                return True
        
        # Default to averaging
        return False


# Predefined strategies
STRATEGIES = {
    'aggressive_inst': SwitchingStrategy(
        name='Aggressive Instantaneous',
        description='Favor instantaneous for maximum accuracy',
        derivative_threshold=1e5,
        envelope_variation_threshold=0.05,
        early_time_threshold=100e-6
    ),
    'balanced': SwitchingStrategy(
        name='Balanced',
        description='Balance accuracy and computational cost',
        derivative_threshold=1e6,
        envelope_variation_threshold=0.1,
        early_time_threshold=50e-6
    ),
    'efficient': SwitchingStrategy(
        name='Efficient',
        description='Favor averaging for computational efficiency',
        derivative_threshold=5e6,
        envelope_variation_threshold=0.2,
        early_time_threshold=20e-6
    ),
    'ml_only': SwitchingStrategy(
        name='ML Only',
        description='Pure ML-based decisions',
        use_ml=True,
        use_rules=False
    ),
    'hybrid': SwitchingStrategy(
        name='Hybrid ML + Rules',
        description='ML with rule-based fallback',
        use_ml=True,
        use_rules=True
    )
}


def train_rl_agent(circuit, 
                   n_episodes: int = 1000,
                   reward_config: RewardConfig = None) -> DQNAgent:
    """
    Train RL agent for method switching.
    
    Args:
        circuit: RLCCircuit to train on
        n_episodes: Number of training episodes
        reward_config: Reward function configuration
        
    Returns:
        Trained DQN agent
    """
    # Create environment
    env = HybridSwitchingEnv(
        circuit=circuit,
        omega_s=580e3,
        t_span=(0, 0.2e-3),
        reward_config=reward_config
    )
    
    # Create agent
    agent = DQNAgent(
        state_dim=State.dim(),
        action_dim=2,
        hidden_dims=[64, 64, 32]
    )
    
    # Training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent


# Example usage and recommendations
if __name__ == "__main__":
    print("=" * 70)
    print("RL-Based Hybrid Phasor Method Switching")
    print("=" * 70)
    
    print("\n📊 Recommended Approach Based on Your Notebook Results:\n")
    
    print("1. CLASS IMBALANCE ISSUE:")
    print("   Your data: 95.7% Instantaneous, 4.3% Averaging")
    print("   → Standard classification fails (always predicts majority)")
    print("   → Solution: RL with shaped rewards for minority class")
    
    print("\n2. FEATURE IMPORTANCE INSIGHTS:")
    print("   Top predictors: derivatives (di/dt, d²i/dt²) and envelope variation")
    print("   → These indicate transient vs steady-state")
    print("   → Use as primary RL state features")
    
    print("\n3. RECOMMENDED SWITCHING CRITERIA:")
    print("   Use INSTANTANEOUS when:")
    print("     • |di/dt| > threshold (fast current change)")
    print("     • |d²v/dt²| > threshold (acceleration)")
    print("     • t < 50µs (early transient)")
    print("     • Envelope variation > 10%")
    print("   Use AVERAGING when:")
    print("     • Quasi-steady-state")
    print("     • Low derivatives")
    print("     • Computational budget limited")
    
    print("\n4. RL REWARD SHAPING:")
    print("   reward = -error - 0.1*cost + 2.0*correct_averaging_bonus")
    print("   This addresses class imbalance by rewarding correct minority decisions")
