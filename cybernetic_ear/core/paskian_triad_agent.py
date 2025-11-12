
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .ewc import ewc_penalty, compute_fisher

class PaskianTriadAgent:
    """
    The Cybernetic Core of the Musicolour system, implementing the Paskian Triad.
    """
    def __init__(self, learning_rate=0.05, slow_learning_rate=0.001, ewc_lambda=0.1):
        self.feature_bus_size = 512 # 256 (timbre) + 128 (rhythm) + 128 (harmony)
        self.learning_rate = learning_rate
        self.slow_learning_rate = slow_learning_rate
        self.ewc_lambda = ewc_lambda

        # 1. Conversation (RL Agent)
        self.fast_network = self.build_policy_network()
        self.slow_network = self.build_policy_network()
        # Don't copy weights - let them be different to encourage divergence
        # self.slow_network.load_state_dict(self.fast_network.state_dict())

        self.fast_optimizer = optim.Adam(self.fast_network.parameters(), lr=self.learning_rate)
        self.slow_optimizer = optim.Adam(self.slow_network.parameters(), lr=self.slow_learning_rate)

        # 2. Boredom (EWC)
        self.fisher_matrix = None
        self.prev_weights = None

        # For R_stasis and R_novelty calculation
        self.prev_state = None
        self.prev_action = None
        self.feature_history = []

        # Dashboard state tracking
        self.stasis_counter = 0  # Boredom accumulator
        self.last_reward_components = {
            'r_novelty': 0.0,
            'r_tension': 0.0,
            'r_stasis': 0.0,
            'total_reward': 0.0
        }
        self.consolidation_count = 0
        self.last_state_diff = 0.0
        self.last_action_diff = 0.0

    def build_policy_network(self):
        """
        Builds the policy network for the RL agent.
        The network takes the feature bus as input and outputs an attention vector over the streams.
        """
        return nn.Sequential(
            nn.Linear(self.feature_bus_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3), # 3 streams: timbre, rhythm, harmony
            nn.Softmax(dim=-1)
        )

    def get_action(self, feature_bus):
        """
        Gets an action (attention vector) from the policy network.
        """
        # Convert feature bus to a tensor
        state = self.feature_bus_to_tensor(feature_bus)

        # Get action from the combined fast and slow networks
        fast_action = self.fast_network(state)
        slow_action = self.slow_network(state)
        attention_weights = (fast_action + slow_action) / 2

        # Add exploration noise to prevent getting stuck
        # More noise when stasis counter is high (agent is bored) OR if stuck

        # Check if attention is too concentrated (stuck on one stream)
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9))
        max_entropy = torch.log(torch.tensor(3.0))  # log(3) for uniform distribution
        concentration = 1.0 - (attention_entropy / max_entropy)  # 0=uniform, 1=concentrated

        # If too concentrated (>80% on one stream), force more uniform
        if concentration > 0.8:  # Very stuck!
            # Blend with uniform distribution
            uniform = torch.ones_like(attention_weights) / 3.0
            attention_weights = 0.7 * uniform + 0.3 * attention_weights
        elif self.stasis_counter > 10 or concentration > 0.6:  # Stuck or bored
            noise_scale = 0.3 * (1 + self.stasis_counter / 30.0)
            noise = torch.randn_like(attention_weights) * noise_scale
            attention_weights = attention_weights + noise
            # Re-normalize to sum to 1
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

        return state.detach(), attention_weights.detach()

    def train_step(self, state, action, reward):
        """
        Performs a single training step for the RL agent using the REINFORCE algorithm.
        """
        # --- Fast Network Update ---
        fast_output = self.fast_network(state)
        fast_log_prob = torch.log(fast_output + 1e-9)

        # Add entropy bonus to prevent collapsing to single stream
        entropy = -torch.sum(fast_output * fast_log_prob)
        entropy_weight = 1.0  # Strong encouragement to explore (was 0.1)

        fast_loss = (-fast_log_prob * reward).sum() - entropy_weight * entropy

        self.fast_optimizer.zero_grad()
        fast_loss.backward()
        # Clip gradients to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.fast_network.parameters(), max_norm=1.0)
        self.fast_optimizer.step()

        # --- Slow Network Update ---
        slow_output = self.slow_network(state)
        slow_log_prob = torch.log(slow_output + 1e-9)
        slow_loss = (-slow_log_prob * reward).sum()

        if self.fisher_matrix is not None:
            ewc_loss = ewc_penalty(self.slow_network, self.fisher_matrix, self.prev_weights, self.ewc_lambda)
            slow_loss += ewc_loss

        self.slow_optimizer.zero_grad()
        slow_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.slow_network.parameters(), max_norm=1.0)
        self.slow_optimizer.step()

    def consolidate(self):
        """
        Computes the Fisher Information Matrix and stores the current weights.
        """
        if not self.feature_history:
            return

        try:
            # Convert feature history to a dataset
            # feature_history contains tensors of shape (1, 512)
            dataset = torch.cat(self.feature_history, dim=0)  # Shape: (N, 512)

            print(f"[EWC] Consolidating with {len(self.feature_history)} states...")
            self.fisher_matrix = compute_fisher(self.slow_network, dataset)
            self.prev_weights = {name: param.clone().detach() for name, param in self.slow_network.named_parameters()}
            print(f"[EWC] Consolidation complete! Memory trace established.")
        except Exception as e:
            print(f"[EWC] Error during consolidation: {e}")
            import traceback
            traceback.print_exc()

    def calculate_reward(self, state, action, feature_bus):
        """
        Calculates the reward for the current state and action, based on paper.md.
        """
        # R_novelty: Reward for mutual innovation, using spectral flux as a proxy for novelty
        spectral_flux = feature_bus.get_feature("spectral_flux") or 0.0
        action_novelty = torch.mean(torch.abs(action - self.prev_action)).item() if self.prev_action is not None else 0.01

        # Combined novelty: reward both environmental novelty AND behavioral novelty
        # But keep rewards in reasonable range (0-10)
        r_novelty = (spectral_flux + action_novelty * 10.0) * 0.5

        self.feature_history.append(state)
        if len(self.feature_history) > 10: # Keep a fixed history size
            self.feature_history.pop(0)

        # R_tension: Reward for anticipatory engagement using TRANSITIONAL harmony
        # Per Chan 2019: tension-resolution trajectories matter more than absolute values
        tension_delta = feature_bus.get_feature("tension_delta") or 0.0
        harmsim_delta = feature_bus.get_feature("harmsim_delta") or 0.0

        # Reward CHANGES in harmony:
        # - Rising tension (tension_delta > 0): anticipation, interesting
        # - Falling tension (tension_delta < 0): resolution, satisfying
        # - Rising harmonic similarity: increasing coherence
        r_tension = abs(tension_delta) * 2.0 + abs(harmsim_delta) * 2.0

        if np.isnan(r_tension):
            r_tension = 0.0

        # R_stasis: Penalty for inaction
        r_stasis = 0.0
        if self.prev_state is not None and self.prev_action is not None:
            state_diff = torch.mean(torch.abs(state - self.prev_state)).item()
            action_diff = torch.mean(torch.abs(action - self.prev_action)).item()

            # Track diffs for debugging
            self.last_state_diff = state_diff
            self.last_action_diff = action_diff

            # Stasis detection: Simpler logic - if action barely changes, increment
            # ONLY decay if there's significant change in BOTH state and action

            # If action is stuck (< 0.02), always increment
            if action_diff < 0.02:
                stasis_factor = max(0, 1 - (action_diff / 0.02))
                r_stasis = -10.0 * stasis_factor
                self.stasis_counter += 1
            # If state is stuck too (< 0.01), increment (silence or repetition)
            elif state_diff < 0.01:
                r_stasis = -5.0
                self.stasis_counter += 1
            # Only decay if BOTH are significantly changing (real exploration)
            elif state_diff > 0.05 and action_diff > 0.05:
                r_stasis = 0.0
                self.stasis_counter = max(0, self.stasis_counter - 1)
            # Otherwise, maintain current count (no change)
            else:
                r_stasis = 0.0

        self.prev_state = state
        self.prev_action = action

        # Combine the rewards with weights
        # Increase stasis weight to force exploration
        total_reward = (0.4 * r_novelty) + (0.2 * r_tension) + (0.4 * r_stasis)

        if np.isnan(total_reward):
            total_reward = 0.0

        # Store reward components for dashboard
        self.last_reward_components = {
            'r_novelty': float(r_novelty) if isinstance(r_novelty, (int, float)) else float(r_novelty.item()) if hasattr(r_novelty, 'item') else 0.0,
            'r_tension': float(r_tension),
            'r_stasis': float(r_stasis),
            'total_reward': float(total_reward)
        }

        # Check if we should consolidate (boredom threshold)
        if self.stasis_counter >= 30:
            self.consolidate()
            self.stasis_counter = 0
            self.consolidation_count += 1

        return torch.tensor(total_reward, dtype=torch.float32).detach()

    def _normalize_and_pad(self, data, size):
        """Normalizes and pads/truncates a feature vector."""
        if data is None:
            return np.zeros(size)
        
        data = np.ravel(data)
        
        # Normalize
        if np.max(data) > 0:
            data = data / np.max(data)
            
        # Pad or truncate
        if len(data) > size:
            data = data[:size]
        else:
            data = np.pad(data, (0, size - len(data)), 'constant')
            
        return data

    def feature_bus_to_tensor(self, feature_bus):
        """
        Converts the feature bus to a structured PyTorch tensor.
        """
        # Define the structure of the feature vector
        stream_sizes = {
            "timbre": 256,
            "rhythm": 128,
            "harmony": 128
        }
        
        # Process features for each stream
        timbre_features = np.concatenate([
            self._normalize_and_pad(feature_bus.get_feature("spectral_centroid"), 1),
            self._normalize_and_pad(feature_bus.get_feature("spectral_flatness"), 1),
            self._normalize_and_pad(feature_bus.get_feature("spectral_rolloff"), 1),
            self._normalize_and_pad(feature_bus.get_feature("spectral_flux"), 1),
            self._normalize_and_pad(feature_bus.get_feature("mfccs"), 13),
            self._normalize_and_pad(feature_bus.get_feature("timbre_texture"), 239)
        ])
        
        rhythm_features = np.concatenate([
            self._normalize_and_pad(feature_bus.get_feature("beat_salience"), 1),
            self._normalize_and_pad(feature_bus.get_feature("syncopation_index"), 1),
            self._normalize_and_pad(feature_bus.get_feature("groove_pattern"), 1),
            self._normalize_and_pad(feature_bus.get_feature("event_density"), 125)
        ])
        
        harmony_features = np.concatenate([
            self._normalize_and_pad(feature_bus.get_feature("sensory_dissonance"), 1),
            self._normalize_and_pad(feature_bus.get_feature("tonal_context"), 24),
            self._normalize_and_pad(feature_bus.get_feature("subharm_tension"), 1),
            self._normalize_and_pad(feature_bus.get_feature("harmsim"), 102)
        ])
        
        # Concatenate all stream features
        feature_vector = np.concatenate([timbre_features, rhythm_features, harmony_features])

        return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    def calculate_plasticity(self):
        """
        Calculates the plasticity (divergence) between fast and slow networks.
        Higher values indicate the fast network is learning new patterns.
        """
        divergence = 0.0
        param_count = 0

        for (name_fast, param_fast), (name_slow, param_slow) in zip(
            self.fast_network.named_parameters(),
            self.slow_network.named_parameters()
        ):
            if param_fast.requires_grad:
                divergence += torch.mean(torch.abs(param_fast - param_slow)).item()
                param_count += 1

        return divergence / param_count if param_count > 0 else 0.0

    def get_agent_state(self, feature_bus):
        """
        Returns the full agent state for dashboard visualization.
        """
        # Get current attention from both networks
        state = self.feature_bus_to_tensor(feature_bus)
        fast_attention = self.fast_network(state).detach().numpy().flatten()
        slow_attention = self.slow_network(state).detach().numpy().flatten()

        return {
            'fast_attention': fast_attention.tolist(),
            'slow_attention': slow_attention.tolist(),
            'reward_components': self.last_reward_components,
            'stasis_counter': self.stasis_counter,
            'plasticity': self.calculate_plasticity(),
            'consolidation_count': self.consolidation_count,
            'ewc_active': self.fisher_matrix is not None,
            'state_diff': self.last_state_diff,
            'action_diff': self.last_action_diff
        }
