
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .ewc import ewc_penalty, compute_fisher

class PaskianTriadAgent:
    """
    The Cybernetic Core of the Musicolour system, implementing the Paskian Triad.
    """
    def __init__(self, learning_rate=0.001, slow_learning_rate=0.0001, ewc_lambda=0.1):
        self.feature_bus_size = 512 # 256 (timbre) + 128 (rhythm) + 128 (harmony)
        self.learning_rate = learning_rate
        self.slow_learning_rate = slow_learning_rate
        self.ewc_lambda = ewc_lambda

        # 1. Conversation (RL Agent)
        self.fast_network = self.build_policy_network()
        self.slow_network = self.build_policy_network()
        self.slow_network.load_state_dict(self.fast_network.state_dict())

        self.fast_optimizer = optim.Adam(self.fast_network.parameters(), lr=self.learning_rate)
        self.slow_optimizer = optim.Adam(self.slow_network.parameters(), lr=self.slow_learning_rate)

        # 2. Boredom (EWC)
        self.fisher_matrix = None
        self.prev_weights = None

        # For R_stasis and R_novelty calculation
        self.prev_state = None
        self.prev_action = None
        self.feature_history = []

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

        return state.detach(), attention_weights.detach()

    def train_step(self, state, action, reward):
        """
        Performs a single training step for the RL agent using the REINFORCE algorithm.
        """
        # --- Fast Network Update ---
        fast_log_prob = torch.log(self.fast_network(state) + 1e-9)
        fast_loss = (-fast_log_prob * reward).sum()
        
        self.fast_optimizer.zero_grad()
        fast_loss.backward()
        self.fast_optimizer.step()

        # --- Slow Network Update ---
        slow_log_prob = torch.log(self.slow_network(state) + 1e-9)
        slow_loss = (-slow_log_prob * reward).sum()
        if self.fisher_matrix is not None:
            ewc_loss = ewc_penalty(self.slow_network, self.fisher_matrix, self.prev_weights, self.ewc_lambda)
            slow_loss += ewc_loss
            
        self.slow_optimizer.zero_grad()
        slow_loss.backward()
        self.slow_optimizer.step()

    def consolidate(self):
        """
        Computes the Fisher Information Matrix and stores the current weights.
        """
        if not self.feature_history:
            return

        # Convert feature history to a dataset
        dataset = torch.cat(self.feature_history, dim=0)
        
        self.fisher_matrix = compute_fisher(self.slow_network, dataset)
        self.prev_weights = {name: param.clone().detach() for name, param in self.slow_network.named_parameters()}

    def calculate_reward(self, state, action, feature_bus):
        """
        Calculates the reward for the current state and action, based on paper.md.
        """
        # R_novelty: Reward for mutual innovation, using spectral flux as a proxy for novelty
        spectral_flux = feature_bus.get_feature("spectral_flux") or 0.0
        action_novelty = torch.mean(torch.abs(action - self.prev_action)) if self.prev_action is not None else 0.0
        r_novelty = spectral_flux * action_novelty * 10.0 # Scale the reward

        self.feature_history.append(state)
        if len(self.feature_history) > 10: # Keep a fixed history size
            self.feature_history.pop(0)

        # R_tension: Reward for anticipatory engagement, using subharmonic tension
        r_tension = feature_bus.get_feature("subharm_tension") or 0.0
        if np.isnan(r_tension):
            r_tension = 0.0

        # R_stasis: Penalty for inaction
        r_stasis = 0.0
        if self.prev_state is not None and self.prev_action is not None:
            state_diff = torch.mean(torch.abs(state - self.prev_state))
            action_diff = torch.mean(torch.abs(action - self.prev_action))
            if state_diff < 0.01 and action_diff < 0.01:
                r_stasis = -5.0 # Significantly increased penalty

        self.prev_state = state
        self.prev_action = action

        # Combine the rewards with weights
        total_reward = (0.6 * r_novelty) + (0.2 * r_tension) + (0.2 * r_stasis)
        
        if np.isnan(total_reward):
            total_reward = 0.0
            
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
