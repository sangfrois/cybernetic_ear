import threading
import numpy as np

class FeatureBus:
    """
    A thread-safe container for holding and accessing feature data from all streams.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.features = {
            # Timbre Stream Features
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flux": 0.0,
            "mfccs": None,
            "timbre_texture": None,

            # Rhythm Stream Features
            "beat_salience": 0.0,
            "syncopation_index": 0.0,
            "groove_pattern": None,
            "event_density": 0.0,

            # Harmony Stream Features
            "sensory_dissonance": 0.0,
            "tonal_context": None,
            "harmonic_ratios": None,
            "spectral_contrast": None,

            # Advanced Harmonic Metrics from Biotuner
            "harmsim": 0.0,              # Harmonic Similarity
            "tenney_height": 0.0,        # Tenney Height (Harmonic Complexity)
            "subharm_tension": 0.0,      # Subharmonic Tension
            "time_resolved_harmonicity": None, # Transitional Tension vector
            "consonance": 0.0,
            "peaks_ratios_tuning": None,
            "harm_tuning": None,
            "peaks": None,
            "amps": None,
            "extended_peaks": None,
            "extended_amps": None,

            # Transitional features (CHANGE metrics from Chan 2019)
            "tension_delta": 0.0,         # Change in subharmonic tension
            "harmsim_delta": 0.0,         # Change in harmonic similarity
            "consonance_delta": 0.0,      # Change in consonance
        }

        # History for computing deltas
        self.prev_features = {
            "subharm_tension": 0.0,
            "harmsim": 0.0,
            "consonance": 0.0
        }

    def update_feature(self, key, value):
        """
        Safely updates a single feature in the bus.
        Also computes transitional (delta) features for key harmonic metrics.

        Args:
            key (str): The name of the feature to update.
            value: The new value for the feature.
        """
        with self._lock:
            if key in self.features:
                self.features[key] = value

                # Compute transitional features (deltas) per Chan 2019
                # These capture tension-resolution trajectories
                if key == "subharm_tension" and value is not None and not np.isnan(value):
                    prev = self.prev_features["subharm_tension"]
                    self.features["tension_delta"] = float(value - prev) if prev != 0 else 0.0
                    self.prev_features["subharm_tension"] = float(value)

                elif key == "harmsim" and value is not None and not np.isnan(value):
                    prev = self.prev_features["harmsim"]
                    self.features["harmsim_delta"] = float(value - prev) if prev != 0 else 0.0
                    self.prev_features["harmsim"] = float(value)

                elif key == "consonance" and value is not None and not np.isnan(value):
                    prev = self.prev_features["consonance"]
                    self.features["consonance_delta"] = float(value - prev) if prev != 0 else 0.0
                    self.prev_features["consonance"] = float(value)
            else:
                # This could be a warning or an error, depending on strictness
                print(f"Warning: Attempted to update non-existent feature '{key}'")

    def get_feature(self, key):
        """
        Safely retrieves a single feature from the bus.

        Args:
            key (str): The name of the feature to retrieve.

        Returns:
            The value of the feature, or None if not found.
        """
        with self._lock:
            return self.features.get(key)

    def get_all_features(self):
        """
        Safely retrieves a copy of the entire feature dictionary.

        Returns:
            A dictionary containing all current features.
        """
        with self._lock:
            return self.features.copy()

    def __str__(self):
        """
        Provides a string representation of the current features for printing.
        """
        with self._lock:
            # Create a formatted string of the features for readability
            feature_str = "--- Feature Bus State ---\n"
            for key, value in self.features.items():
                if isinstance(value, np.ndarray):
                    feature_str += f"- {key}: array of shape {value.shape}\n"
                elif isinstance(value, list):
                     feature_str += f"- {key}: list of length {len(value)}\n"
                elif value is None:
                    feature_str += f"- {key}: None\n"
                else:
                    feature_str += f"- {key}: {value:.4f}\n"
            feature_str += "-------------------------"
            return feature_str
