import librosa
import numpy as np
from .base_stream import BaseStream
from scipy.spatial.distance import pdist, squareform

from biotuner.biotuner_object import compute_biotuner as Biotuner
import torch
from cybernetic_ear.core.biotuner_object import biotuner_realtime

class HarmonyStream(BaseStream):
    """
    Analyzes the harmonic and tensional features of the audio stream,
    as described in the paper.md, using the Biotuner submodule.
    """
    def __init__(self, rate=22050, buffer_size_seconds=4, enabled=True):
        self.rate = rate
        self.buffer_size = rate * buffer_size_seconds
        self.audio_buffer = np.array([], dtype=np.float32)
        self.enabled = enabled
        # self.biotuner = Biotuner(self.rate) # No longer needed as biotuner_realtime handles instantiation

    def process_chunk(self, chunk, feature_bus):
        """
        Processes a chunk to extract harmonic features and updates the feature bus.
        This stream is stateful and processes audio in larger buffered segments.
        """
        if not self.enabled:
            return
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

        if len(self.audio_buffer) >= self.buffer_size:
            buffer = self.audio_buffer

            # --- Sensory Dissonance (Plomp-Levelt) ---
            sensory_dissonance = self.calculate_sensory_dissonance(buffer)
            feature_bus.update_feature("sensory_dissonance", sensory_dissonance)

            # --- Tonal Context (Krumhansl-Schmuckler) ---
            tonal_context = self.calculate_tonal_context(buffer)
            feature_bus.update_feature("tonal_context", tonal_context)

            # --- Advanced Harmonic Analysis (Biotuner) ---
            self.run_biotuner_analysis(buffer, feature_bus)

            # Slide the buffer window
            self.audio_buffer = self.audio_buffer[len(chunk):]

    def run_biotuner_analysis(self, audio_buffer, feature_bus):
        """
        Runs the Biotuner analysis on the audio buffer and updates the feature bus.
        """
        try:
            peaks, extended_peaks, metrics, tuning, harm_tuning, amps, extended_amps = biotuner_realtime(
                audio_buffer,
                self.rate,
                n_peaks=10,
                peaks_function="harmonic_recurrence", # More suitable for audio
                min_freq=20, # Audible range
                max_freq=10000, # Audible range
                precision=0.1,
                n_harm_extended=3,
                n_harm_subharm=3,
                delta_lim=250,
            )
            
            # Update feature bus with all relevant metrics
            feature_bus.update_feature("harmsim", metrics["harmsim"])
            feature_bus.update_feature("tenney_height", metrics["tenney"])
            feature_bus.update_feature("subharm_tension", metrics["subharm_tension"][0])
            feature_bus.update_feature("consonance", metrics["cons"])
            feature_bus.update_feature("peaks_ratios_tuning", tuning)
            feature_bus.update_feature("harm_tuning", harm_tuning)
            feature_bus.update_feature("peaks", peaks)
            feature_bus.update_feature("amps", amps)
            feature_bus.update_feature("extended_peaks", extended_peaks)
            feature_bus.update_feature("extended_amps", extended_amps)

        except Exception as e:
            print(f"Error in Biotuner analysis: {e}")

    def calculate_tonal_context(self, audio_buffer):
        """
        Calculates the tonal context of an audio buffer using the Krumhansl-Schmuckler key-finding algorithm.
        """
        # 1. Compute a Pitch Class Profile (PCP)
        chromagram = librosa.feature.chroma_stft(y=audio_buffer, sr=self.rate)
        pcp = np.mean(chromagram, axis=1)

        # 2. Define Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # 3. Calculate correlation between PCP and key profiles
        major_correlations = []
        minor_correlations = []
        for i in range(12):
            major_correlations.append(np.corrcoef(pcp, np.roll(major_profile, i))[0, 1])
            minor_correlations.append(np.corrcoef(pcp, np.roll(minor_profile, i))[0, 1])

        return np.array(major_correlations + minor_correlations)

    def calculate_sensory_dissonance(self, audio_buffer):
        """
        Calculates the sensory dissonance of an audio buffer based on the Plomp-Levelt model.
        """
        stft = np.abs(librosa.stft(audio_buffer))
        total_dissonance = 0
        n_frames = stft.shape[1]

        for i in range(n_frames):
            frame = stft[:, i]
            peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)

            if len(peaks) < 2:
                continue

            peak_freqs = librosa.fft_frequencies(sr=self.rate, n_fft=(stft.shape[0] - 1) * 2)[peaks]
            peak_amps = frame[peaks]

            # Plomp-Levelt dissonance curve parameters
            d_max = 1.0
            s1 = 0.24
            s2 = 1.48
            
            freq_diffs = squareform(pdist(peak_freqs[:, np.newaxis], 'euclidean'))
            
            def dissonance(f_diff):
                return d_max * (np.exp(-s1 * f_diff) - np.exp(-s2 * f_diff))

            dissonance_matrix = dissonance(freq_diffs)
            
            amp_products = np.outer(peak_amps, peak_amps)
            
            frame_dissonance = np.sum(dissonance_matrix * amp_products) / 2
            total_dissonance += frame_dissonance

        return total_dissonance / n_frames if n_frames > 0 else 0.0

    def process_file(self, audio_buffer: np.ndarray) -> dict:
        """
        Processes an entire audio file to extract harmonic features.
        """
        print("Processing Harmony Features...")
        
        sensory_dissonance = self.calculate_sensory_dissonance(audio_buffer)

        return {
            "sensory_dissonance": sensory_dissonance,
        }