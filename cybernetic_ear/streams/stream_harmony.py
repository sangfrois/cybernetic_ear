import librosa
import numpy as np
from .base_stream import BaseStream
from scipy.spatial.distance import pdist, squareform

from biotuner.biotuner_object import compute_biotuner as Biotuner
import torch
from cybernetic_ear.core.biotuner_object import biotuner_realtime
import threading
import queue

class HarmonyStream(BaseStream):
    """
    Analyzes the harmonic and tensional features of the audio stream in a separate thread
    to avoid blocking the main audio processing pipeline.
    """
    def __init__(self, rate=22050, buffer_size_seconds=4, biotuner_enabled=True):
        self.rate = rate
        self.buffer_size = rate * buffer_size_seconds
        self.audio_buffer = np.array([], dtype=np.float32)
        self.biotuner_enabled = biotuner_enabled

        self.chunk_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._process_loop)

        self.biotuner_update_interval = 10
        self.frame_count = 0
        self.last_biotuner_features = {}

        self.dissonance_update_interval = 5
        self.dissonance_frame_count = 0
        self.last_sensory_dissonance = 0.0

        self.tonal_context_update_interval = 5
        self.tonal_context_frame_count = 0
        self.last_tonal_context = np.zeros(24)

    def start(self, feature_bus):
        self.feature_bus = feature_bus
        self.processing_thread.start()

    def stop(self):
        self.stop_event.set()
        self.processing_thread.join()

    def _process_loop(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.chunk_queue.get(timeout=1)
                self.process_chunk_threaded(chunk)
            except queue.Empty:
                continue

    def process_chunk(self, chunk, feature_bus):
        """
        This method is called from the main audio thread.
        It puts the chunk into a queue for background processing.
        """
        self.chunk_queue.put(chunk)

    def process_chunk_threaded(self, chunk):
        """
        This method runs in the background thread to process audio chunks.
        """
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
        self.frame_count += 1

        if len(self.audio_buffer) >= self.buffer_size:
            buffer = self.audio_buffer

            # --- Sensory Dissonance (Plomp-Levelt) ---
            self.dissonance_frame_count += 1
            if self.dissonance_frame_count >= self.dissonance_update_interval:
                sensory_dissonance = self.calculate_sensory_dissonance(buffer)
                self.last_sensory_dissonance = sensory_dissonance
                self.dissonance_frame_count = 0
            else:
                sensory_dissonance = self.last_sensory_dissonance
            self.feature_bus.update_feature("sensory_dissonance", sensory_dissonance)

            # --- Tonal Context (Krumhansl-Schmuckler) ---
            self.tonal_context_frame_count += 1
            if self.tonal_context_frame_count >= self.tonal_context_update_interval:
                tonal_context = self.calculate_tonal_context(buffer)
                self.last_tonal_context = tonal_context
                self.tonal_context_frame_count = 0
            else:
                tonal_context = self.last_tonal_context
            self.feature_bus.update_feature("tonal_context", tonal_context)

            # --- Advanced Harmonic Analysis (Biotuner) ---
            if self.biotuner_enabled:
                if self.frame_count >= self.biotuner_update_interval:
                    self.run_biotuner_analysis(buffer, self.feature_bus)
                    self.frame_count = 0  # Reset counter
                else:
                    for key, value in self.last_biotuner_features.items():
                        self.feature_bus.update_feature(key, value)

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
                n_peaks=5,
                peaks_function="harmonic_recurrence", # More suitable for audio
                min_freq=100, # Audible range
                max_freq=5000, # Reduced for performance
                precision=0.5,
                n_harm_extended=3,
                n_harm_subharm=3,
                delta_lim=250,
            )
            
            # Update feature bus with all relevant metrics
            self.last_biotuner_features = {
                "harmsim": metrics["harmsim"],
                "tenney_height": metrics["tenney"],
                "subharm_tension": metrics["subharm_tension"][0],
                "consonance": metrics["cons"],
                "peaks_ratios_tuning": tuning,
                "harm_tuning": harm_tuning,
                "peaks": peaks,
                "amps": amps,
                "extended_peaks": extended_peaks,
                "extended_amps": extended_amps,
            }
            for key, value in self.last_biotuner_features.items():
                feature_bus.update_feature(key, value)

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
        try:
            stft = np.abs(librosa.stft(audio_buffer))
            total_dissonance = 0
            n_frames = stft.shape[1]
            frames_with_peaks = 0

            # Sample every 5th frame for performance
            for i in range(0, n_frames, 5):
                frame = stft[:, i]

                # More lenient peak detection
                peaks = librosa.util.peak_pick(frame, pre_max=2, post_max=2, pre_avg=2, post_avg=3, delta=0.01, wait=5)

                if len(peaks) < 2:
                    continue

                frames_with_peaks += 1
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

            result = total_dissonance / frames_with_peaks if frames_with_peaks > 0 else 0.0
            return float(result)
        except Exception as e:
            print(f"Error calculating sensory dissonance: {e}")
            return 0.0

    def process_file(self, audio_buffer: np.ndarray) -> dict:
        """
        Processes an entire audio file to extract harmonic features.
        """
        print("Processing Harmony Features...")
        
        sensory_dissonance = self.calculate_sensory_dissonance(audio_buffer)

        return {
            "sensory_dissonance": sensory_dissonance,
        }