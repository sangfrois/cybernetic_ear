import librosa
import numpy as np
from .base_stream import BaseStream
from ..core.networks import GrooveGRU

import os
import torch

class RhythmStream(BaseStream):
    """
    Analyzes the rhythmic and temporal features of the audio stream.
    """
    def __init__(self, rate=22050, buffer_size_seconds=5):
        self.rate = rate
        self.buffer_size = rate * buffer_size_seconds
        self.audio_buffer = np.array([], dtype=np.float32)
        
        self.beats = []
        self.onsets = []
        self.tempo = 0.0
        self.gru_model = self.load_gru_model()

    def load_gru_model(self):
        model = GrooveGRU()
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'groove_gru.pth'))
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("Loaded pre-trained GrooveGRU model.")
        else:
            print("Warning: Pre-trained GrooveGRU model not found. Using an untrained model.")
        return model

    def process_chunk(self, chunk, feature_bus):
        """
        Processes a chunk to extract rhythmic features and updates the feature bus.
        This stream is stateful and processes audio in larger buffered segments.
        """
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

        if len(self.audio_buffer) >= self.buffer_size:
            buffer = self.audio_buffer
            
            self.tempo, beat_frames = librosa.beat.beat_track(y=buffer, sr=self.rate, units='frames')
            self.beats = librosa.frames_to_time(beat_frames, sr=self.rate)

            onset_frames = librosa.onset.onset_detect(y=buffer, sr=self.rate, units='time')
            self.onsets = onset_frames

            # Calculate beat salience based on onset strength
            onset_env = librosa.onset.onset_strength(y=buffer, sr=self.rate)
            beat_salience = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0
            feature_bus.update_feature("beat_salience", beat_salience)

            duration = len(buffer) / self.rate
            event_density = len(self.onsets) / duration if duration > 0 else 0
            feature_bus.update_feature("event_density", event_density)

            if len(self.beats) > 0 and len(self.onsets) > 0:
                off_beat_count = 0
                beat_tolerance = 0.1
                microtiming_deviations = []
                for onset_time in self.onsets:
                    min_diff_index = np.argmin(np.abs(self.beats - onset_time))
                    min_diff = self.beats[min_diff_index] - onset_time
                    if abs(min_diff) > beat_tolerance:
                        off_beat_count += 1
                    microtiming_deviations.append(min_diff)
                
                total_onsets = len(self.onsets)
                syncopation_index = (off_beat_count / total_onsets) if total_onsets > 0 else 0.0
                feature_bus.update_feature("syncopation_index", syncopation_index)

                # TODO: Implement GRU model for groove pattern analysis
                if microtiming_deviations:
                    # Placeholder for GRU processing
                    groove_pattern = self.gru_model.process(microtiming_deviations)
                    feature_bus.update_feature("groove_pattern", groove_pattern)

            # Slide the buffer window
            self.audio_buffer = self.audio_buffer[len(chunk):]

    def process_file(self, audio_buffer: np.ndarray) -> dict:
        """
        Processes an entire audio file to extract rhythmic features.
        """
        print("Processing Rhythm Features...")

        tempo, beats = librosa.beat.beat_track(y=audio_buffer, sr=self.rate, units='frames')
        beat_times = librosa.frames_to_time(beats, sr=self.rate)
        
        onset_frames = librosa.onset.onset_detect(y=audio_buffer, sr=self.rate, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=self.rate)

        duration = len(audio_buffer) / self.rate
        event_density = len(onset_frames) / duration if duration > 0 else 0

        syncopation_index = 0.0
        if len(beats) > 0 and len(onset_frames) > 0:
            beat_set = set(beats)
            off_beat_count = 0
            for onset_frame in onset_frames:
                min_diff = np.min(np.abs(beats - onset_frame))
                beat_period = np.mean(np.diff(beats)) if len(beats) > 1 else self.rate / (tempo / 60)
                if min_diff > beat_period / 8:
                    off_beat_count += 1
            total_onsets = len(onset_frames)
            syncopation_index = (off_beat_count / total_onsets) if total_onsets > 0 else 0.0

        num_frames = len(librosa.feature.spectral_centroid(y=audio_buffer, sr=self.rate)[0])
        groove_pattern = np.zeros(num_frames)

        return {
            "beat_salience": np.full(num_frames, 1.0 if tempo > 0 else 0.0),
            "event_density": np.full(num_frames, event_density),
            "syncopation_index": np.full(num_frames, syncopation_index),
            "groove_pattern": groove_pattern,
            "beats": beat_times,
            "onsets": onset_times,
            "tempo": tempo
        }

