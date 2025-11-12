import argparse
import json
import numpy as np
import librosa
import sys
import os

# Add the 'cybernetic_ear' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cybernetic_ear'))

from streams.stream_timbre import TimbreStream
from streams.stream_rhythm import RhythmStream
from streams.stream_harmony import HarmonyStream

def analyze_file(file_path):
    """
    Analyzes a given audio file using the offline 'process_file' method
    of the feature streams and saves the result to a structured log file.
    """
    # --- 1. Initialization ---
    print("=============================================")
    print("=      Cybernetic Ear - Offline Analysis    =")
    print("=            (Full Analysis)              =")
    print("=============================================")
    
    SAMPLE_RATE = 22050
    
    # Instantiate streams
    timbre_stream = TimbreStream(rate=SAMPLE_RATE)
    rhythm_stream = RhythmStream(rate=SAMPLE_RATE)
    harmony_stream = HarmonyStream(rate=SAMPLE_RATE)

    # --- 2. Load Audio ---
    print(f"\nLoading audio file: {file_path}")
    try:
        audio_buffer, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        print(f"Audio loaded successfully. Duration: {len(audio_buffer)/SAMPLE_RATE:.2f} seconds.")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # --- 3. Process Audio using process_file ---
    print("\nProcessing audio through feature streams (offline mode)...")
    
    timbre_features = timbre_stream.process_file(audio_buffer)
    rhythm_features = rhythm_stream.process_file(audio_buffer)
    harmony_features = harmony_stream.process_file(audio_buffer)
    
    # --- 4. Merge and Save Log ---
    # Merge all feature dictionaries into one session log
    session_log = {**timbre_features, **rhythm_features, **harmony_features}
    
    log_file_path = 'session_log.json'
    print(f"\nProcessing complete. Saving session log to {log_file_path}...")
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

    with open(log_file_path, 'w') as f:
        json.dump(session_log, f, cls=NumpyEncoder)
        
    print("Session log saved successfully.")
    print("You can now run the plotting script: python analysis/plot_analysis.py session_log.json <your_audio_file>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline analysis on an audio file.")
    parser.add_argument("audio_file", type=str, help="The path to the audio file to analyze.")
    
    args = parser.parse_args()
    
    analyze_file(args.audio_file)

