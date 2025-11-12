import sys
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

from .dashboard.dashboard import run_dashboard, socketio
import threading

import time
# import numpy as np

from .audio_stream import AudioStream
from .features.feature_bus import FeatureBus
from .streams.stream_timbre import TimbreStream
from .streams.stream_rhythm import RhythmStream
from .streams.stream_harmony import HarmonyStream

from .core.paskian_triad_agent import PaskianTriadAgent

import torch

import numpy as np

def to_json_serializable(data):
    """
    Recursively converts a dictionary or list to be JSON serializable.
    """
    if isinstance(data, dict):
        return {k: to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, np.ndarray)):
        return [to_json_serializable(i) for i in data]
    elif isinstance(data, torch.Tensor):
        return to_json_serializable(data.detach().numpy())
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    else:
        return data

def main():
    """
    Main function to run the Cybernetic Ear.
    Initializes and connects all the components, then runs the main loop.
    """
    parser = argparse.ArgumentParser(description="Cybernetic Ear: Real-time audio analysis.")
    parser.add_argument('--disable-biotuner', action='store_true', help="Disable the Biotuner analysis in the harmony stream.")
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    # --- 0. Startup Message ---
    # print("=============================================")
    # print("=         Cybernetic Ear - Real-Time Run      =")
    # print("=============================================")
    # print("This script tests the real-time feature extraction pipeline.")

    # --- Pre-flight check: List audio devices ---
    # AudioStream.list_devices()

    # --- 1. Initialization ---
    # print("\nInitializing the Cybernetic Ear...")
    
    # Configuration
    SAMPLE_RATE = 22050
    CHUNK_SIZE = 2048
    
    # Instantiate major components
    feature_bus = FeatureBus()
    audio_stream = AudioStream(rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
    
    # Instantiate feature processing streams
    timbre_stream = TimbreStream(rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
    rhythm_stream = RhythmStream(rate=SAMPLE_RATE, buffer_size_seconds=5) 
    harmony_stream = HarmonyStream(rate=SAMPLE_RATE, buffer_size_seconds=5, biotuner_enabled=not args.disable_biotuner)

    # Instantiate the Cybernetic Core
    agent = PaskianTriadAgent()

    # --- 2. Connect Components ---
    # print("Connecting components...")
    
    # The feature bus is passed to each callback, so they all update the same object.
    audio_stream.callbacks = [
        lambda chunk: timbre_stream.process_chunk(chunk, feature_bus),
        lambda chunk: rhythm_stream.process_chunk(chunk, feature_bus),
        lambda chunk: harmony_stream.process_chunk(chunk, feature_bus)
    ]

    # --- 3. Start Dashboard ---
    # print("Starting dashboard in a background thread...")
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    # print("Dashboard is running on http://localhost:5001")

    # --- 4. Start Processing ---
    harmony_stream.start(feature_bus)
    audio_stream.start()
    
    # print("\n--- The Cybernetic Ear is now listening ---")
    # print("Press Ctrl+C to stop.")
    
    try:
        # Main loop to periodically update the dashboard
        while True:
            # Update the Cybernetic Core
            state, attention_weights = agent.get_action(feature_bus)
            reward = agent.calculate_reward(state, attention_weights, feature_bus)
            agent.train_step(state, attention_weights, reward)

            # Prepare data for the dashboard
            features = feature_bus.get_all_features()
            dashboard_data = {
                'features': features,
                'agent': {
                    'attention': attention_weights.detach().numpy().flatten().tolist(),
                    'reward': reward.detach().item()
                },
                'biotuner': features.get('biotuner', {})
            }
            
            # Emit data to the dashboard
            socketio.emit('update_data', to_json_serializable(dashboard_data))
            socketio.sleep(1) # Update interval
            
    except KeyboardInterrupt:
        pass
        # print("\nShutdown signal received.")
    except Exception as e:
        pass
        # print(f"An unexpected error occurred: {e}")
    finally:
        # --- 5. Cleanup ---
        # print("Stopping all streams and cleaning up.")
        harmony_stream.stop()
        audio_stream.stop()
        # print("Cybernetic Ear has been shut down.")


if __name__ == '__main__':
    main()

