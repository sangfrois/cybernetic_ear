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
    print("Starting dashboard in a background thread...")
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    print("Dashboard is running on http://localhost:5001")

    # --- 4. Start Processing ---
    harmony_stream.start(feature_bus)
    audio_stream.start()

    print("\n--- The Cybernetic Ear is now listening ---")
    print("Press Ctrl+C to stop.")

    # Give dashboard time to start
    time.sleep(2)
    
    try:
        # Main loop to periodically update the dashboard
        update_count = 0
        while True:
            try:
                # Update the Cybernetic Core
                state, attention_weights = agent.get_action(feature_bus)
                reward = agent.calculate_reward(state, attention_weights, feature_bus)
                agent.train_step(state, attention_weights, reward)

                # Prepare data for the dashboard
                features = feature_bus.get_all_features()
                agent_state = agent.get_agent_state(feature_bus)

                dashboard_data = {
                    'features': features,
                    'agent': {
                        'attention': attention_weights.detach().numpy().flatten().tolist(),
                        'fast_attention': agent_state['fast_attention'],
                        'slow_attention': agent_state['slow_attention'],
                        'reward': reward.detach().item(),
                        'reward_components': agent_state['reward_components'],
                        'stasis_counter': agent_state['stasis_counter'],
                        'plasticity': agent_state['plasticity'],
                        'consolidation_count': agent_state['consolidation_count'],
                        'ewc_active': agent_state['ewc_active']
                    },
                    'biotuner': features.get('biotuner', {})
                }

                # Emit data to the dashboard
                socketio.emit('update_data', to_json_serializable(dashboard_data))

                # Debug output every 10 updates
                update_count += 1
                if update_count % 5 == 0:  # More frequent updates to see behavior
                    print(f"\n{'='*70}")
                    print(f"Update #{update_count}")
                    print(f"  AGENT STATE:")
                    print(f"    Reward: {reward.item():.3f} (N={agent_state['reward_components']['r_novelty']:.2f}, T={agent_state['reward_components']['r_tension']:.2f}, S={agent_state['reward_components']['r_stasis']:.2f})")
                    print(f"    Stasis: {agent_state['stasis_counter']}/30 (action_diff={agent_state.get('action_diff', 0):.4f}, state_diff={agent_state.get('state_diff', 0):.4f})")
                    print(f"    Plasticity: {agent_state['plasticity']:.4f}")
                    attn = attention_weights.detach().numpy().flatten()
                    fast_attn = agent_state['fast_attention']
                    slow_attn = agent_state['slow_attention']
                    print(f"    Combined Attention: T={attn[0]:.2f} R={attn[1]:.2f} H={attn[2]:.2f}")
                    print(f"    Fast:  T={fast_attn[0]:.2f} R={fast_attn[1]:.2f} H={fast_attn[2]:.2f}")
                    print(f"    Slow:  T={slow_attn[0]:.2f} R={slow_attn[1]:.2f} H={slow_attn[2]:.2f}")
                    print(f"  FEATURES (Energy Check):")
                    flux = features.get('spectral_flux', 0)
                    beat = features.get('beat_salience', 0)
                    density = features.get('event_density', 0)
                    total_energy = abs(flux) + abs(beat) + abs(density)
                    print(f"    Total Energy: {total_energy:.3f} (flux={flux:.3f}, beat={beat:.3f}, density={density:.2f})")
                    print(f"    Harmony: tension={features.get('subharm_tension', 0):.3f} (Δ={features.get('tension_delta', 0):+.3f})")
                    print(f"{'='*70}")

                socketio.sleep(1) # Update interval

            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                socketio.sleep(1)
            
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

