# The Cybernetic Ear

The Cybernetic Ear is a real-time audio analysis and creative feedback system inspired by the principles of cybernetics and Paskian Conversation Theory. It analyzes a live audio stream through three parallel lenses—Timbre, Rhythm, and Harmony—and uses a sophisticated AI agent to model a "conversation" with the music.

The project's core is the `PaskianTriadAgent`, an AI that learns to direct its "attention" based on novelty, tension, and stasis in the incoming audio, aiming to create a dynamic and engaging interaction loop.

## Core Concepts

- **Three Streams:** The system processes audio through three distinct feature streams:
    1.  **Timbre:** Analyzes spectral features like centroid, flatness, and MFCCs using a 1D CNN.
    2.  **Rhythm:** Tracks beats, syncopation, and micro-timing deviations ("groove") using a GRU.
    3.  **Harmony:** Models psychoacoustic principles like sensory dissonance and tonal context, and uses the Biotuner library for deep harmonic analysis.
- **Paskian Triad AI:** The agent's learning architecture is based on:
    1.  **Conversation (RL Agent):** A core reinforcement learning agent that seeks to maximize a reward signal based on conversational fluidity.
    2.  **Boredom (Continual Learning):** Uses Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting and model habituation.
    3.  **Interest (Meta-Learning):** A Fast/Slow weight architecture allows the agent to adapt to new patterns while maintaining a stable "personality."

## Setup

This project is managed using a Conda environment.

1.  **Activate the Conda Environment:**
    ```bash
    conda activate music
    ```

2.  **Install Dependencies:**
    The required Python packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Cybernetic Ear, execute the `main.py` module from the project's root directory. The application will start listening to the default audio input device and launch a web-based dashboard to visualize the analysis in real-time.

```bash
python -m cybernetic_ear.main
```

The dashboard will be available at `http://localhost:5001`.

### Options

-   **Disable Harmony Stream:** The harmony analysis, which includes the Biotuner, can be computationally intensive. You can disable it by using the `--disable-harmony` flag:
    ```bash
    python -m cybernetic_ear.main --disable-harmony
    ```
