import numpy as np
from biotuner.biotuner_object import compute_biotuner, harmonic_tuning

def biotuner_realtime(
    data,
    sfreq,
    n_peaks=5,
    peaks_function="EMD",
    min_freq=1,
    max_freq=65,
    precision=0.1,
    n_harm_extended=3,
    n_harm_subharm=3,
    delta_lim=250,
):
    """
    This function extracts dominant frequency peaks from an incoming 1D signal and computes several 
    music-theoretical and harmonic metrics from the spectral content using the Biotuner library.
    """
    
    # run biotuner peak extraction
    bt = compute_biotuner(peaks_function=peaks_function, sf=sfreq)
    try:
        bt.peaks_extraction(
            np.array(data),
            graph=False,
            min_freq=min_freq,
            max_freq=max_freq,
            precision=precision,
            nIMFs=5,
            n_peaks=n_peaks,
            smooth_fft=1,
        )
    except (UnboundLocalError, ValueError):
        # Return empty or default values if no peaks are found
        default_metrics = {'subharm_tension': [np.nan], 'harmsim': np.nan, 'tenney': np.nan}
        default_biotuner_data = {'ratios': [], 'interpretation_cues': 'No peaks found.'}
        return [], [], default_metrics, [], [0, 1], [], [], default_biotuner_data


    try:
        # try computing the extended peaks
        bt.peaks_extension(method="harmonic_fit", n_harm=n_harm_extended)
    except TypeError:
        # This can happen if only one peak is detected
        bt.extended_peaks = []
        bt.extended_amps = []


    bt.compute_peaks_metrics(n_harm=n_harm_subharm, delta_lim=delta_lim)
    if hasattr(bt, "all_harmonics"):
        harm_tuning = harmonic_tuning(bt.all_harmonics)
    else:
        harm_tuning = [0, 1]
    peaks = bt.peaks if hasattr(bt, 'peaks') else []
    amps = bt.amps if hasattr(bt, 'amps') else []
    extended_peaks = bt.extended_peaks if hasattr(bt, 'extended_peaks') else []
    extended_amps = bt.extended_amps if hasattr(bt, 'extended_amps') else []
    metrics = bt.peaks_metrics

    if "subharm_tension" not in metrics or not isinstance(metrics["subharm_tension"], (list, np.ndarray)) or len(metrics["subharm_tension"]) == 0 or not isinstance(metrics["subharm_tension"][0], float):
        metrics["subharm_tension"] = [np.nan]
    if "harmsim" in metrics:
        metrics["harmsim"] = metrics["harmsim"] / 100
    else:
        metrics["harmsim"] = np.nan
    # rescale tenney height from between 4 to 9 to between 0 and 1
    if "tenney" in metrics:
        metrics["tenney"] = (metrics["tenney"] - 4) / 5
    else:
        metrics["tenney"] = np.nan

    tuning = bt.peaks_ratios if hasattr(bt, 'peaks_ratios') else []
    return peaks, extended_peaks, metrics, tuning, harm_tuning, amps, extended_amps
