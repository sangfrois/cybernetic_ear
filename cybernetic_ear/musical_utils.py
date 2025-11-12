"""
Musical utility functions.
"""

def ratio_to_interval(ratio, threshold=0.03):
    """
    Converts a float ratio to its nearest musical interval name.
    """
    # Table of Just Intonation intervals
    intervals = {
        "Unison": 1.0,
        "Minor Second": 16/15,
        "Major Second": 9/8,
        "Minor Third": 6/5,
        "Major Third": 5/4,
        "Perfect Fourth": 4/3,
        "Tritone": 45/32,
        "Perfect Fifth": 3/2,
        "Minor Sixth": 8/5,
        "Major Sixth": 5/3,
        "Minor Seventh": 9/5,
        "Major Seventh": 15/8,
        "Octave": 2.0
    }

    # Normalize the ratio to be within an octave (1.0 to 2.0)
    while ratio >= 2.0:
        ratio /= 2.0
    while ratio < 1.0:
        ratio *= 2.0

    # Find the closest interval
    closest_interval = min(intervals.items(), key=lambda item: abs(item[1] - ratio))

    # Check if the ratio is within the threshold of the closest interval
    if abs(closest_interval[1] - ratio) <= threshold:
        return closest_interval[0]
    else:
        return " dissonant"
