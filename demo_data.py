"""Explicitly synthetic rows for demonstrating the application without a dataset."""

import numpy as np
import pandas as pd


def make_synthetic_demo_dataset() -> pd.DataFrame:
    """Create deterministic toy data; it is not representative network traffic."""
    random = np.random.default_rng(42)
    normal_count = 120
    suspicious_count = 120
    labels = np.array(["BENIGN"] * normal_count + ["Attack"] * suspicious_count)
    normal_duration = random.integers(5, 250, normal_count)
    suspicious_duration = random.integers(120, 2000, suspicious_count)
    normal_packets = random.integers(1, 20, normal_count)
    suspicious_packets = random.integers(15, 200, suspicious_count)
    frame = pd.DataFrame(
        {
            "Flow Duration": np.concatenate([normal_duration, suspicious_duration]),
            "Packet Count": np.concatenate([normal_packets, suspicious_packets]),
            "Byte Count": np.concatenate(
                [
                    normal_packets * random.integers(40, 500, normal_count),
                    suspicious_packets * random.integers(100, 1400, suspicious_count),
                ]
            ),
            "Protocol": np.concatenate(
                [
                    random.choice(["TCP", "UDP", "ICMP"], normal_count),
                    random.choice(["TCP", "UDP", "ICMP"], suspicious_count),
                ]
            ),
            "label": labels,
        }
    )
    return frame.sample(frac=1, random_state=42).reset_index(drop=True)
