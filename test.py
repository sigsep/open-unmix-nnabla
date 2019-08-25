import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
import model
import utils
import warnings
import tqdm


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    _, audio = scipy.signal.istft(
        X / (n_fft // 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio