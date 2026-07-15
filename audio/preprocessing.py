import librosa
import numpy as np
from config import SAMPLE_RATE, N_MFCC
from utils.logger import setup_logger

logger = setup_logger("AudioPreprocessing")


def load_audio(uploaded_file):
    try:
        logger.info("Loading audio file")
        y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        return y, sr
    except Exception as e:
        logger.exception("Audio loading failed")
        raise RuntimeError(f"Invalid or corrupted audio file: {e}") from e


def extract_mfcc(y, sr):
    try:
        logger.info("Extracting MFCC features")

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )

        logger.info(f"MFCC shape: {mfcc.shape}")

        return mfcc

    except Exception as e:
        logger.exception("MFCC extraction failed")
        raise RuntimeError(f"Feature extraction failed: {e}") from e