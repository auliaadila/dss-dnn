import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf


class AudioPreprocessor:
    """Preprocessor to cache audio files and avoid repeated soundfile calls."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, audio_root: str) -> Path:
        """Get cache file path for a given audio root directory."""
        # Use absolute path to ensure consistent hashing
        abs_root = os.path.abspath(audio_root)
        cache_name = f"audio_cache_{abs(hash(abs_root))}.pkl"
        return self.cache_dir / cache_name

    def _discover_audio_files(self, root: str) -> List[str]:
        """Discover all audio files in the root directory."""
        wavs = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
        if not wavs:
            raise RuntimeError(f"No audio files found in {root}")
        return sorted(wavs)

    def preprocess_and_cache(
        self, audio_root: str, force_refresh: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess audio files and cache them.

        Args:
            audio_root: Root directory containing audio files
            force_refresh: If True, ignore existing cache and reprocess

        Returns:
            Dictionary mapping file paths to audio arrays
        """
        cache_path = self._get_cache_path(audio_root)

        # Check if cache exists and is valid
        if cache_path.exists() and not force_refresh:
            print(f"Loading cached audio data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Discover audio files
        audio_files = self._discover_audio_files(audio_root)
        print(f"Found {len(audio_files)} audio files in {audio_root}")

        # Load and cache all audio files
        audio_cache = {}
        print("Preprocessing audio files...")

        for i, audio_path in enumerate(audio_files):
            if i % 50 == 0:
                print(f"Processing {i + 1}/{len(audio_files)}: {audio_path}")

            try:
                # Load full audio file
                audio_data, sample_rate = sf.read(audio_path, dtype="float32")

                # Store in cache
                audio_cache[audio_path] = {
                    "data": audio_data,
                    "sample_rate": sample_rate,
                    "frames": len(audio_data),
                }

            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

        # Save cache
        print(f"Saving cache to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(audio_cache, f)

        return audio_cache

    def get_cached_audio(self, audio_root: str) -> Dict[str, np.ndarray]:
        """Get cached audio data for a given root directory."""
        cache_path = self._get_cache_path(audio_root)

        if not cache_path.exists():
            return self.preprocess_and_cache(audio_root)

        with open(cache_path, "rb") as f:
            return pickle.load(f)
