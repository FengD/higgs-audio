#!/usr/bin/env python3
"""Test script for speed adjustment functionality."""

import torch
import numpy as np
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from text_to_speech import _adjust_speed as tts_adjust_speed
from text_to_speech_multi_speaker import _adjust_speed as multi_adjust_speed


def test_speed_adjustment():
    """Test the speed adjustment functions."""
    # Create a simple test signal - a sine wave
    sampling_rate = 22050
    duration = 2.0  # 2 seconds
    freq = 440.0  # 440 Hz (A4 note)
    
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    test_signal = np.sin(2 * np.pi * freq * t)
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(test_signal.astype(np.float32))
    
    print("Testing speed adjustment functions...")
    print(f"Original audio length: {len(test_signal)} samples")
    
    # Test different speeds
    speeds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    qualities = ['low', 'medium', 'high']
    
    for speed in speeds:
        print(f"\nTesting speed factor: {speed}")
        for quality in qualities:
            try:
                # Test single speaker function
                adjusted = tts_adjust_speed(audio_tensor, sampling_rate, speed, quality)
                print(f"  TTS ({quality}): {len(adjusted)} samples")
                
                # Test multi-speaker function
                adjusted_multi = multi_adjust_speed(audio_tensor, sampling_rate, speed, quality)
                print(f"  Multi ({quality}): {len(adjusted_multi)} samples")
            except Exception as e:
                print(f"  Error with {quality} quality: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_speed_adjustment()