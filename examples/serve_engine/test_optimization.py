#!/usr/bin/env python3
"""Test script to verify the optimization of speed adjustment and audio concatenation."""

import torch
import numpy as np
import torchaudio
from text_to_speech import _adjust_speed as tts_adjust_speed, _concat_audio_segments as tts_concat
from text_to_speech_multi_speaker import _adjust_speed as multi_adjust_speed, _concat_audio_segments as multi_concat

def create_test_signal(sampling_rate=22050, duration=1.0, freq=440.0):
    """Create a simple test signal - a sine wave."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    test_signal = np.sin(2 * np.pi * freq * t)
    return torch.from_numpy(test_signal.astype(np.float32))

def test_speed_adjustment():
    """Test the optimized speed adjustment functions."""
    print("Testing speed adjustment optimization...")
    
    # Create test signal
    audio_tensor = create_test_signal()
    sampling_rate = 22050
    
    print(f"Original audio length: {len(audio_tensor)} samples")
    
    # Test different speeds
    speeds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    qualities = ['low', 'medium', 'high']
    
    for speed in speeds:
        print(f"\nTesting speed factor: {speed}")
        for quality in qualities:
            try:
                # Test single speaker function
                adjusted_tts = tts_adjust_speed(audio_tensor, sampling_rate, speed, quality)
                print(f"  TTS ({quality}): {len(adjusted_tts)} samples")
                
                # Test multi-speaker function
                adjusted_multi = multi_adjust_speed(audio_tensor, sampling_rate, speed, quality)
                print(f"  Multi ({quality}): {len(adjusted_multi)} samples")
                
                # Verify they produce similar results
                diff = torch.abs(adjusted_tts - adjusted_multi).mean()
                print(f"  Difference: {diff:.6f}")
            except Exception as e:
                print(f"  Error with {quality} quality: {e}")

def test_audio_concatenation():
    """Test the optimized audio concatenation functions."""
    print("\n\nTesting audio concatenation optimization...")
    
    # Create test signals
    sampling_rate = 22050
    signal1 = create_test_signal(sampling_rate, 0.5, 440.0)  # 0.5s at 440Hz
    signal2 = create_test_signal(sampling_rate, 0.5, 880.0)  # 0.5s at 880Hz
    signal3 = create_test_signal(sampling_rate, 0.5, 1320.0) # 0.5s at 1320Hz
    segments = [signal1, signal2, signal3]
    
    print(f"Segment lengths: {[len(s) for s in segments]}")
    
    # Test with different silence values
    silence_values = [0, 10, 50]
    
    for silence_ms in silence_values:
        print(f"\nTesting with silence_ms={silence_ms}")
        
        # Test single speaker function
        concat_tts = tts_concat(segments, sampling_rate, silence_ms)
        print(f"  TTS concatenated length: {len(concat_tts)} samples")
        
        # Test multi-speaker function
        concat_multi = multi_concat(segments, sampling_rate, silence_ms)
        print(f"  Multi concatenated length: {len(concat_multi)} samples")
        
        # Verify they produce similar results
        if len(concat_tts) == len(concat_multi):
            diff = torch.abs(concat_tts - concat_multi).mean()
            print(f"  Difference: {diff:.6f}")
        else:
            print(f"  Length mismatch: {len(concat_tts)} vs {len(concat_multi)}")

def test_cross_fade_effect():
    """Test the cross-fade effect in audio concatenation."""
    print("\n\nTesting cross-fade effect...")
    
    # Create two signals that should show clear cross-fade
    sampling_rate = 22050
    # Signal 1: 1s of constant value 1.0
    signal1 = torch.ones(sampling_rate, dtype=torch.float32)
    # Signal 2: 1s of constant value -1.0
    signal2 = -torch.ones(sampling_rate, dtype=torch.float32)
    segments = [signal1, signal2]
    
    # Test with no silence (direct cross-fade)
    result = tts_concat(segments, sampling_rate, silence_ms=0)
    print(f"Cross-fade result length: {len(result)} samples")
    
    # The middle portion should have values transitioning from 1 to -1
    middle_start = len(signal1) - sampling_rate // 40  # 25ms before end
    middle_end = len(signal1) + sampling_rate // 40   # 25ms after start
    middle_portion = result[middle_start:middle_end]
    print(f"Middle transition values: {middle_portion[:10].tolist()}...")

if __name__ == "__main__":
    test_speed_adjustment()
    test_audio_concatenation()
    test_cross_fade_effect()
    print("\n\nAll tests completed!")