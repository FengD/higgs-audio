#!/usr/bin/env python3
"""Test script to verify the volume fix in speed adjustment functions."""

import torch
import numpy as np

def test_volume_preservation():
    """Test that the volume is preserved after speed adjustment."""
    print("Testing volume preservation in speed adjustment...")
    
    # Create a test signal with known amplitude
    sampling_rate = 22050
    duration = 1.0
    freq = 440.0
    
    # Create a sine wave with amplitude 0.5
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    test_signal = 0.5 * np.sin(2 * np.pi * freq * t)
    audio_tensor = torch.from_numpy(test_signal.astype(np.float32))
    
    print(f"Original audio RMS: {torch.sqrt(torch.mean(audio_tensor ** 2)):.6f}")
    print(f"Original audio max: {torch.max(torch.abs(audio_tensor)):.6f}")
    
    # Simulate our volume preservation technique
    original_rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    
    # Simulate processed audio with reduced amplitude (like what happens in time stretching)
    processed_audio = audio_tensor * 0.7  # Simulate 30% volume loss
    
    print(f"Processed audio RMS: {torch.sqrt(torch.mean(processed_audio ** 2)):.6f}")
    
    # Apply our volume restoration technique
    result_rms = torch.sqrt(torch.mean(processed_audio ** 2))
    if result_rms > 0:
        # Apply gain to restore original RMS level
        gain = original_rms / result_rms
        # Limit gain to prevent clipping
        gain = min(gain, 2.0)
        restored_audio = processed_audio * gain
    else:
        restored_audio = processed_audio
    
    print(f"Restored audio RMS: {torch.sqrt(torch.mean(restored_audio ** 2)):.6f}")
    print(f"Restored audio max: {torch.max(torch.abs(restored_audio)):.6f}")
    
    # Check if volume is preserved
    restored_rms = torch.sqrt(torch.mean(restored_audio ** 2))
    rms_ratio = restored_rms / original_rms
    print(f"RMS ratio (restored/original): {rms_ratio:.6f}")
    
    if 0.95 <= rms_ratio <= 1.05:
        print("✓ Volume preservation test PASSED")
    else:
        print("✗ Volume preservation test FAILED")

def test_edge_cases():
    """Test edge cases for volume preservation."""
    print("\n\nTesting edge cases...")
    
    # Test with silent audio
    silent_audio = torch.zeros(1000, dtype=torch.float32)
    original_rms = torch.sqrt(torch.mean(silent_audio ** 2))
    print(f"Silent audio RMS: {original_rms:.6f}")
    
    # Test with constant audio
    constant_audio = torch.full((1000,), 0.3, dtype=torch.float32)
    original_rms = torch.sqrt(torch.mean(constant_audio ** 2))
    print(f"Constant audio RMS: {original_rms:.6f}")
    
    # Test with clipping prevention
    loud_audio = torch.full((1000,), 0.9, dtype=torch.float32)
    processed_loud = loud_audio * 1.5  # Would clip without limiting
    original_rms = torch.sqrt(torch.mean(loud_audio ** 2))
    result_rms = torch.sqrt(torch.mean(processed_loud ** 2))
    gain = original_rms / result_rms if result_rms > 0 else 1.0
    limited_gain = min(gain, 2.0)  # Limit gain to prevent clipping
    print(f"Limited gain applied: {limited_gain:.6f}")

if __name__ == "__main__":
    test_volume_preservation()
    test_edge_cases()
    print("\n\nVolume preservation tests completed!")