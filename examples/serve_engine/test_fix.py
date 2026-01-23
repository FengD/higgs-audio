#!/usr/bin/env python3
"""
Test script to verify the TTS API fixes.
"""

import os
import sys
import requests
import time
import json

def test_api_fixes():
    """Test the fixes for TTS API audio playback issue."""
    
    # API endpoint
    api_url = "http://localhost:8000"
    
    # Test data - simplified version of the input_multiuser.txt content
    test_text = """[SPEAKER0] 你好，这是一个测试。
[SPEAKER1] 你好，我也很高兴参与测试。"""
    
    # Speaker mappings for the test
    speaker_mappings = [
        {
            "speaker_tag": "SPEAKER0",
            "voice_sample_path": "../voice_prompts/zh_man.wav",
            "voice_text_path": "../voice_prompts/zh_man.txt"
        },
        {
            "speaker_tag": "SPEAKER1", 
            "voice_sample_path": "../voice_prompts/zh_woman.wav",
            "voice_text_path": "../voice_prompts/zh_woman.txt"
        }
    ]
    
    # Test request data
    request_data = {
        "text_content": test_text,
        "speaker_voice_mappings": speaker_mappings,
        "speed": 1.0,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    print("Testing TTS API fixes...")
    
    try:
        # 1. Test API root endpoint
        print("\n1. Testing API root endpoint...")
        response = requests.get(f"{api_url}/api")
        if response.status_code == 200:
            print("✓ API root endpoint works correctly")
        else:
            print(f"✗ API root endpoint failed with status {response.status_code}")
            
        # 2. Test health check endpoint
        print("\n2. Testing health check endpoint...")
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed. Model loaded: {health_data.get('model_loaded', False)}")
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            
        # 3. Test TTS generation endpoint
        print("\n3. Testing TTS generation endpoint...")
        response = requests.post(
            f"{api_url}/tts/multi-speaker",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ TTS generation successful")
            print(f"  Message: {result.get('message')}")
            print(f"  Output file: {result.get('output_file')}")
            print(f"  Speakers detected: {result.get('speakers_detected')}")
            
            # 4. Test audio file serving
            output_file = result.get('output_file', '')
            if output_file:
                print("\n4. Testing audio file serving...")
                # Extract filename from the path
                filename = output_file.split('/')[-1]
                audio_response = requests.get(f"{api_url}/tmp/{filename}")
                
                if audio_response.status_code == 200:
                    print("✓ Audio file serving works correctly")
                    print(f"  Content-Type: {audio_response.headers.get('content-type')}")
                    print(f"  Content-Length: {len(audio_response.content)} bytes")
                else:
                    print(f"✗ Audio file serving failed with status {audio_response.status_code}")
                    
                # Also test the alternative endpoint
                audio_response2 = requests.get(f"{api_url}/audio/{filename}")
                if audio_response2.status_code == 200:
                    print("✓ Alternative audio file serving endpoint works correctly")
                else:
                    print(f"✗ Alternative audio file serving endpoint failed with status {audio_response2.status_code}")
        else:
            print(f"✗ TTS generation failed with status {response.status_code}")
            print(f"  Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to the API server. Please make sure the TTS API service is running.")
        print("  You can start it with: cd examples/serve_engine && python start_tts_api.py")
    except Exception as e:
        print(f"✗ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_fixes()