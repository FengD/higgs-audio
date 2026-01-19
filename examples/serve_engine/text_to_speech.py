#!/usr/bin/env python3
"""Text to speech generation with voice cloning and speed control."""

import os
import base64
import torch
import torchaudio
import time
import numpy as np
from loguru import logger
import click
import re
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available, speed adjustment will not work")

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",
        "。": ". ",
        "：": ": ",
        "；": "; ",
        "？": "? ",
        "！": "! ",
        "（": " (",
        "）": ") ",
        "【": " [",
        "】": "] ",
        "《": "",
        "》": "",
        "“": '',
        "”": '',
        "‘": " '",
        "’": "' ",
        "、": ", ",
        "——": ", ",
        "—": ", ",
        "…": "...",
        "·": ". ",
        "「": '',
        "」": ',',
        "『": '',
        "』": ',',
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text


def get_text_input_sample(text_content: str, voice_sample_path: str = None, voice_sample_text: str = None):
    """Create a ChatMLSample from text content with optional voice cloning."""
    # Perform some basic normalization
    text_content = normalize_chinese_punctuation(text_content)
    # Other normalizations (e.g., parentheses and other symbols. Will be improved in the future)
    text_content = text_content.replace("(", " ")
    text_content = text_content.replace(")", " ")
    
    # Handle special tags
    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        text_content = text_content.replace(tag, replacement)
    
    # Clean up extra whitespace
    lines = text_content.split("\n")
    text_content = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    text_content = text_content.strip()

    logger.info(f"Loaded text content with {len(text_content)} characters")
    logger.info(f"Normalized text content: {text_content}")


    messages = []
    
    # If voice cloning is requested, add the reference audio
    if voice_sample_path and voice_sample_text and os.path.exists(voice_sample_path):
        # Try to read the reference text file
        if os.path.exists(voice_sample_text):
            with open(voice_sample_text, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()
        else:
            reference_text = voice_sample_text
            
        reference_audio = encode_base64_content_from_file(voice_sample_path)
        
        messages.extend([
            Message(
                role="user",
                content=reference_text,
            ),
            Message(
                role="assistant",
                content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
            )
        ])
    
    # Add the main text to generate
    messages.append(
        Message(
            role="user",
            content=f"<|generation_instruction_start|>\nGenerate audio for the following text:\n<|generation_instruction_end|>\n\n{text_content}",
        )
    )
    
    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: clear voice\n"
        "<|scene_desc_end|>"
    )
    
    messages.insert(0, Message(
        role="system",
        content=system_prompt,
    ))

    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


@click.command()
@click.option("--text-file", type=click.Path(exists=True), required=True, help="Path to the text file to generate audio from")
@click.option("--output-file", type=click.Path(), default="output_tts.wav", help="Output audio file path")
@click.option("--voice-sample", type=click.Path(), help="Path to voice sample WAV file for cloning")
@click.option("--voice-text", type=str, help="Text corresponding to the voice sample, or path to text file")
@click.option("--speed", type=float, default=1.0, help="Speech speed factor (0.5-2.0)")
@click.option("--temperature", type=float, default=0.7, help="Generation temperature")
@click.option("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
def main(text_file: str, output_file: str, voice_sample: str, voice_text: str, speed: float, temperature: float, max_tokens: int):
    """Generate speech from text with optional voice cloning and speed control."""
    
    # Validate speed parameter
    if speed < 0.5 or speed > 2.0:
        logger.warning("Speed should be between 0.5 and 2.0, setting to default 1.0")
        speed = 1.0
    
    # Read text content from file
    with open(text_file, "r", encoding="utf-8") as f:
        text_content = f.read().strip()
    
    if not text_content:
        logger.error("Text file is empty")
        return
    
    # logger.info(f"Loaded text content with {len(text_content)} characters")

    # logger.info(f"Raw text content: {text_content}")
    # Create input sample
    input_sample = get_text_input_sample(text_content, voice_sample, voice_text)
    
    # Model paths (you may need to adjust these paths)
    MODEL_PATH = "/mnt/fd9ef272-d51b-4896-bfc8-9beaa52ae4a5/dingfeng1/higgs-audio-v2-generation-3B-base/"
    AUDIO_TOKENIZER_PATH = "/mnt/fd9ef272-d51b-4896-bfc8-9beaa52ae4a5/dingfeng1/higgs-audio-v2-tokenizer/"
    
    # Check if model paths exist, if not use HF hub
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model path {MODEL_PATH} does not exist, using HuggingFace hub")
        MODEL_PATH = "boson/Higgs-Audio-v2-3B"
    
    if not os.path.exists(AUDIO_TOKENIZER_PATH):
        logger.warning(f"Audio tokenizer path {AUDIO_TOKENIZER_PATH} does not exist, using HuggingFace hub")
        AUDIO_TOKENIZER_PATH = "boson/Higgs-Audio-v2-tokenizer"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )

    logger.info("Starting generation...")
    start_time = time.time()
    
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=input_sample,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Generation time: {elapsed_time:.2f} seconds")

    if output.audio is not None:
        # Apply speed adjustment if needed
        if speed != 1.0 and LIBROSA_AVAILABLE:
            logger.info(f"Adjusting audio speed to {speed}x")
            # Change audio speed using librosa
            audio_data = output.audio
            original_sr = output.sampling_rate
            
            # Apply time stretching
            adjusted_audio = librosa.effects.time_stretch(audio_data.astype(np.float32), rate=speed)
            
            # Save the adjusted audio
            torchaudio.save(output_file, torch.from_numpy(adjusted_audio)[None, :], original_sr)
        elif speed != 1.0 and not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, saving audio at normal speed")
            torchaudio.save(output_file, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        else:
            # Save audio at normal speed
            torchaudio.save(output_file, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        
        logger.info(f"Generated text:\n{output.generated_text}")
        logger.info(f"Saved audio to {output_file}")
    else:
        logger.error("Failed to generate audio")


if __name__ == "__main__":
    main()