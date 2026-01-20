#!/usr/bin/env python3
"""Multi-speaker text to speech generation with voice cloning and speed control."""

import os
import base64
import torch
import torchaudio
import time
import numpy as np
from loguru import logger
import click
import re
from typing import List, Optional, Tuple, Dict, Any
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


def _clean_and_normalize_text(text_content: str) -> str:
    """Normalize punctuation/tags and collapse whitespace while keeping newlines meaningful."""
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

    # Clean up extra whitespace per-line
    lines = text_content.split("\n")
    text_content = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    return text_content.strip()


def _split_long_text(
    text: str,
    max_chars: int,
    min_chars: int = 1,
) -> List[str]:
    """
    Split long text into chunks by sentence boundaries, trying to keep each chunk <= max_chars.
    Falls back to comma/whitespace splitting and finally hard-splitting for very long sentences.
    """
    if max_chars <= 0:
        return [text]
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Sentence boundaries: handle both Chinese and English punctuation, plus newlines.
    end_punct = set(list("。！？!?."))  # strong boundaries
    mid_punct = set(list("，,;；:："))  # weaker boundaries (fallback)

    def flush_chunk(chunks: List[str], buf: str):
        buf = buf.strip()
        if not buf:
            return
        if len(buf) < min_chars and chunks:
            chunks[-1] = (chunks[-1].rstrip() + " " + buf).strip()
        else:
            chunks.append(buf)

    # First pass: split into "sentences"
    sentences: List[str] = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in end_punct or ch == "\n":
            s = "".join(cur).strip()
            if s:
                sentences.append(s)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        sentences.append(tail)

    # Second pass: pack sentences into chunks <= max_chars
    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        sent = " ".join(sent.split())
        if not sent:
            continue

        if len(sent) > max_chars:
            # Flush current buffer before dealing with oversized sentence.
            flush_chunk(chunks, buf)
            buf = ""

            # Try splitting by weaker punctuation.
            parts: List[str] = []
            tmp = []
            for ch in sent:
                tmp.append(ch)
                if ch in mid_punct:
                    p = "".join(tmp).strip()
                    if p:
                        parts.append(p)
                    tmp = []
            rest = "".join(tmp).strip()
            if rest:
                parts.append(rest)

            # Pack parts; if still too large, hard-split.
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(p) <= max_chars:
                    flush_chunk(chunks, p)
                else:
                    for i in range(0, len(p), max_chars):
                        flush_chunk(chunks, p[i : i + max_chars])
            continue

        candidate = (buf.rstrip() + " " + sent).strip() if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush_chunk(chunks, buf)
            buf = sent
    flush_chunk(chunks, buf)

    # Final cleanup: drop empties
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def _split_long_text_for_speaker(
    text: str,
    max_chars: int,
    min_chars: int = 1,
) -> List[str]:
    """
    Split long text for a speaker into chunks by sentence boundaries, trying to keep each chunk <= max_chars.
    Falls back to comma/whitespace splitting and finally hard-splitting for very long sentences.
    This is a copy of _split_long_text adapted for speaker-specific usage.
    """
    if max_chars <= 0:
        return [text]
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Sentence boundaries: handle both Chinese and English punctuation, plus newlines.
    end_punct = set(list("。！？!?."))  # strong boundaries
    mid_punct = set(list("，,;；:："))  # weaker boundaries (fallback)

    def flush_chunk(chunks: List[str], buf: str):
        buf = buf.strip()
        if not buf:
            return
        if len(buf) < min_chars and chunks:
            chunks[-1] = (chunks[-1].rstrip() + " " + buf).strip()
        else:
            chunks.append(buf)

    # First pass: split into "sentences"
    sentences: List[str] = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in end_punct or ch == "\n":
            s = "".join(cur).strip()
            if s:
                sentences.append(s)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        sentences.append(tail)

    # Second pass: pack sentences into chunks <= max_chars
    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        sent = " ".join(sent.split())
        if not sent:
            continue

        if len(sent) > max_chars:
            # Flush current buffer before dealing with oversized sentence.
            flush_chunk(chunks, buf)
            buf = ""

            # Try splitting by weaker punctuation.
            parts: List[str] = []
            tmp = []
            for ch in sent:
                tmp.append(ch)
                if ch in mid_punct:
                    p = "".join(tmp).strip()
                    if p:
                        parts.append(p)
                    tmp = []
            rest = "".join(tmp).strip()
            if rest:
                parts.append(rest)

            # Pack parts; if still too large, hard-split.
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(p) <= max_chars:
                    flush_chunk(chunks, p)
                else:
                    for i in range(0, len(p), max_chars):
                        flush_chunk(chunks, p[i : i + max_chars])
            continue

        candidate = (buf.rstrip() + " " + sent).strip() if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush_chunk(chunks, buf)
            buf = sent
    flush_chunk(chunks, buf)

    # Final cleanup: drop empties
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def _concat_audio_segments(
    segments: List[torch.Tensor],
    sampling_rate: int,
    silence_ms: int = 0,
) -> torch.Tensor:
    """
    Concatenate 1D audio tensors (float) into one 1D tensor.
    - Else inserts silence between segments.
    """
    if not segments:
        return torch.zeros(0, dtype=torch.float32)

    segs = [s.detach().flatten().to(dtype=torch.float32).cpu() for s in segments]

    # silence-based concatenation
    gap = int(sampling_rate * max(silence_ms, 0) / 1000)
    silence = torch.zeros(gap, dtype=torch.float32) if gap > 0 else None
    out = segs[0]
    for nxt in segs[1:]:
        if silence is not None:
            out = torch.cat([out, silence, nxt], dim=0)
        else:
            out = torch.cat([out, nxt], dim=0)
    return out


def _adjust_speed(audio: torch.Tensor, sampling_rate: int, speed: float) -> torch.Tensor:
    """Time-stretch to adjust speed while keeping pitch (best-effort)."""
    if speed == 1.0:
        return audio
    if speed <= 0:
        raise ValueError("speed must be > 0")

    audio = audio.detach().flatten().to(dtype=torch.float32).cpu()

    if LIBROSA_AVAILABLE:
        audio_np = audio.numpy().astype(np.float32, copy=False)
        adjusted = librosa.effects.time_stretch(audio_np, rate=float(speed))
        return torch.from_numpy(adjusted.astype(np.float32, copy=False))

    # Fallback: try sox tempo via torchaudio (may not be available depending on build).
    try:
        wav = audio[None, :]
        effects = [["tempo", str(speed)]]
        out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sampling_rate, effects)
        return out[0].to(dtype=torch.float32).cpu()
    except Exception as e:
        logger.warning(f"Speed adjustment requested but not available (install librosa). Error: {e}")
        return audio


def parse_speaker_text(text_content: str) -> List[Tuple[str, str]]:
    """
    Parse text content with speaker tags and return a list of (speaker, text) tuples.
    
    Args:
        text_content: Text with [SPEAKER*] tags
        
    Returns:
        List of (speaker_tag, text) tuples
    """
    lines = text_content.split('\n')
    speaker_texts = []
    current_speaker = None
    current_text = []
    
    speaker_pattern = re.compile(r'\[(SPEAKER\d+)\]')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        speaker_match = speaker_pattern.match(line)
        if speaker_match:
            # Save previous speaker's text if exists
            if current_speaker and current_text:
                speaker_texts.append((current_speaker, '\n'.join(current_text).strip()))
                current_text = []
            
            # Set new speaker
            current_speaker = speaker_match.group(1)
            # Get the text after the speaker tag
            text_after_tag = line[speaker_match.end():].strip()
            if text_after_tag:
                current_text.append(text_after_tag)
        else:
            # Continue with current speaker
            if current_speaker:
                current_text.append(line)
            else:
                # No speaker tag found, default to SPEAKER0
                if not speaker_texts or speaker_texts[-1][0] != 'SPEAKER0':
                    current_speaker = 'SPEAKER0'
                current_text.append(line)
    
    # Don't forget the last speaker's text
    if current_speaker and current_text:
        speaker_texts.append((current_speaker, '\n'.join(current_text).strip()))
    
    return speaker_texts


def get_speaker_voices(speaker_voice_mapping: Dict[str, Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
    """
    Get speaker to voice mapping.
    
    Args:
        speaker_voice_mapping: Dictionary mapping speaker tags to (voice_sample_path, voice_text) tuples
        
    Returns:
        Dictionary mapping speaker tags to (voice_sample_path, voice_text) tuples
    """
    return speaker_voice_mapping


def get_text_input_sample_multi_speaker(
    speaker_texts: Tuple[str, str],
    speaker_voice_mapping: Dict[str, Tuple[str, str]],
    *,
    log_text: bool = True,
):
    """Create a ChatMLSample from multi-speaker text content with voice cloning."""
    
    speaker, text = speaker_texts

    if log_text:
        logger.info(f"Processing {speaker} speaker segments")
        logger.info(f"{speaker}: {len(text)} characters")
    
    messages = []
    
    # Add reference audios for each speaker
    # for speaker_tag, (voice_sample_path, voice_text_path) in speaker_voice_mapping.items():
    #     logger.info(f"{speaker_tag}")
    
    # Check if speaker exists in speaker_voice_mapping
    if speaker not in speaker_voice_mapping:
        logger.warning(f"Speaker '{speaker}' not found in speaker_voice_mapping. Skipping voice cloning.")
        voice_sample_path, voice_text_path = None, None
    else:
        voice_sample_path, voice_text_path = speaker_voice_mapping[speaker]
        
    if voice_sample_path and voice_text_path and os.path.exists(voice_sample_path):
        # Try to read the reference text file
        if os.path.exists(voice_text_path):
            with open(voice_text_path, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()
        else:
            reference_text = voice_text_path
            
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
    
    # Add the main text segments
    text_content = _clean_and_normalize_text(text)
    messages.append(
        Message(
            role="user",
            content=f"<|generation_instruction_start|>\nGenerate audio for the following text:\n<|generation_instruction_end|>\n\n{text_content}",
        )
    )
    
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


@click.command()
@click.option("--text-file", type=click.Path(exists=True), required=True, help="Path to the text file to generate audio from")
@click.option("--output-file", type=click.Path(), default="output_tts_multi.wav", help="Output audio file path")
@click.option("--speaker-voices", type=str, multiple=True, help="Speaker to voice mapping in format SPEAKER0:path/to/voice.wav,path/to/voice.txt")
@click.option("--speed", type=float, default=1.0, help="Speech speed factor (0.5-2.0)")
@click.option("--temperature", type=float, default=0.7, help="Generation temperature")
@click.option("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
@click.option("--chunk-max-chars", type=int, default=200, show_default=True, help="Max chars per chunk for long text; auto-split when exceeded")
@click.option("--chunk-min-chars", type=int, default=20, show_default=True, help="Min chars per chunk; very short tails will be merged")
@click.option("--silence-ms", type=int, default=20, show_default=True, help="Silence (ms) inserted between chunks when concatenating")
@click.option("--save-chunks-dir", type=click.Path(), default=None, help="If set, save per-chunk wav files into this directory for debugging")
def main(
    text_file: str,
    output_file: str,
    speaker_voices: List[str],
    speed: float,
    temperature: float,
    max_tokens: int,
    chunk_max_chars: int,
    chunk_min_chars: int,
    silence_ms: int,
    save_chunks_dir: Optional[str],
):
    """Generate speech from multi-speaker text with voice cloning and speed control."""
    
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

    # Parse speaker to voice mapping
    speaker_voice_mapping = {}
    if speaker_voices:
        for mapping in speaker_voices:
            if ':' in mapping:
                speaker_tag, voice_info = mapping.split(':', 1)
                if ',' in voice_info:
                    voice_sample, voice_text = voice_info.rsplit(',', 1)
                    speaker_voice_mapping[speaker_tag] = (voice_sample.strip(), voice_text.strip())
                else:
                    logger.warning(f"Invalid voice mapping format for {speaker_tag}. Expected 'speaker:path/to/voice.wav,path/to/voice.txt'")
            else:
                logger.warning(f"Invalid speaker mapping format: {mapping}")
    else:
        # Default mappings if none provided
        speaker_voice_mapping = {
            "SPEAKER0": ("../voice_prompts/belinda.wav", "../voice_prompts/belinda.txt"),
            "SPEAKER1": ("../voice_prompts/chadwick.wav", "../voice_prompts/chadwick.txt"),
        }
        logger.info("Using default speaker voices: SPEAKER0=belinda, SPEAKER1=chadwick")
    
    # Parse speaker texts
    speaker_texts = parse_speaker_text(text_content)
    if not speaker_texts:
        logger.error("No valid speaker text found")
        return
    
    logger.info(f"Parsed {len(speaker_texts)} speaker segments")
    for speaker, text in speaker_texts:
        logger.info(f"{speaker}: {len(text)} characters")
    
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

    if save_chunks_dir:
        os.makedirs(save_chunks_dir, exist_ok=True)

    logger.info("Starting generation...")
    
    # For multi-speaker, we generate each speaker separately and concatenate the results
    all_audio_segments: List[torch.Tensor] = []
    sampling_rate: Optional[int] = None
    total_generated_text: List[str] = []
    
    total_start_time = time.time()
    
    # Process each speaker's text separately
    for speaker_idx, (speaker_tag, speaker_text) in enumerate(speaker_texts):
        logger.info(f"Generating audio for {speaker_tag} (segment {speaker_idx+1}/{len(speaker_texts)})")
        logger.info(f"Text length: {len(speaker_text)} characters")
        logger.info(f"{speaker_text}")
        # Split long text for this speaker
        text_chunks = _split_long_text_for_speaker(speaker_text, max_chars=chunk_max_chars, min_chars=chunk_min_chars)
        if not text_chunks:
            logger.warning(f"No valid text chunks for {speaker_tag}")
            continue
            
        if len(text_chunks) == 1:
            logger.info(f"Text for {speaker_tag} fits in a single chunk")
        else:
            logger.info(f"Splitting text for {speaker_tag} into {len(text_chunks)} chunks")
        
        # Generate audio for each chunk
        speaker_audio_segments: List[torch.Tensor] = []
        
        for chunk_idx, chunk in enumerate(text_chunks):
            logger.info(f"Generating chunk {chunk_idx+1}/{len(text_chunks)} for {speaker_tag}")
            
            # Create input sample for this chunk with the specific speaker's voice
            chunk_speaker_texts = (speaker_tag, chunk)
            input_sample = get_text_input_sample_multi_speaker(
                chunk_speaker_texts,
                speaker_voice_mapping,
            )

            start_time = time.time()
            output: HiggsAudioResponse = serve_engine.generate(
                chat_ml_sample=input_sample,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=30,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Chunk {chunk_idx+1} generation time: {elapsed_time:.2f} seconds")

            if output.audio is None:
                logger.error(f"Failed to generate audio for chunk {chunk_idx+1} of {speaker_tag}")
                continue

            # Convert to torch tensor
            out_sr = int(output.sampling_rate)
            if sampling_rate is None:
                sampling_rate = out_sr
                
            audio_tensor = torch.from_numpy(output.audio.astype(np.float32, copy=False))
            
            # Resample if needed
            if out_sr != sampling_rate:
                logger.warning(
                    f"Sampling rate changed from {sampling_rate} to {out_sr} for {speaker_tag} chunk {chunk_idx+1}; resampling to {sampling_rate}."
                )
                wav = audio_tensor[None, :]
                wav = torchaudio.functional.resample(wav, out_sr, sampling_rate)
                audio_tensor = wav[0]
                
            speaker_audio_segments.append(audio_tensor)
            total_generated_text.append(output.generated_text or "")
            
            # Save chunk if requested
            if save_chunks_dir:
                chunk_path = os.path.join(save_chunks_dir, f"{speaker_tag}_chunk_{chunk_idx+1:03d}.wav")
                torchaudio.save(chunk_path, audio_tensor[None, :].cpu(), sampling_rate)
        
        # Concatenate all chunks for this speaker
        if speaker_audio_segments:
            speaker_audio = _concat_audio_segments(speaker_audio_segments, sampling_rate, silence_ms=0)
            all_audio_segments.append(speaker_audio)
    
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"Total generation time: {total_elapsed_time:.2f} seconds")
    
    # Concatenate all speakers with silence between them
    if not all_audio_segments:
        logger.error("No audio segments generated")
        return
        
    assert sampling_rate is not None
    final_audio = _concat_audio_segments(all_audio_segments, sampling_rate, silence_ms=silence_ms)

    # Apply speed adjustment if needed
    if speed != 1.0:
        logger.info(f"Adjusting final audio speed to {speed}x")
        final_audio = _adjust_speed(final_audio, sampling_rate, speed)

    torchaudio.save(output_file, final_audio[None, :], sampling_rate)
    logger.info(f"Generated text:\n{''.join(total_generated_text)}")
    logger.info(f"Saved audio to {output_file}")


if __name__ == "__main__":
    main()