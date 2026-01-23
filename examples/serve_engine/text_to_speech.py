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
from typing import List, Optional, Tuple
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available, using fallback speed adjustment")

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64



def normalize_chinese_punctuation(text):
    # Collapse ellipsis variants without introducing new punctuation
    text = re.sub(r'(\.\s*){3,}', '.', text)
    text = re.sub(r'(。\s*){3,}', '。', text)
    text = text.replace("…", "。")

    # Basic punctuation normalization
    punct_map = {
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
        "“": "",
        "”": "",
        "‘": " '",
        "’": "' ",
        "、": ", ",
        "——": ", ",
        "—": ", ",
        "·": ". ",
    }

    for src, dst in punct_map.items():
        text = text.replace(src, dst)

    # Ensure at most one trailing punctuation mark
    text = re.sub(r'([.:;]){2,}$', r'\1', text)

    return text.strip()




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
    
    # For better transition, we'll apply cross-fading when concatenating segments
    out = segs[0]
    fade_samples = min(int(sampling_rate * 0.01), len(out) // 4)  # 10ms fade or quarter of segment
    
    for nxt in segs[1:]:
        if gap > 0:
            # Insert silence and apply cross-fade
            silence = torch.zeros(gap, dtype=torch.float32)
            
            # Apply cross-fade between end of out and beginning of nxt if they're long enough
            if len(out) > fade_samples and len(nxt) > fade_samples:
                # Create fade-out for end of 'out'
                fade_out = torch.linspace(1.0, 0.0, fade_samples)
                out_overlap = out[-fade_samples:] * fade_out
                
                # Create fade-in for beginning of 'nxt'
                fade_in = torch.linspace(0.0, 1.0, fade_samples)
                nxt_overlap = nxt[:fade_samples] * fade_in
                
                # Combine with cross-fade
                cross_faded = out_overlap + nxt_overlap
                
                # Reconstruct with cross-fade
                out = torch.cat([out[:-fade_samples], cross_faded, nxt[fade_samples:]], dim=0)
            else:
                # Simple concatenation with silence
                out = torch.cat([out, silence, nxt], dim=0)
        else:
            # Direct concatenation with cross-fade if possible
            if len(out) > fade_samples and len(nxt) > fade_samples:
                # Create fade-out for end of 'out'
                fade_out = torch.linspace(1.0, 0.0, fade_samples)
                out_overlap = out[-fade_samples:] * fade_out
                
                # Create fade-in for beginning of 'nxt'
                fade_in = torch.linspace(0.0, 1.0, fade_samples)
                nxt_overlap = nxt[:fade_samples] * fade_in
                
                # Combine with cross-fade
                cross_faded = out_overlap + nxt_overlap
                
                # Reconstruct with cross-fade
                out = torch.cat([out[:-fade_samples], cross_faded, nxt[fade_samples:]], dim=0)
            else:
                # Simple concatenation
                out = torch.cat([out, nxt], dim=0)
                
    return out


def _adjust_speed(audio: torch.Tensor, sampling_rate: int, speed: float, quality: str = 'high') -> torch.Tensor:
    """Time-stretch to adjust speed while keeping pitch (best-effort)."""
    if speed == 1.0:
        return audio
    if speed <= 0:
        raise ValueError("speed must be > 0")

    # Store original amplitude statistics for later restoration
    original_rms = torch.sqrt(torch.mean(audio ** 2))
    
    audio = audio.detach().flatten().to(dtype=torch.float32).cpu()

    # Pre-processing to reduce artifacts
    # Apply a gentle pre-emphasis filter to reduce distortion
    if quality == 'high':
        # For high quality, apply pre-processing to reduce artifacts
        audio_np = audio.numpy()
        # Apply a simple pre-emphasis filter to reduce distortion
        pre_emphasis = 0.97
        emphasized = np.concatenate([audio_np[:1], audio_np[1:] - pre_emphasis * audio_np[:-1]])
        audio = torch.from_numpy(emphasized.astype(np.float32))

    if LIBROSA_AVAILABLE:
        # Use different algorithms based on quality setting
        audio_np = audio.numpy().astype(np.float32, copy=False)
        
        if quality == 'high':
            # Use high quality WSOLA with additional parameters
            # For better quality, we can try using a higher frame length
            adjusted = librosa.effects.time_stretch(audio_np, rate=float(speed))
        elif quality == 'medium':
            # Use medium quality settings
            adjusted = librosa.effects.time_stretch(audio_np, rate=float(speed))
        else:  # low quality
            # Use faster but lower quality processing
            adjusted = librosa.effects.time_stretch(audio_np, rate=float(speed))
            
        # Post-processing to reduce artifacts
        adjusted_np = adjusted.astype(np.float32, copy=False)
        
        # Apply advanced smoothing filter to reduce artifacts for all quality levels
        # This helps maintain clarity while reducing distortion
        if len(adjusted_np) > 100:
            # Use a Hann window for smoother transitions
            window_size = min(31, len(adjusted_np) // 50 + 1)  # Larger adaptive window for better smoothing
            if window_size > 1 and window_size % 2 == 1:  # Must be odd
                # Create Hann window
                hann_window = np.hanning(window_size)
                hann_window = hann_window / np.sum(hann_window)  # Normalize
                
                # Apply convolution with Hann window
                padded = np.pad(adjusted_np, window_size//2, mode='edge')
                smoothed = np.copy(adjusted_np)
                for i in range(len(smoothed)):
                    smoothed[i] = np.sum(padded[i:i+window_size] * hann_window)
                adjusted_np = smoothed
        
        result = torch.from_numpy(adjusted_np)
    else:
        # Fallback: try sox tempo via torchaudio with quality settings
        try:
            wav = audio[None, :]
            
            if quality == 'high':
                # High quality with better algorithm and additional effects
                effects = [
                    ["tempo", "-s", str(speed)],  # -s flag enables better quality algorithm
                    ["rate", str(sampling_rate)]   # Ensure consistent sample rate
                ]
            elif quality == 'medium':
                # Medium quality settings
                effects = [
                    ["tempo", "-s", str(speed)],
                    ["rate", str(sampling_rate)]
                ]
            else:  # low quality
                # Fast processing with basic algorithm
                effects = [
                    ["tempo", str(speed)],
                    ["rate", str(sampling_rate)]
                ]
                
            out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sampling_rate, effects)
            
            # Post-processing for torchaudio output
            result = out[0].to(dtype=torch.float32).cpu()
            
            # Apply advanced smoothing for all quality modes to reduce artifacts
            if len(result) > 100:
                result_np = result.numpy()
                # Use a Hann window for smoother transitions
                window_size = min(31, len(result_np) // 50 + 1)  # Larger adaptive window for better smoothing
                if window_size > 1 and window_size % 2 == 1:  # Must be odd
                    # Create Hann window
                    hann_window = np.hanning(window_size)
                    hann_window = hann_window / np.sum(hann_window)  # Normalize
                    
                    # Apply convolution with Hann window
                    padded = np.pad(result_np, window_size//2, mode='edge')
                    smoothed = np.copy(result_np)
                    for i in range(len(smoothed)):
                        smoothed[i] = np.sum(padded[i:i+window_size] * hann_window)
                    result = torch.from_numpy(smoothed.astype(np.float32))
        except Exception as e:
            logger.warning(f"Speed adjustment requested but not available (install librosa for better quality). Error: {e}")
            result = audio
    
    # Restore amplitude to original level to prevent volume loss
    if len(result) > 0:
        result_rms = torch.sqrt(torch.mean(result ** 2))
        if result_rms > 0:
            # Apply gain to restore original RMS level
            gain = original_rms / result_rms
            # Limit gain to prevent clipping
            gain = min(gain, 2.0)
            result = result * gain
            
    return result


def get_text_input_sample(
    text_content: str,
    voice_sample_path: str = None,
    voice_sample_text: str = None,
    *,
    log_text: bool = True,
):
    """Create a ChatMLSample from text content with optional voice cloning."""
    text_content = _clean_and_normalize_text(text_content)

    if log_text:
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
    
    # system_prompt = (
    #     "Generate audio following instruction.\n\n"
    #     "<|scene_desc_start|>\n"
    #     "SPEAKER0: clear voice\n"
    #     "<|scene_desc_end|>"
    # )
    
    # messages.insert(0, Message(
    #     role="system",
    #     content=system_prompt,
    # ))

    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


@click.command()
@click.option("--text-file", type=click.Path(exists=True), required=True, help="Path to the text file to generate audio from")
@click.option("--output-file", type=click.Path(), default="output_tts.wav", help="Output audio file path")
@click.option("--voice-sample", type=click.Path(), help="Path to voice sample WAV file for cloning")
@click.option("--voice-text", type=str, help="Text corresponding to the voice sample, or path to text file")
@click.option("--speed", type=float, default=1.0, help="Speech speed factor (0.5-2.0)")
@click.option("--speed-quality", type=click.Choice(['low', 'medium', 'high']), default='high', help="Quality of speed adjustment processing")
@click.option("--temperature", type=float, default=0.7, help="Generation temperature")
@click.option("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
@click.option("--chunk-max-chars", type=int, default=200, show_default=True, help="Max chars per chunk for long text; auto-split when exceeded")
@click.option("--chunk-min-chars", type=int, default=1, show_default=True, help="Min chars per chunk; very short tails will be merged")
@click.option("--silence-ms", type=int, default=20, show_default=True, help="Silence (ms) inserted between chunks when concatenating")
@click.option("--save-chunks-dir", type=click.Path(), default=None, help="If set, save per-chunk wav files into this directory for debugging")
def main(
    text_file: str,
    output_file: str,
    voice_sample: str,
    voice_text: str,
    speed: float,
    speed_quality: str,
    temperature: float,
    max_tokens: int,
    chunk_max_chars: int,
    chunk_min_chars: int,
    silence_ms: int,
    save_chunks_dir: Optional[str],
):
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

    # Long-text chunking (best-effort). We split on the *raw* text, then normalize each chunk.
    chunks = _split_long_text(text_content, max_chars=chunk_max_chars, min_chars=chunk_min_chars)
    if not chunks:
        logger.error("No valid text after preprocessing")
        return
    if len(chunks) == 1:
        logger.info("Text fits in a single chunk; generating once.")
    else:
        logger.info(f"Long text detected; split into {len(chunks)} chunks (chunk-max-chars={chunk_max_chars}).")
    
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
    all_audio: List[torch.Tensor] = []
    all_text: List[str] = []
    sampling_rate: Optional[int] = None
    total_start = time.time()

    for idx, chunk in enumerate(chunks, start=1):
        logger.info(f"Generating chunk {idx}/{len(chunks)} (chars={len(chunk)})...")
        logger.info(f"chunks: {chunk}")
        input_sample = get_text_input_sample(
            chunk,
            voice_sample,
            voice_text,
        )

        start_time = time.time()
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=input_sample,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.96,
            top_k=30,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        elapsed_time = time.time() - start_time
        logger.info(f"Chunk {idx} generation time: {elapsed_time:.2f} seconds")

        if output.audio is None:
            logger.error(f"Failed to generate audio for chunk {idx}")
            return

        # Convert to torch tensor (and resample if needed) for concatenation
        out_sr = int(output.sampling_rate)
        if sampling_rate is None:
            sampling_rate = out_sr
            audio_tensor = torch.from_numpy(output.audio.astype(np.float32, copy=False))
        elif out_sr != sampling_rate:
            logger.warning(
                f"Sampling rate changed from {sampling_rate} to {out_sr} at chunk {idx}; resampling to {sampling_rate}."
            )
            # Resample to the first chunk's sampling rate.
            wav = torch.from_numpy(output.audio.astype(np.float32, copy=False))[None, :]
            wav = torchaudio.functional.resample(wav, out_sr, sampling_rate)
            audio_tensor = wav[0]
        else:
            audio_tensor = torch.from_numpy(output.audio.astype(np.float32, copy=False))

        all_audio.append(audio_tensor)
        all_text.append(output.generated_text or "")

        if save_chunks_dir:
            chunk_path = os.path.join(save_chunks_dir, f"chunk_{idx:03d}.wav")
            torchaudio.save(chunk_path, audio_tensor[None, :].cpu(), sampling_rate)

    assert sampling_rate is not None
    merged = _concat_audio_segments(all_audio, sampling_rate, silence_ms=silence_ms)

    # Apply speed adjustment if needed (after merging, so timing stays consistent across chunks)
    if speed != 1.0:
        logger.info(f"Adjusting merged audio speed to {speed}x with quality '{speed_quality}'")
        merged = _adjust_speed(merged, sampling_rate, speed, speed_quality)

    torchaudio.save(output_file, merged[None, :], sampling_rate)
    total_elapsed = time.time() - total_start
    logger.info(f"Total generation time: {total_elapsed:.2f} seconds (chunks={len(chunks)})")
    logger.info(f"Generated text (concat):\n{''.join(all_text)}")
    logger.info(f"Saved audio to {output_file}")


if __name__ == "__main__":
    main()