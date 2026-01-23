#!/usr/bin/env python3
"""API service for multi-speaker text to speech generation with voice cloning and speed control."""

import os
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from text_to_speech_multi_speaker import (
    parse_speaker_text,
    get_speaker_voices,
    get_text_input_sample_multi_speaker,
    HiggsAudioServeEngine,
    logger
)

app = FastAPI(title="Higgs Audio Multi-Speaker TTS API", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for the model
serve_engine = None
device = None

class SpeakerVoiceMapping(BaseModel):
    speaker_tag: str
    voice_sample_path: str
    voice_text_path: str

class TTSMultiSpeakerRequest(BaseModel):
    text_content: str
    speaker_voice_mappings: List[SpeakerVoiceMapping]
    speed: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 2048
    chunk_max_chars: int = 200
    chunk_min_chars: int = 20
    silence_ms: int = 20

class TTSMultiSpeakerResponse(BaseModel):
    message: str
    output_file: str
    speakers_detected: List[str]

def initialize_model():
    """Initialize the Higgs Audio model."""
    global serve_engine, device
    
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
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts."""
    logger.info("Initializing Higgs Audio model...")
    initialize_model()
    logger.info("Model initialization complete.")

@app.get("/")
async def root(request: Request):
    """Root endpoint - serves the web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {"message": "Higgs Audio Multi-Speaker TTS API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": serve_engine is not None}

@app.post("/tts/multi-speaker", response_model=TTSMultiSpeakerResponse)
async def generate_multi_speaker_speech(request: TTSMultiSpeakerRequest):
    """Generate speech from multi-speaker text with voice cloning."""
    try:
        # Parse speaker texts
        speaker_texts = parse_speaker_text(request.text_content)
        if not speaker_texts:
            raise HTTPException(status_code=400, detail="No valid speaker text found")
        
        # Create speaker voice mapping dictionary
        speaker_voice_mapping = {}
        for mapping in request.speaker_voice_mappings:
            speaker_voice_mapping[mapping.speaker_tag] = (
                mapping.voice_sample_path,
                mapping.voice_text_path
            )
        
        # Create temporary output file
        output_file = f"/tmp/tts_output_{uuid.uuid4()}.wav"
        
        # Process the text using the existing functionality
        from text_to_speech_multi_speaker import main as tts_main
        import sys
        from io import StringIO
        
        # Capture stdout to prevent printing
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Call the main function with the appropriate arguments
            # We need to convert our request to command line arguments
            import click.testing
            from text_to_speech_multi_speaker import main
            
            # Create a temporary text file with the content
            text_file = f"/tmp/tts_input_{uuid.uuid4()}.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(request.text_content)
            
            # Prepare speaker voices arguments
            speaker_voices_args = []
            for mapping in request.speaker_voice_mappings:
                speaker_voices_args.append(
                    f"{mapping.speaker_tag}:{mapping.voice_sample_path},{mapping.voice_text_path}"
                )
            
            # Call the main function with patched arguments
            # Since we can't directly call click commands this way, we'll reimplement the core logic
            
            # Reimplement the core logic from text_to_speech_multi_speaker.py
            from text_to_speech_multi_speaker import (
                _split_long_text_for_speaker,
                _concat_audio_segments,
                _adjust_speed,
                encode_base64_content_from_file
            )
            import torch
            import torchaudio
            import numpy as np
            import time
            
            # Validate speed parameter
            speed = max(0.5, min(2.0, request.speed))
            
            logger.info(f"Parsed {len(speaker_texts)} speaker segments")
            for speaker, text in speaker_texts:
                logger.info(f"{speaker}: {len(text)} characters")
            
            if not serve_engine:
                raise HTTPException(status_code=500, detail="Model not initialized")
            
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
                
                # Split long text for this speaker
                text_chunks = _split_long_text_for_speaker(
                    speaker_text, 
                    max_chars=request.chunk_max_chars, 
                    min_chars=request.chunk_min_chars
                )
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
                    output = serve_engine.generate(
                        chat_ml_sample=input_sample,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
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
            
                # Concatenate all chunks for this speaker
                if speaker_audio_segments:
                    speaker_audio = _concat_audio_segments(speaker_audio_segments, sampling_rate, silence_ms=0)
                    all_audio_segments.append(speaker_audio)
            
            total_elapsed_time = time.time() - total_start_time
            logger.info(f"Total generation time: {total_elapsed_time:.2f} seconds")
            
            # Concatenate all speakers with silence between them
            if not all_audio_segments:
                raise HTTPException(status_code=500, detail="No audio segments generated")
                
            assert sampling_rate is not None
            final_audio = _concat_audio_segments(all_audio_segments, sampling_rate, silence_ms=request.silence_ms)
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                logger.info(f"Adjusting final audio speed to {speed}x")
                final_audio = _adjust_speed(final_audio, sampling_rate, speed)
            
            # Save the final audio
            torchaudio.save(output_file, final_audio[None, :], sampling_rate)
            logger.info(f"Saved audio to {output_file}")
            
            # Clean up temporary files
            if os.path.exists(text_file):
                os.remove(text_file)
            
        finally:
            sys.stdout = old_stdout
        
        # Return success response with relative path
        speakers_detected = [speaker for speaker, _ in speaker_texts]
        # 返回相对于/tmp目录的文件名，而不是完整路径
        relative_filename = os.path.basename(output_file)
        return TTSMultiSpeakerResponse(
            message="Successfully generated speech",
            output_file=f"/tmp/{relative_filename}",
            speakers_detected=speakers_detected
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")
@app.get("/audio/{file_path}")
async def get_audio(file_path: str):
    """Serve generated audio files."""
    full_path = f"/tmp/{file_path}"
    logger.info(f"Attempting to serve audio file: {full_path}")
    if os.path.exists(full_path) and full_path.endswith('.wav'):
        logger.info(f"Serving audio file: {full_path}")
        return FileResponse(full_path, media_type='audio/wav')
    else:
        logger.error(f"Audio file not found: {full_path}")
        raise HTTPException(status_code=404, detail="File not found")



# 添加一个新的端点来直接提供/tmp目录下的文件
@app.get("/tmp/{file_path}")
async def get_tmp_audio(file_path: str):
    """Serve generated audio files from /tmp directory."""
    full_path = f"/tmp/{file_path}"
    if os.path.exists(full_path) and full_path.endswith('.wav'):
        return FileResponse(full_path, media_type='audio/wav')
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)