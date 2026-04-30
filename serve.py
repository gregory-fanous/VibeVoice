"""
VibeVoice TTS — OpenAI-compatible API Server

Supports both models:
  --model-path microsoft/VibeVoice-Realtime-0.5B  (fast, ~3x less VRAM)
  --model-path microsoft/VibeVoice-1.5B           (higher quality)

Exposes POST /v1/audio/speech with the same interface as OpenAI TTS API.
Drop-in replacement: set TTS_OPENAI_BASE_URL=http://localhost:8001/v1
"""

import argparse
import copy
import io
import os
import threading
import time
import wave
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Voice alias mapping
# ---------------------------------------------------------------------------
VOICE_MAP = {
    "alloy": "in-Samuel_man",
    "echo": "en-Frank_man",
    "fable": "en-Emma_woman",
    "onyx": "en-Frank_man",
    "nova": "en-Grace_woman",
    "shimmer": "en-Mike_man",
    "carter": "en-Carter_man",
    "frank": "en-Frank_man",
    "alice": "en-Alice_woman",
    "emma": "en-Emma_woman",
    "maya": "en-Maya_woman",
    "grace": "en-Grace_woman",
    "mike": "en-Mike_man",
    "davis": "en-Davis_man",
    "samuel": "in-Samuel_man",
    "mary": "en-Mary_woman_bgm",
    # gpt-4o-mini-tts voices
    "marin": "en-Emma_woman",
    "cedar": "en-Carter_man",
    "ash": "en-Frank_man",
    "ballad": "en-Carter_man",
    "coral": "en-Grace_woman",
    "sage": "en-Emma_woman",
    "verse": "en-Carter_man",
}

BASE_DIR = os.path.dirname(__file__)
VOICES_DIR_WAV = os.path.join(BASE_DIR, "demo", "voices")
VOICES_DIR_STREAMING = os.path.join(BASE_DIR, "demo", "voices", "streaming_model")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
model = None
processor = None
device = None
is_streaming_model = False
voice_cache = {}  # Cache loaded .pt voice prompts
generation_lock = threading.Lock()


def get_voice_path(voice_id: str) -> str:
    """Resolve a voice ID to the correct voice file."""
    mapped = VOICE_MAP.get(voice_id.lower(), voice_id)

    if is_streaming_model:
        # Streaming model uses .pt files
        pt_path = os.path.join(VOICES_DIR_STREAMING, f"{mapped}.pt")
        if os.path.exists(pt_path):
            return pt_path
        # Case-insensitive scan
        for f in os.listdir(VOICES_DIR_STREAMING):
            if f.lower().replace(".pt", "") == mapped.lower():
                return os.path.join(VOICES_DIR_STREAMING, f)
        default = os.path.join(VOICES_DIR_STREAMING, "en-Carter_man.pt")
        print(f"[WARN] Voice '{voice_id}' not found, using default: en-Carter_man")
        return default
    else:
        # 1.5B model uses .wav files
        wav_path = os.path.join(VOICES_DIR_WAV, f"{mapped}.wav")
        if os.path.exists(wav_path):
            return wav_path
        for f in os.listdir(VOICES_DIR_WAV):
            if f.lower().replace(".wav", "") == mapped.lower():
                return os.path.join(VOICES_DIR_WAV, f)
        default = os.path.join(VOICES_DIR_WAV, "en-Carter_man.wav")
        print(f"[WARN] Voice '{voice_id}' not found, using default: en-Carter_man")
        return default


def load_cached_voice(voice_path: str):
    """Load and cache a .pt voice prompt for the streaming model."""
    if voice_path in voice_cache:
        return voice_cache[voice_path]
    target = device if device != "cpu" else "cpu"
    data = torch.load(voice_path, map_location=target, weights_only=False)
    voice_cache[voice_path] = data
    return data


def load_model(model_path: str, dev: str, num_steps: int = 10):
    """Load VibeVoice model and processor (auto-detects 0.5B vs 1.5B)."""
    global model, processor, device, is_streaming_model
    device = dev

    # Detect model variant
    is_streaming_model = "realtime" in model_path.lower() or "0.5b" in model_path.lower()

    if is_streaming_model:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

        print(f"Loading VibeVoice Realtime 0.5B processor from {model_path}...")
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        load_dtype = torch.float32 if dev == "mps" else torch.bfloat16
        attn = "sdpa"
        print(f"Loading VibeVoice Realtime 0.5B model on {device} ({attn}, dtype={load_dtype})...")

        if dev == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path, torch_dtype=load_dtype, attn_implementation=attn, device_map=None,
            )
            model.to("mps")
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path, torch_dtype=load_dtype, device_map="auto", attn_implementation=attn,
            )

        model.eval()
        model.set_ddpm_inference_steps(num_steps=num_steps)
        print(f"Realtime 0.5B model loaded. Inference steps: {num_steps}")
    else:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        print(f"Loading VibeVoice 1.5B processor from {model_path}...")
        processor = VibeVoiceProcessor.from_pretrained(model_path)

        print(f"Loading VibeVoice 1.5B model on {device} (SDPA attention)...")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        )
        if device != "cpu":
            model = model.to(device)
        model.eval()
        model.set_ddpm_inference_steps(num_steps=num_steps)
        print(f"1.5B model loaded. Inference steps: {num_steps}")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: str = "vibevoice"
    input: str
    voice: str = "alloy"
    speed: float = 1.0
    response_format: Optional[str] = "wav"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app(model_path: str, dev: str, num_steps: int) -> FastAPI:
    model_label = "realtime-0.5b" if ("realtime" in model_path.lower() or "0.5b" in model_path.lower()) else "1.5b"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_model(model_path, dev, num_steps)
        yield

    app = FastAPI(title="VibeVoice TTS Server", version="2.0.0", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": f"vibevoice-{model_label}", "object": "model", "owned_by": "microsoft", "type": "tts"}],
        }

    @app.post("/v1/audio/speech")
    def generate_speech(req: SpeechRequest):
        if not req.input or not req.input.strip():
            raise HTTPException(status_code=400, detail="Input text is required")
        if len(req.input) > 10000:
            raise HTTPException(status_code=400, detail="Input text too long (max 10000 chars)")

        voice_path = get_voice_path(req.voice)
        voice_name = os.path.splitext(os.path.basename(voice_path))[0]
        print(f"[TTS] voice={req.voice} → {voice_name} | {len(req.input)} chars | model={model_label}")

        script = f"Speaker 1: {req.input}"

        try:
            start = time.time()

            with generation_lock:
                if is_streaming_model:
                    # --- 0.5B Realtime model path ---
                    cached_prompt = load_cached_voice(voice_path)
                    inputs = processor.process_input_with_cached_prompt(
                        text=script,
                        cached_prompt=cached_prompt,
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    target = device if device != "cpu" else "cpu"
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(target)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=1.5,
                        tokenizer=processor.tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                        all_prefilled_outputs=copy.deepcopy(cached_prompt),
                    )
                else:
                    # --- 1.5B model path ---
                    inputs = processor(
                        text=[script],
                        voice_samples=[[voice_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=1.3,
                        tokenizer=processor.tokenizer,
                        generation_config={"do_sample": False},
                        verbose=False,
                    )

            elapsed = time.time() - start

            if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                raise HTTPException(status_code=500, detail="No audio generated")

            audio_tensor = outputs.speech_outputs[0]
            sample_rate = 24000

            if hasattr(audio_tensor, "cpu"):
                audio_np = audio_tensor.cpu().float().numpy()
            else:
                audio_np = audio_tensor

            audio_np = audio_np.squeeze()
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak
            audio_int16 = (audio_np * 32767).astype(np.int16)

            duration = len(audio_int16) / sample_rate
            print(f"[TTS] Generated {duration:.1f}s audio in {elapsed:.1f}s (RTF: {elapsed/duration:.2f}x)")

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            return Response(
                content=buf.getvalue(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                    "X-Audio-Duration": f"{duration:.2f}",
                    "X-Generation-Time": f"{elapsed:.2f}",
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/audio/voices")
    async def list_voices():
        voices = []
        seen = set()
        for alias, vv_name in VOICE_MAP.items():
            if alias in seen:
                continue
            seen.add(alias)
            voices.append({"id": alias, "name": alias.title(), "vibevoice_voice": vv_name})
        return {"voices": voices}

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": f"vibevoice-{model_label}", "device": device}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VibeVoice TTS API Server")
    parser.add_argument("--model-path", default="microsoft/VibeVoice-Realtime-0.5B", help="HuggingFace model ID")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--num-steps", type=int, default=2, help="DDPM inference steps (2=fastest, 5=balanced, 10=quality)")
    args = parser.parse_args()

    app = create_app(args.model_path, args.device, args.num_steps)
    uvicorn.run(app, host=args.host, port=args.port)
