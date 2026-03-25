# Voice Changer Demo

Real-time voice cloning demo. Speak into your mic (or type text) and hear it played back in a cloned voice with streaming audio output.

**Flow:** mic input -> Whisper transcription -> Qwen3-TTS voice clone -> streaming audio playback

## Prerequisites

### 1. vllm-omni server

The app connects to a running vllm-omni instance serving the Qwen3-TTS model over WebSocket.

```sh
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --trust-remote-code --port 8000
```

### 2. Python dependencies

```sh
pip install gradio websockets soundfile numpy openai-whisper
```

### 3. Whisper model

The app loads the Whisper `base` model (~140MB) locally for transcription. It downloads automatically on first run.

## Usage

```sh
python examples/demos/voice_changer/app.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--server-url` | `localhost:8000` | Host and port of the vllm-omni server |
| `--host` | `0.0.0.0` | Gradio bind address |
| `--port` | `3000` | Gradio UI port |
| `--share` | off | Create a public Gradio share link |

Then open `http://localhost:3000` in a browser.

## How it works

1. Upload or record a 5-15 second reference audio clip of the target voice
2. Use the **Voice-to-Voice** tab to speak and hear it re-synthesized in the cloned voice, or the **Text-to-Voice** tab to type text
3. Audio streams back in chunks for low-latency playback

No API keys or cloud services are required. Whisper runs locally and TTS runs on the self-hosted vllm-omni server.
