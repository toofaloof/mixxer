# 🎶 AI Mixing Copilot - "Mixxer"

An AI-powered Gradio app that separates audio into stems, applies smart mixing adjustments (EQ, compression, FX), and produces a remastered track. Inspired by tools like Suno.ai.

## 🚀 Features

- Demucs-based stem separation
- AI-assisted mixing with OpenAI + audio analysis
- Built with Gradio for web interaction
- Mastering pipeline with compression, EQ, stereo spread, and soft limiting
- 3min song takes ~40 seconds to get each stem and the whole song mixed and mastered on a RTX 3090, 24Gb

## 🧠 Requirements
- Open-AI-API-Key
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Coming Soon:
Developing a system-level integration for FL Studio that maps all accessible plugin parameters to MIDI controls, enabling real-time sound manipulation driven by natural language prompts controlled by a sound-classifier model.
