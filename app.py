import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
from pydub import AudioSegment, effects
import gradio as gr
import openai
import json
import librosa
import numpy as np
from scipy.signal import butter, lfilter

openai.api_key = os.getenv("OPENAI_API_KEY")


## bandpass
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

    
## eq
def apply_eq(audio_data, sr, eq_settings):
    bands = {
        "low": (20, 200),
        "mid": (200, 2000),
        "high": (2000, 10000)
    }

    output = np.zeros_like(audio_data)

    for band, gain in eq_settings.items():
        gain = float(gain)  # 🔑 convert to float here
        lowcut, highcut = bands[band]
        b, a = butter_bandpass(lowcut, highcut, sr)
        filtered = lfilter(b, a, audio_data)
        output += filtered * (10 ** (gain / 20))  # linear gain

    return output / np.max(np.abs(output))

"""def compress_audio(audio, threshold_db=-20.0, ratio=4.0, makeup_gain_db=6.0):
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-6)

    # Apply compression
    over_threshold = audio_db > threshold_db
    gain_reduction = np.where(over_threshold,
                              (audio_db - threshold_db) * (1 - 1 / ratio),
                              0)

    # Convert back to linear gain
    gain_factor = 10 ** (-gain_reduction / 20.0)
    compressed = audio * gain_factor

    # Apply makeup gain
    compressed *= 10 ** (makeup_gain_db / 20.0)

    # Normalize
    return compressed / np.max(np.abs(compressed))"""


## compressor
def compress_audio(audio, threshold_db=-20.0, ratio=4.0, makeup_gain_db=6.0):
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-6)

    # Apply compression
    over_threshold = audio_db > threshold_db
    gain_reduction = np.where(over_threshold,
                              (audio_db - threshold_db) * (1 - 1 / ratio),
                              0)

    # Convert back to linear gain
    gain_factor = 10 ** (-gain_reduction / 20.0)
    compressed = audio * gain_factor

    # Apply makeup gain
    compressed *= 10 ** (makeup_gain_db / 20.0)

    # Normalize
    return audio

## compressor needs more finetuning and more parameters for ai-agent to proper work


## Audio Spectroscopy
def analyze_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Compute RMS (Root Mean Square) energy for volume
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)

    # Compute Spectral Centroid for brightness
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_spectral_centroid = np.mean(spectral_centroid)

    # Compute Spectral Bandwidth for frequency spread
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    avg_spectral_bandwidth = np.mean(spectral_bandwidth)

    # Compute Zero-Crossing Rate for noisiness
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
    avg_zero_crossing_rate = np.mean(zero_crossing_rate)

    # Compute Dynamic Range
    dynamic_range = np.max(rms) - np.min(rms)

    return {
        'avg_rms': avg_rms,
        'avg_spectral_centroid': avg_spectral_centroid,
        'avg_spectral_bandwidth': avg_spectral_bandwidth,
        'avg_zero_crossing_rate': avg_zero_crossing_rate,
        'dynamic_range': dynamic_range
    }


## AI-API
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_preset_adjustments(input_data):
    client = openai.OpenAI()

    user_prompt = input_data['user_prompt']
    audio_features = input_data['audio_features']
    prompt = f"""
    You are an AI audio mixing assistant. A user said: "{user_prompt}".
    The audio analysis yielded the following features: {audio_features}.
    Based on this, suggest adjustments to audio presets for vocals, drums, bass, and other.
    Respond in JSON format like:
    {{
      "vocals": {{"eq": {{"low": X, "mid": Y, "high": Z}}, "compression": {{"threshold": A, "ratio": B, "makeup": C}}, "gain": G}},
      ...
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )

    # Extract and parse JSON
    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("Could not parse AI response into JSON")


def sanitize_presets(presets):
    def to_number(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0  # fallback if it's not a number

    for stem, settings in presets.items():
        for section in settings:
            if isinstance(settings[section], dict):
                settings[section] = {k: to_number(v) for k, v in settings[section].items()}
            else:
                settings[section] = to_number(settings[section])
    return presets


# ## Main-Code
OUTPUT_DIR = "output/stems"
MIX_REPORT = "output/mix_report.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- AI Mix Presets ----
MIX_PRESETS = {
    "vocals": {"gain": 4, "pan": 0, "reverb": True},
    "drums": {"gain": 2, "pan": -40, "reverb": False},
    "bass": {"gain": 0, "pan": 20, "reverb": False},
    "other": {"gain": -2, "pan": 30, "reverb": True},
}

EQ_PRESETS = {
    "vocals": {"low": -6, "mid": 2, "high": 4},
    "drums": {"low": 4, "mid": 2, "high": 3},
    "bass": {"low": 6, "mid": -2, "high": -6},
    "other": {"low": 0, "mid": 1, "high": 2},
}

COMPRESSION_PRESETS = {
    "vocals": {"threshold": -22, "ratio": 3.0, "makeup": 5},
    "drums": {"threshold": -18, "ratio": 6.0, "makeup": 8},
    "bass": {"threshold": -20, "ratio": 4.0, "makeup": 6},
    "other": {"threshold": -24, "ratio": 2.0, "makeup": 4},
}


def apply_overrides(overrides):
    for stem, values in overrides.items():
        if stem not in MIX_PRESETS:
            print(f"[⚠️] Unknown stem: '{stem}' — skipping.")
            continue
        if "eq" in values:
            EQ_PRESETS[stem].update(values["eq"])
        if "compression" in values:
            COMPRESSION_PRESETS[stem].update(values["compression"])
        if "gain" in values:
            MIX_PRESETS[stem]["gain"] = values["gain"]

# ---- Core Functions ----
model = get_model(name="htdemucs")
model.to('cuda')

def separate_stems(audio_path):
    wav, sr = torchaudio.load(audio_path)
    wav = wav.to('cuda')
    if wav.shape[0] == 1:
        wav = torch.cat([wav, wav], dim=0)
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        sources = apply_model(model, wav)

    sources = sources[0]
    stem_paths = []
    for i, name in enumerate(model.sources):
        path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        sf.write(path, sources[i].cpu().T.numpy(), model.samplerate)
        stem_paths.append((name, path))
    return stem_paths

def apply_effects_to_stem(stem_name, path):
    print(f"[🎚️] Applying FX + EQ + Compression to {stem_name}...")

    # --- Load & EQ ---
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    eq_settings = EQ_PRESETS.get(stem_name, {"low": 0, "mid": 0, "high": 0})
    data_eq = apply_eq(data, sr, eq_settings)

    # --- Compression ---
    comp = COMPRESSION_PRESETS.get(stem_name, {})
    data_compressed = compress_audio(
        data_eq,
        threshold_db=comp.get("threshold", -20),
        ratio=comp.get("ratio", 4),
        makeup_gain_db=comp.get("makeup", 6)
    )

    # Save processed version temporarily
    sf.write(path, data_compressed, sr)

    # --- Pydub FX (Gain, Pan, Reverb) ---
    sound = AudioSegment.from_file(path)
    preset = MIX_PRESETS.get(stem_name, {"gain": 0, "pan": 0, "reverb": False})
    sound += preset["gain"]
    sound = sound.pan(preset["pan"] / 100.0)

    if preset["reverb"]:
        tail = sound.low_pass_filter(4000).fade_out(500).apply_gain(-6)
        sound = sound.overlay(tail, position=50)

    sound.export(path, format="wav")


def mix_report(stems):
    report = "🧠 AI Mix Suggestions\n---------------------\n"
    for stem, _ in stems:
        preset = MIX_PRESETS.get(stem, {})
        eq = EQ_PRESETS.get(stem, {})
        report += f"{stem}:\n"
        report += f"  ➤ Gain: {preset.get('gain', 0)} dB\n"
        report += f"  ➤ Pan: {preset.get('pan', 0)}%\n"
        report += f"  ➤ Reverb: {'Yes' if preset.get('reverb') else 'No'}\n"
        report += f"  ➤ EQ: Low {eq.get('low', 0)} dB, Mid {eq.get('mid', 0)} dB, High {eq.get('high', 0)} dB\n\n"
        comp = COMPRESSION_PRESETS.get(stem, {})
        report += f"  ➤ Compression: Threshold {comp.get('threshold')} dB, Ratio {comp.get('ratio')} : 1, Makeup {comp.get('makeup')} dB\n\n"

    return report


# Optional panning config (based on stem name)
PANNING_MAP = {
    "vocals": 0.0,     # Center
    "drums": -0.2,     # Slight left
    "bass": 0.2,       # Slight right
    "other": 0.4       # More right
}

def soft_limiter(audio, threshold_db=-1.0):
    # Hard limiting to avoid clipping above threshold
    peak_db = audio.max_dBFS
    if peak_db > threshold_db:
        reduction = peak_db - threshold_db
        return audio.apply_gain(-reduction)
    return audio

def merge_stems_to_master(stem_paths, output_path="output/remastered.wav"):
    print("[🎼] Merging stems into a master mix with AGC, stereo, and limiting...")

    stems = []

    # Load, match length, normalize volume, pan
    max_length = 0
    for name, path in stem_paths:
        audio = AudioSegment.from_file(path)

        # AGC: normalize each stem individually
        # audio = effects.normalize(audio)

        # Pan according to predefined stereo spread (if stereo)
        pan = PANNING_MAP.get(name, 0.0)
        audio = audio.pan(pan)

        stems.append(audio)
        max_length = max(max_length, len(audio))

    # Pad all stems to same length
    stems = [s + AudioSegment.silent(duration=(max_length - len(s))) for s in stems]

    # Overlay stems
    final_mix = stems[0]
    for stem in stems[1:]:
        final_mix = final_mix.overlay(stem)

    # Apply soft limiter
    final_mix = soft_limiter(final_mix, threshold_db=-1.0)

    # Export final
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_mix.export(output_path, format="wav")
    print(f"[✅] Master mix exported: {output_path}")
    return output_path
# ---- Gradio Interface Logic ----

def process_audio(file, prompt):
    print(f"[🚀] Processing audio with user prompt: '{prompt}'")

    # Analyze the input audio
    audio_features = analyze_audio(file.name)
    print(f"[🔍] Extracted audio features: {audio_features}")

    # Combine user prompt with audio features for AI processing
    combined_input = {
        'user_prompt': prompt,
        'audio_features': audio_features
    }

    try:
        # Assuming get_preset_adjustments can handle combined input
        adjustments = get_preset_adjustments(combined_input)
        adjustments = sanitize_presets(adjustments)
        apply_overrides(adjustments)
        print("[✅] Applied AI-generated mix adjustments.")
    except Exception as e:
        print(f"[⚠️] Failed to apply AI adjustments: {e}")

    # Proceed with stem separation and further processing
    stems = separate_stems(file.name)
    output_files = []
    previews = {}

    for name, path in stems:
        apply_effects_to_stem(name, path)
        output_files.append(path)
        previews[name] = path

    master_mix_path = merge_stems_to_master(stems)
    output_files.append(master_mix_path)
    previews["master"] = master_mix_path

    report_text = mix_report(stems)
    return (
        output_files,
        report_text,
        previews.get("vocals"),
        previews.get("drums"),
        previews.get("bass"),
        previews.get("other"),
        previews.get("master")
    )





## Gradio-UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎶 Audio AI Copilot Demo\nUpload a track and describe how you want it to sound!")

    with gr.Row():
        audio_input = gr.File(label="🎧 Upload Audio", file_types=[".mp3", ".wav", ".flac"])
        user_prompt = gr.Textbox(label="📝 Describe the Sound You Want (e.g. 'make it cleaner and louder')")
        process_button = gr.Button("✨ Mix & Separate")

    with gr.Row():
        output_files = gr.File(label="📂 Download Stems + Master", file_types=[".wav"], file_count="multiple")
        mix_output = gr.Textbox(label="🧠 Mix Report", lines=10)

    with gr.Row():
        preview_vocals = gr.Audio(label="🎙️ Vocals")
        preview_drums = gr.Audio(label="🥁 Drums")
        preview_bass = gr.Audio(label="🎸 Bass")
        preview_other = gr.Audio(label="🎹 Other")
        preview_master = gr.Audio(label="🎼 Remastered Track")

    process_button.click(
        fn=process_audio,
        inputs=[audio_input, user_prompt],
        outputs=[
            output_files,
            mix_output,
            preview_vocals,
            preview_drums,
            preview_bass,
            preview_other,
            preview_master
        ]
    )


# ---- Run the app ----
if __name__ == "__main__":
    demo.launch()
