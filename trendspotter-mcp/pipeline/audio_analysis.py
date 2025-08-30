# src/audio_embed.py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf

# YAMNet model for audio embeddings
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

class AudioEmbedder:
    def __init__(self):
        self.model = hub.load(YAMNET_HANDLE)
        # yamnet returns scores, embeddings, spectrogram
    def embed_file(self, filepath):
        wav_data, sr = sf.read(filepath, dtype='float32')
        if sr != 16000:
            # resample to 16k if needed
            wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        scores, embeddings, spectrogram = self.model(wav_data)
        # embeddings is (frames, 1024). Average pool
        emb = np.mean(embeddings.numpy(), axis=0)
        return emb

# Optional: Whisper via huggingface transformers (speech->text)
# You can use openai/whisper-small or whisper via transformers pipeline
from transformers import pipeline
def speech_to_text_whisper(audio_path, model_name="openai/whisper-small"):
    pipe = pipeline("automatic-speech-recognition", model=model_name)
    res = pipe(audio_path)
    return res.get("text", "")
