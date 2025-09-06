# src/huggingface_features.py
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# You can replace this with any sentence-transformer or lightweight model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model once (GPU if available)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _model.to(_device)
_model.eval()

@torch.no_grad()
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts/hashtags using a Hugging Face transformer model.

    Args:
        texts (List[str]): List of text strings (e.g., hashtags or video titles)

    Returns:
        np.ndarray: Matrix of embeddings (len(texts), hidden_size)
    """
    if not texts:
        return np.empty((0, _model.config.hidden_size))

    enc = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(_device)
    outputs = _model(**enc)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

if __name__ == "__main__":
    samples = ["#AI", "#MachineLearning", "Deep learning revolution"]
    vecs = embed_texts(samples)
    print("Embeddings shape:", vecs.shape)
    print("First vector (truncated):", vecs[0][:5])
