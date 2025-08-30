from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class TextEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

if __name__ == "__main__":
    emb = TextEmbedder()
    sample = ["#trend1", "new skincare routine", "summer makeup challenge"]
    print(emb.embed_texts(sample).shape)
