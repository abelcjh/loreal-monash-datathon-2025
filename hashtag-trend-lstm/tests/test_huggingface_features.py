from src.huggingface_features import embed_texts

def test_embed_texts():
    texts = ["Hello world", "#AI is trending"]
    embeddings = embed_texts(texts)
    
    # Shape check: 2 texts -> 2 embeddings
    assert embeddings.shape[0] == 2
    
    # Each embedding should have non-zero length
    assert embeddings.shape[1] > 0
    
    # Embeddings should be floats
    assert embeddings.dtype.kind == "f"