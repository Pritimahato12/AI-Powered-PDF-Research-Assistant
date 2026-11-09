from sentence_transformers import SentenceTransformer


def load_embedding_model():

  model = SentenceTransformer(
      "sentence-transformers/all-MiniLM-L6-v2"
  )

  return model

SentenceTransformer(
  "sentence-transformers/all-MiniLM-L6-v2"
)

def generate_embeddings(model, chunks):

  texts = [
    chunk["text"]
    for chunk in chunks
  ]

  embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
  )

  return embeddings