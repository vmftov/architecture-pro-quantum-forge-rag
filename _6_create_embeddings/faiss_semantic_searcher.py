import faiss
from _6_create_embeddings.text_encoder import TextEncoder


class FaissSemanticSearcher:
    def __init__(self, encoder: TextEncoder):
        self.faiss_index = faiss.IndexFlatIP(encoder.get_embedding_dimension())
        # faiss_index = faiss.IndexHNSWFlat(dim, M) # M - число связей на узел (больше = точнее, но больше памяти)

    def append_embeddings(self, embeddings):
        self.faiss_index.add(embeddings)

    def search(self, embeddings):
        return self.faiss_index.search(embeddings, k=3)
