import faiss
import numpy as np
import os


class VectorDB:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = {}
        self.vector_file = 'vectors.npy'
        self.id_file = 'ids.npy'
        self.load_vectors()  # Load existing vectors if available

    def add_vectors(self, vectors, ids):
        self.index.add(np.array(vectors))
        for i, id in enumerate(ids):
            self.id_map[len(self.id_map)] = id  # Append to map with new index
        self.save_vectors(vectors, ids)

    def search(self, query_vector, k):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return [(self.id_map[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def save_vectors(self, vectors, ids):
        # Save vectors and IDs to files
        if os.path.exists(self.vector_file):
            existing_vectors = np.load(self.vector_file)
            combined_vectors = np.vstack((existing_vectors, vectors))
        else:
            combined_vectors = np.array(vectors)

        np.save(self.vector_file, combined_vectors)

        if os.path.exists(self.id_file):
            existing_ids = np.load(self.id_file).tolist()
            combined_ids = existing_ids + ids  # No need for .tolist() here
        else:
            combined_ids = ids  # Directly use ids as it's already a list

        np.save(self.id_file, combined_ids)

    def load_vectors(self):
        # Load existing vectors and IDs from files
        if os.path.exists(self.vector_file):
            self.index.add(np.load(self.vector_file))

        if os.path.exists(self.id_file):
            self.id_map = {i: id for i, id in enumerate(np.load(self.id_file))}