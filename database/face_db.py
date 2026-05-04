import json 
import logging
import os
from typing import Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

class FaceDatabase:
    """
    FAISS-backed face embed Database using IndexFlatIP (Inner Product) on L2-normalized vectors
    for Consine Similarity search
    """
    def __init__(self, embedding_size: int = 512, db_path: str = "./database/face_database") -> None:
        """
        Init the database
        Args:
            embedding_size: default 512
            db_path: Dir to persist FAISS index and Metadata
        """
        self.embedding_size = embedding_size
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.meta_file = os.path.join(db_path, "metadata.json")

        os.makedirs(db_path, exist_ok=True)
        
        self.index = faiss.IndexFlatIP(embedding_size)

        # list of names: metadata[i] refers to index row i
        self.metadata: list[str] = []
    
    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """
        L2-normalise an embedding vector
        """
        v = vec.astype(np.float32).ravel() # ravel is known as in-place flatten -> no need memory allocation -> faster
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    def add_face(self, embedding: np.ndarray, name: str) -> None:
        """
        Add a single face embed to the db

        Args:
            embedding: face embedding vector
            name: indentituy label for that embedding 
        """
        vec = self._normalise(embedding).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append(name)
    
    def search(self, embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """
        Find the closest Identity (vector/embed) for a query embedding
        
        Args:
            embedding: Query embedding
            threshold: min cosine similarity to accept a match

        Returns:
            Tuple of (name, similarity) for the best match
        """
        if self.index.ntotal == 0:
            return ("Unknown", 0.0)
        
        vec = self._normalise(embedding).reshape(1, -1)
        similarities, indices = self.index.search(vec, 1) # return 1 result only
        
        similarity = float(similarities[0][0])
        idx = int(indices[0][0])

        if similarity > threshold and idx < len(self.metadata):
            return (self.metadata[idx], similarity)

        return ("Unknown", similarity)
    
    def batch_add_faces(self, embeddings: list[np.ndarray], names: list[str]) -> None:
        """
        Add multiple face embeddings
        """
        if not embeddings:
            return
        mat = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)

        mat /= norms
        self.index.add(mat)
        self.metadata.extend(names)

    def batch_search(self, embeddings: list[np.ndarray], threshold: float = 0.4) -> list[Tuple[str, float]]:
        """
        Search closest Identities for multiple embeddings in a single FAISS call

        Returns:
            List of (name, similarity) tuples in input order
        """
        if not embeddings:
            return []
        
        if self.index.ntotal == 0:
            return [("Unknown", 0.0)] * len(embeddings)
        
        # Stack and Normalise all Queries
        mat = np.stack(embeddings).astype(np.float32) # (N, D)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) # axis 1 -> across row -> norm of each embedding
        norms = np.maximum(norms, 1e-10)
        mat /= norms

        similarities, indices = self.index.search(mat, 1)

        results: list[Tuple[str, float]] = []

        for sim_row, idx_row in zip(similarities, indices):
            similarity = sim_row[0]
            idx = int(idx_row[0])

            if similarity > threshold and idx < len(self.metadata):
                results.append((self.metadata[idx], similarity))
            else:
                results.append(("Unknown", similarity))
        
        return results

    def save(self) -> None:
        """
        Persist FAISS Index and Metadata to disk
        """
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully save Face Database with {self.index.ntotal} faces")
        except Exception as e:
            logger.error(f"Fail to save Face Database: {e}")
            raise

    def load(self) -> bool:
        """
        Load Faise Index and Metadata from disk
        
        Returns:
            True if load succeed, else False
        """
        if not (os.path.exists(self.index_file) and os.path.exists(self.meta_file)):
            return False
        try:
            # load into temp variables to avoid partial failure
            loaded_index = faiss.read_index(self.index_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                loaded_meatadata: list[str] = json.load(f)

            # assign only after both succeeded
            self.index = loaded_index
            self.metadata = loaded_meatadata  
            logger.info(f"Sucessfully load Face Database with {self.index.ntotal} faces")
            return True
        except Exception as e:
            logger.error(f"Fail to load Face Database: {e}")
            return False
