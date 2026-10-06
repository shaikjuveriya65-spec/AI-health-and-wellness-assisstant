from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# File containing our wellness information
DATA_FILE = Path(
    "data/wellness_knowledge.txt"
)


# Folder where ChromaDB will store data
DB_DIR = "chroma_db"


# ChromaDB collection name
COLLECTION_NAME = "wellness_knowledge"


# Hugging Face embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class WellnessRAG:

    def __init__(self):

        # Load Hugging Face embedding model
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=DB_DIR
        )

        # Create or load collection
        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

        # Load knowledge base
        self._load_knowledge_base()


    def _split_text(self, text):

        # Split text using blank lines
        chunks = [
            chunk.strip()
            for chunk in text.split("\n\n")
            if chunk.strip()
        ]

        return chunks


    def _load_knowledge_base(self):

        # Do not add documents again
        if self.collection.count() > 0:
            return


        # Read wellness information
        text = DATA_FILE.read_text(
            encoding="utf-8"
        )


        # Create chunks
        chunks = self._split_text(text)


        # Create embeddings
        embeddings = (
            self.embedding_model.encode(
                chunks,
                normalize_embeddings=True
            ).tolist()
        )


        # Create unique IDs
        ids = [
            f"wellness_{i}"
            for i in range(len(chunks))
        ]


        # Create metadata
        metadatas = [
            {
                "source":
                "wellness_knowledge.txt"
            }
            for _ in chunks
        ]


        # Store everything in ChromaDB
        self.collection.add(

            ids=ids,

            documents=chunks,

            embeddings=embeddings,

            metadatas=metadatas
        )


    def search(self, question, top_k=3):

        # Convert question into embedding
        query_embedding = (
            self.embedding_model.encode(
                question,
                normalize_embeddings=True
            ).tolist()
        )


        # Search ChromaDB
        results = self.collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=top_k
        )


        # Get documents
        documents = results.get(
            "documents",
            [[]]
        )[0]


        # Get metadata
        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]


        # Join retrieved documents
        context = "\n\n".join(
            documents
        )


        # Get source names
        sources = []

        for metadata in metadatas:

            if metadata:

                source = metadata.get(
                    "source"
                )

                if source and source not in sources:
                    sources.append(source)


        return context, sources