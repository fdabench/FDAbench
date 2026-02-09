"""
Vector Index Builder for FDABench

This utility builds vector indices from document directories using OpenAI Embeddings + FAISS.
The generated indices are compatible with FDABench's VectorSearchTool.

No LlamaIndex dependency - uses OpenAI API directly with FAISS for vector storage.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
import logging
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss: pip install faiss-cpu or faiss-gpu")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 50


class TextChunker:
    """Simple text chunker with sentence-aware splitting."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        # Simple sentence-aware chunking
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': metadata.copy()
                })

                # Keep overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                current_chunk = [overlap_text] if self.chunk_overlap > 0 else []
                current_length = len(overlap_text) if self.chunk_overlap > 0 else 0

            current_chunk.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'metadata': metadata.copy()
            })

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple approach)."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class OpenAIEmbedder:
    """OpenAI Embedding wrapper with batching and concurrency support."""

    # Max tokens per API call is 8192, reserve some buffer
    MAX_TOKENS_PER_CALL = 8000
    # Approximate: 1 token ≈ 4 chars for English, 8000 tokens ≈ 32000 chars
    MAX_CHARS_PER_TEXT = 30000  # ~7500 tokens, safe under 8192 limit

    def __init__(self, api_key: str = None, model: str = EMBEDDING_MODEL, max_workers: int = 50):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed explicitly")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_workers = max_workers

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits."""
        if len(text) > self.MAX_CHARS_PER_TEXT:
            return text[:self.MAX_CHARS_PER_TEXT]
        return text

    def _embed_single(self, text: str, idx: int) -> tuple:
        """Embed a single text. Returns (idx, embedding)."""
        # Clean and truncate
        text = text.strip() if text.strip() else " "
        text = self._truncate_text(text)

        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )

        return (idx, response.data[0].embedding)

    def embed_texts(self, texts: List[str], batch_size: int = BATCH_SIZE) -> tuple:
        """
        Embed a list of texts using OpenAI API with concurrency.
        Each text is embedded individually for maximum parallelism.
        Failed embeddings are skipped and logged.

        Args:
            texts: List of texts to embed
            batch_size: Ignored (kept for compatibility), uses single-text calls

        Returns:
            tuple: (embeddings array, valid_indices list)
                - embeddings: numpy array of successful embeddings
                - valid_indices: list of original indices that succeeded
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total_texts = len(texts)
        results = {}  # idx -> embedding
        failed_count = 0

        # Process each text individually with max concurrency
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._embed_single, text, idx): idx
                for idx, text in enumerate(texts)
            }

            with tqdm(total=total_texts, desc="Embedding texts") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        idx, embedding = future.result()
                        results[idx] = embedding
                    except Exception as e:
                        logger.warning(f"Skipping chunk {idx}, embedding failed: {e}")
                        failed_count += 1
                    pbar.update(1)

        if failed_count > 0:
            logger.warning(f"Total failed embeddings: {failed_count}/{total_texts}")

        # Return ordered results with valid indices
        valid_indices = sorted(results.keys())
        embeddings = [results[i] for i in valid_indices]

        return np.array(embeddings, dtype=np.float32), valid_indices

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)


class FAISSVectorIndex:
    """FAISS-based vector index with chunk storage."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.index = None
        self.chunks = []  # List of chunk dicts with 'id', 'text', 'metadata'
        self._id_to_idx = {}  # Map chunk_id to index position

    def build(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Build FAISS index from embeddings and chunks.

        Args:
            embeddings: numpy array of shape (n, dimension)
            chunks: List of chunk dictionaries
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != chunks count ({len(chunks)})")

        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings)

        # Create index - using IndexFlatIP for exact inner product search
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self.chunks = chunks
        self._id_to_idx = {chunk['id']: idx for idx, chunk in enumerate(chunks)}

        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector (1D numpy array)
            top_k: Number of results to return

        Returns:
            List of result dicts with 'chunk', 'score', 'rank'
        """
        if self.index is None:
            raise ValueError("Index not built yet")

        # Ensure query is 2D and normalized
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            results.append({
                'chunk': self.chunks[idx],
                'score': float(dist),
                'rank': rank + 1
            })

        return results

    def save(self, save_dir: str):
        """
        Save index and chunks to directory.

        Saves:
        - faiss.index: FAISS index file
        - chunks.json: Chunk texts and metadata
        - config.json: Index configuration
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss.index")
        faiss.write_index(self.index, index_path)

        # Save chunks as JSON
        chunks_path = os.path.join(save_dir, "chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            'dimension': self.dimension,
            'num_vectors': self.index.ntotal,
            'embedding_model': EMBEDDING_MODEL
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Index saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str) -> 'FAISSVectorIndex':
        """
        Load index from directory.

        Args:
            load_dir: Directory containing saved index

        Returns:
            Loaded FAISSVectorIndex instance
        """
        # Load config
        config_path = os.path.join(load_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        instance = cls(dimension=config['dimension'])

        # Load FAISS index
        index_path = os.path.join(load_dir, "faiss.index")
        instance.index = faiss.read_index(index_path)

        # Load chunks
        chunks_path = os.path.join(load_dir, "chunks.json")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            instance.chunks = json.load(f)

        instance._id_to_idx = {chunk['id']: idx for idx, chunk in enumerate(instance.chunks)}

        logger.info(f"Loaded index from {load_dir} with {instance.index.ntotal} vectors")
        return instance


def read_pdf_content(file_path: str, timeout_seconds: int = 30) -> str:
    """
    Read text content from a PDF file with timeout.

    Args:
        file_path: Path to PDF file
        timeout_seconds: Max seconds to spend on a single PDF

    Returns:
        Extracted text content
    """
    import signal
    import threading

    text_parts = []
    result = {"text": ""}

    def extract_with_pdfplumber():
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            if text_parts:
                result["text"] = '\n\n'.join(text_parts)
        except Exception as e:
            logger.debug(f"pdfplumber failed for {file_path}: {e}")

    def extract_with_pypdf2():
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            if text_parts:
                result["text"] = '\n\n'.join(text_parts)
        except Exception as e:
            logger.debug(f"PyPDF2 failed for {file_path}: {e}")

    # Try pdfplumber first with timeout
    if HAS_PDFPLUMBER:
        thread = threading.Thread(target=extract_with_pdfplumber)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            logger.warning(f"PDF extraction timeout for {file_path} (pdfplumber)")
        elif result["text"]:
            return result["text"]

    # Fallback to PyPDF2 with timeout
    if HAS_PYPDF2:
        text_parts = []
        result = {"text": ""}
        thread = threading.Thread(target=extract_with_pypdf2)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            logger.warning(f"PDF extraction timeout for {file_path} (PyPDF2)")
        elif result["text"]:
            return result["text"]

    return ""


def load_documents_from_directory(
    directory_path: str,
    file_extensions: List[str] = None,
    recursive: bool = True
) -> List[Dict]:
    """
    Load documents from directory.

    Args:
        directory_path: Path to directory
        file_extensions: List of extensions to include (e.g., ['.txt', '.md', '.pdf'])
                        If None, loads common text formats including PDF
        recursive: Whether to search subdirectories

    Returns:
        List of document dicts with 'content', 'metadata' keys
    """
    if file_extensions is None:
        # Include PDF by default
        file_extensions = ['.txt', '.md', '.json', '.csv', '.html', '.xml', '.pdf']

    file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                       for ext in file_extensions]

    documents = []
    base_path = Path(directory_path)

    if not base_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    pattern = '**/*' if recursive else '*'

    for file_path in base_path.glob(pattern):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in file_extensions:
            continue

        try:
            # Handle PDF files specially
            if file_path.suffix.lower() == '.pdf':
                if not (HAS_PYPDF2 or HAS_PDFPLUMBER):
                    logger.warning(f"Skipping PDF {file_path}: No PDF library installed. "
                                   "Install with: pip install pdfplumber PyPDF2")
                    continue
                content = read_pdf_content(str(file_path))
            else:
                # Regular text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            if content.strip():
                documents.append({
                    'content': content,
                    'metadata': {
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'file_type': file_path.suffix
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            continue

    return documents


def build_unified_index(
    base_doc_path: str,
    unified_index_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 200,
    file_extensions: List[str] = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Build a single unified vector index from all category folders.

    Args:
        base_doc_path: Path to directory containing category folders
        unified_index_path: Path where the unified index will be saved
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        file_extensions: File extensions to include
        api_key: OpenAI API key (optional, uses env var if not provided)

    Returns:
        dict: Summary of processing results
    """
    # Initialize components
    embedder = OpenAIEmbedder(api_key=api_key)
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Get all categories
    categories = get_all_categories(base_doc_path)
    logger.info(f"Found {len(categories)} categories to process")

    results = {
        'total_categories': len(categories),
        'total_documents': 0,
        'total_chunks': 0,
        'categories_processed': [],
        'categories_failed': []
    }

    all_chunks = []

    # Load and chunk documents from all categories
    for category in tqdm(categories, desc="Loading documents from categories"):
        try:
            doc_path = os.path.join(base_doc_path, category)
            documents = load_documents_from_directory(
                doc_path,
                file_extensions=file_extensions,
                recursive=True
            )

            for doc in documents:
                # Add category to metadata
                doc['metadata']['category'] = category

                # Chunk the document
                doc_chunks = chunker.chunk_text(doc['content'], doc['metadata'])
                all_chunks.extend(doc_chunks)

            results['total_documents'] += len(documents)
            results['categories_processed'].append(category)
            logger.info(f"  Loaded {len(documents)} documents from {category}")

        except Exception as e:
            logger.warning(f"Failed to load from {category}: {e}")
            results['categories_failed'].append((category, str(e)))
            continue

    results['total_chunks'] = len(all_chunks)
    logger.info(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks created, cannot build index")
        return results

    # Get texts for embedding
    texts = [chunk['text'] for chunk in all_chunks]

    # Embed all chunks
    logger.info("Embedding chunks with OpenAI API...")
    embeddings, valid_indices = embedder.embed_texts(texts)

    # Filter chunks to only include successful embeddings
    valid_chunks = [all_chunks[i] for i in valid_indices]
    logger.info(f"Successfully embedded {len(valid_chunks)}/{len(all_chunks)} chunks")

    if not valid_chunks:
        logger.error("No chunks were successfully embedded")
        return results

    # Build FAISS index
    logger.info("Building FAISS index...")
    vector_index = FAISSVectorIndex(dimension=EMBEDDING_DIM)
    vector_index.build(embeddings, valid_chunks)

    # Save index
    vector_index.save(unified_index_path)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Unified Vector Index Building Summary (FAISS + OpenAI)")
    logger.info("="*60)
    logger.info(f"Categories processed: {len(results['categories_processed'])}")
    logger.info(f"Categories failed: {len(results['categories_failed'])}")
    logger.info(f"Total documents: {results['total_documents']}")
    logger.info(f"Total chunks: {results['total_chunks']}")
    logger.info(f"Index path: {unified_index_path}")

    if results['categories_failed']:
        logger.info("\nFailed categories:")
        for cat, error in results['categories_failed']:
            logger.info(f"  - {cat}: {error}")

    return results


def get_all_categories(base_path: str) -> List[str]:
    """
    Get all category folders from the base directory.

    Args:
        base_path: Base directory containing category folders

    Returns:
        List of category names (directory names)
    """
    if not os.path.exists(base_path):
        raise ValueError(f"Base path does not exist: {base_path}")

    categories = []
    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        if os.path.isdir(full_path):
            categories.append(item)
    return sorted(categories)


def build_indices_from_directories(
    base_doc_path: str,
    base_index_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 200,
    skip_existing: bool = False,
    file_extensions: List[str] = None,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Build vector indices for all categories in the base document path.

    Args:
        base_doc_path: Path to directory containing category folders
        base_index_path: Path to directory where indices will be saved
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        skip_existing: If True, skip categories that already have indices
        file_extensions: File extensions to include
        api_key: OpenAI API key

    Returns:
        dict: Summary of processing results
    """
    embedder = OpenAIEmbedder(api_key=api_key)
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    categories = get_all_categories(base_doc_path)
    logger.info(f"Found {len(categories)} categories")

    results = {
        'total': len(categories),
        'success': [],
        'skipped': [],
        'failed': []
    }

    for category in tqdm(categories, desc="Processing categories"):
        try:
            doc_path = os.path.join(base_doc_path, category)
            index_path = os.path.join(base_index_path, category)

            # Skip if exists
            if skip_existing and os.path.exists(os.path.join(index_path, "faiss.index")):
                logger.info(f"Skipping {category} - index already exists")
                results['skipped'].append(category)
                continue

            logger.info(f"Processing category: {category}")

            # Load documents
            documents = load_documents_from_directory(
                doc_path,
                file_extensions=file_extensions,
                recursive=True
            )
            logger.info(f"  Loaded {len(documents)} documents")

            if not documents:
                logger.warning(f"  No documents found in {category}")
                results['failed'].append((category, "No documents found"))
                continue

            # Chunk documents
            all_chunks = []
            for doc in documents:
                doc['metadata']['category'] = category
                doc_chunks = chunker.chunk_text(doc['content'], doc['metadata'])
                all_chunks.extend(doc_chunks)

            logger.info(f"  Created {len(all_chunks)} chunks")

            if not all_chunks:
                logger.warning(f"  No chunks created for {category}")
                results['failed'].append((category, "No chunks created"))
                continue

            # Embed
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings, valid_indices = embedder.embed_texts(texts)

            # Filter chunks to only include successful embeddings
            valid_chunks = [all_chunks[i] for i in valid_indices]
            if not valid_chunks:
                logger.warning(f"  No chunks successfully embedded for {category}")
                results['failed'].append((category, "Embedding failed"))
                continue

            # Build index
            vector_index = FAISSVectorIndex(dimension=EMBEDDING_DIM)
            vector_index.build(embeddings, valid_chunks)

            # Save
            vector_index.save(index_path)

            results['success'].append(category)

        except Exception as e:
            logger.error(f"Error processing category {category}: {str(e)}")
            results['failed'].append((category, str(e)))
            continue

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Vector Index Building Summary (FAISS + OpenAI)")
    logger.info("="*60)
    logger.info(f"Total categories: {results['total']}")
    logger.info(f"Successfully processed: {len(results['success'])}")
    logger.info(f"Skipped (existing): {len(results['skipped'])}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results['failed']:
        logger.info("\nFailed categories:")
        for cat, error in results['failed']:
            logger.info(f"  - {cat}: {error}")

    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Build vector indices for FDABench using OpenAI + FAISS"
    )
    parser.add_argument(
        '--doc-path',
        type=str,
        required=True,
        help="Path to directory containing document categories"
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default="./storage",
        help="Path to directory where indices will be saved (default: ./storage)"
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024,
        help="Size of text chunks (default: 1024)"
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help="Skip categories that already have indices"
    )
    parser.add_argument(
        '--unified',
        action='store_true',
        help="Build a single unified index from all categories"
    )
    parser.add_argument(
        '--file-extensions',
        type=str,
        nargs='+',
        default=None,
        help="File extensions to include (e.g., .txt .md). Default: common text formats"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help="OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)"
    )

    args = parser.parse_args()

    if args.unified:
        results = build_unified_index(
            base_doc_path=args.doc_path,
            unified_index_path=args.index_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            file_extensions=args.file_extensions,
            api_key=args.api_key
        )
        exit(0 if results['total_chunks'] > 0 else 1)
    else:
        results = build_indices_from_directories(
            base_doc_path=args.doc_path,
            base_index_path=args.index_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            skip_existing=args.skip_existing,
            file_extensions=args.file_extensions,
            api_key=args.api_key
        )
        exit(0 if not results['failed'] else 1)


if __name__ == "__main__":
    main()
