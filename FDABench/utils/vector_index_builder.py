"""
Vector Index Builder for FDABench

This utility builds vector indices from document directories using llama_index.
The generated indices are compatible with FDABench's VectorSearchTool and
can be used for semantic search in data agent workflows.

Compatible with:
- FDABench/tools/search_tools.py (VectorSearchTool)
- search_llm.py (vectorDB_search function)
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

import os
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """
    Set up the environment and embedding model.
    Reads OPENAI_API_KEY from environment variables.

    Returns:
        OpenAIEmbedding: Configured embedding model
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY must be set in environment variables or .env file"
        )

    embed_model = OpenAIEmbedding(
        embed_batch_size=10,
        model="text-embedding-3-small"
    )
    Settings.embed_model = embed_model
    return embed_model


def load_documents(directory_path, file_extensions=None):
    """
    Load documents from the specified directory.

    Args:
        directory_path: Path to directory containing documents
        file_extensions: List of file extensions to load (e.g., ['.txt', '.md'])
                        If None, loads all supported formats

    Returns:
        List of Document objects
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")

    reader_kwargs = {}
    if file_extensions:
        reader_kwargs['required_exts'] = file_extensions

    logger.info(f"Loading documents from {directory_path}")
    return SimpleDirectoryReader(directory_path, **reader_kwargs).load_data()


def create_nodes(documents, chunk_size=1024, chunk_overlap=20):
    """
    Split documents into nodes with the specified chunk size.

    Args:
        documents: List of Document objects
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Node objects
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.get_nodes_from_documents(documents)


def build_index(nodes, embed_model=None):
    """
    Build a vector store index from nodes.

    Args:
        nodes: List of Node objects
        embed_model: Embedding model to use. If None, uses global Settings.

    Returns:
        VectorStoreIndex: Built vector index
    """
    if embed_model:
        return VectorStoreIndex(nodes, embed_model=embed_model)
    else:
        return VectorStoreIndex(nodes)


def persist_index(index, persist_dir):
    """
    Persist the index to the specified directory.

    The persisted format includes:
    - docstore.json: Document storage
    - index_store.json: Index metadata
    - default__vector_store.json: Vector embeddings
    - graph_store.json: Graph relationships

    Args:
        index: VectorStoreIndex to persist
        persist_dir: Directory to save the index
    """
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Index persisted to {persist_dir}")


def get_all_categories(base_path):
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
    base_doc_path,
    base_index_path,
    chunk_size=1024,
    skip_existing=False
):
    """
    Build vector indices for all categories in the base document path.

    Args:
        base_doc_path: Path to directory containing category folders
        base_index_path: Path to directory where indices will be saved
        chunk_size: Size of text chunks
        skip_existing: If True, skip categories that already have indices

    Returns:
        dict: Summary of processing results
    """
    # Setup
    embed_model = setup_environment()

    # Get all categories
    categories = get_all_categories(base_doc_path)
    logger.info(f"Found {len(categories)} categories: {categories[:5]}...")

    results = {
        'total': len(categories),
        'success': [],
        'skipped': [],
        'failed': []
    }

    # Process each category
    for category in tqdm(categories, desc="Processing categories"):
        try:
            doc_path = os.path.join(base_doc_path, category)
            index_path = os.path.join(base_index_path, category)

            # Skip if index already exists
            if skip_existing and os.path.exists(index_path):
                logger.info(f"Skipping {category} - index already exists")
                results['skipped'].append(category)
                continue

            logger.info(f"Processing category: {category}")

            # Load documents
            documents = load_documents(doc_path)
            logger.info(f"  Loaded {len(documents)} documents")

            # Process documents
            nodes = create_nodes(documents, chunk_size=chunk_size)
            logger.info(f"  Created {len(nodes)} nodes")

            # Build index
            index = build_index(nodes, embed_model=embed_model)
            logger.info(f"  Built index")

            # Persist index
            persist_index(index, index_path)
            logger.info(f"  Saved to {index_path}")

            results['success'].append(category)

        except Exception as e:
            logger.error(f"Error processing category {category}: {str(e)}")
            results['failed'].append((category, str(e)))
            continue

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Vector Index Building Summary")
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
        description="Build vector indices for FDABench from document directories"
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
        help="Path to directory where indices will be saved (default: ./storage from project root)"
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024,
        help="Size of text chunks (default: 1024)"
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help="Skip categories that already have indices"
    )

    args = parser.parse_args()

    # Build indices
    results = build_indices_from_directories(
        base_doc_path=args.doc_path,
        base_index_path=args.index_path,
        chunk_size=args.chunk_size,
        skip_existing=args.skip_existing
    )

    # Exit with error code if any failed
    if results['failed']:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
