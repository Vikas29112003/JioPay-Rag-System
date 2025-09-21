"""
Embedding Script for Structural Chunked Data
This script embeds structural chunked data using E5 and BGE/MiniLM models.
"""

import json
import os
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuralDataEmbedder:
    """
    Class to embed structural chunked data using various embedding models.
    Supports E5 and BGE/MiniLM models.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Define supported models (using lighter/faster models)
        self.models = {
            'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
            'e5-small': 'intfloat/e5-small-v2',
            'bge-small': 'BAAI/bge-small-en-v1.5',
            'e5-base': 'intfloat/e5-base-v2',
            'bge-base': 'BAAI/bge-base-en-v1.5'
        }
        
        self.loaded_models = {}
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """Load a specific embedding model."""
        if model_name not in self.loaded_models:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not supported. Available: {list(self.models.keys())}")
            
            logger.info(f"Loading model: {self.models[model_name]}")
            model = SentenceTransformer(self.models[model_name], device=self.device)
            self.loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
        
        return self.loaded_models[model_name]
    
    def load_structural_data(self, file_path: str) -> Dict[str, Any]:
        """Load structural chunked data from JSON file."""
        logger.info(f"Loading structural data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data['chunks'])} chunks")
        return data
    
    def preprocess_text_for_embedding(self, text: str, model_name: str) -> str:
        """Preprocess text based on model requirements."""
        # E5 models require specific prefixes
        if model_name.startswith('e5'):
            # For passage/document embedding
            return f"passage: {text}"
        
        # BGE models can use query prefix for retrieval
        elif model_name.startswith('bge'):
            return text  # BGE doesn't require special prefixes for documents
        
        # MiniLM and others
        else:
            return text
    
    def embed_chunks(self, chunks: List[Dict], model_name: str, batch_size: int = 32) -> np.ndarray:
        """Embed a list of chunks using the specified model."""
        model = self.load_model(model_name)
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            processed_text = self.preprocess_text_for_embedding(chunk['text'], model_name)
            texts.append(processed_text)
        
        logger.info(f"Embedding {len(texts)} chunks with {model_name}")
        
        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with {model_name}"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for better similarity computation
            )
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], 
                       model_name: str, metadata: Dict, output_dir: str):
        """Save embeddings along with chunk metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"structural_embeddings_{model_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for saving
        embedded_data = {
            "metadata": {
                **metadata,
                "embedding_model": self.models[model_name],
                "embedding_model_name": model_name,
                "embedding_dimension": embeddings.shape[1],
                "total_embedded_chunks": len(chunks),
                "embedding_created_at": timestamp,
                "device_used": str(self.device)
            },
            "embedded_chunks": []
        }
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            embedded_chunk = {
                **chunk,
                "embedding": embeddings[i].tolist(),
                "embedding_model": model_name
            }
            embedded_data["embedded_chunks"].append(embedded_chunk)
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(embedded_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Embeddings saved to: {filepath}")
        return filepath
    
    def process_structural_data(self, input_file: str, output_dir: str, 
                              models_to_use: List[str] = None, batch_size: int = 32):
        """
        Main method to process structural chunked data and generate embeddings.
        
        Args:
            input_file: Path to structural chunked data JSON file
            output_dir: Directory to save embedded data
            models_to_use: List of model names to use for embedding
            batch_size: Batch size for embedding generation
        """
        if models_to_use is None:
            models_to_use = ['e5-base', 'bge-base', 'minilm']
        
        # Load structural data
        data = self.load_structural_data(input_file)
        chunks = data['chunks']
        metadata = data['metadata']
        
        logger.info(f"Processing {len(chunks)} structural chunks")
        logger.info(f"Original chunking strategy: {metadata['strategy']}")
        
        # Generate embeddings for each model
        results = {}
        for model_name in models_to_use:
            try:
                logger.info(f"Processing with model: {model_name}")
                
                # Generate embeddings
                embeddings = self.embed_chunks(chunks, model_name, batch_size)
                
                # Save embeddings
                output_path = self.save_embeddings(
                    embeddings, chunks, model_name, metadata, output_dir
                )
                
                results[model_name] = {
                    'output_path': output_path,
                    'embedding_shape': embeddings.shape,
                    'model_full_name': self.models[model_name]
                }
                
                logger.info(f"Successfully processed {model_name}")
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def generate_embedding_summary(self, results: Dict, output_dir: str):
        """Generate a summary of the embedding process."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(output_dir, f"embedding_summary_{timestamp}.json")
        
        summary = {
            "embedding_summary": {
                "created_at": timestamp,
                "models_processed": len(results),
                "successful_models": len([r for r in results.values() if 'error' not in r]),
                "failed_models": len([r for r in results.values() if 'error' in r]),
                "results": results
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Embedding summary saved to: {summary_file}")
        return summary_file


def main():
    """Main execution function."""
    # Paths
    input_file = "/home/vikas/Desktop/LLM_Assignment_02/JioPay-Rag-System/Dataset/Chunked data/structural_size_None_overlap_None_20250921_185815.json"
    output_dir = "/home/vikas/Desktop/LLM_Assignment_02/JioPay-Rag-System/Dataset/Embeded data"
    
    # Models to use
    models_to_use = ['minilm', 'e5-base', 'bge-base']
    
    # Initialize embedder
    embedder = StructuralDataEmbedder()
    
    try:
        # Process structural data
        logger.info("Starting structural data embedding process...")
        results = embedder.process_structural_data(
            input_file=input_file,
            output_dir=output_dir,
            models_to_use=models_to_use,
            batch_size=16  # Smaller batch size for stability
        )
        
        # Generate summary
        summary_file = embedder.generate_embedding_summary(results, output_dir)
        
        logger.info("Embedding process completed successfully!")
        logger.info(f"Results summary saved to: {summary_file}")
        
        # Print results
        print("\n" + "="*50)
        print("EMBEDDING RESULTS SUMMARY")
        print("="*50)
        for model_name, result in results.items():
            if 'error' in result:
                print(f"❌ {model_name}: FAILED - {result['error']}")
            else:
                print(f"✅ {model_name}: SUCCESS")
                print(f"   - Output: {result['output_path']}")
                print(f"   - Shape: {result['embedding_shape']}")
                print(f"   - Model: {result['model_full_name']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
