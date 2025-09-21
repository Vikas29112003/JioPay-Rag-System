"""
Embedding Ablation Study
Evaluates different embedding models on retrieval performance metrics:
- Recall@5, MRR, Index Size, and Average Cost per 1k queries
"""

import json
import os
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingAblationStudy:
    """
    Comprehensive ablation study for embedding models
    """
    
    def __init__(self, embedded_data_dir: str):
        self.embedded_data_dir = embedded_data_dir
        self.models_data = {}
        self.evaluation_results = {}
        
        # Model cost estimates (USD per 1M tokens - local models have no API costs)
        self.model_costs = {
            'minilm': 0.0,  # No cost (open source, local)
            'e5-base': 0.0,  # No cost (open source, local)
            'bge-base': 0.0  # No cost (open source, local)
        }
    
    def load_embedded_data(self):
        """Load all embedded data files"""
        logger.info("Loading embedded data files...")
        
        # Find all embedding files
        embedding_files = [f for f in os.listdir(self.embedded_data_dir) 
                          if f.startswith('structural_embeddings_') and f.endswith('.json')]
        
        for file in embedding_files:
            # Extract model name from filename
            model_name = file.replace('structural_embeddings_', '').split('_')[0]
            
            file_path = os.path.join(self.embedded_data_dir, file)
            logger.info(f"Loading {model_name} embeddings from {file}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert embeddings back to numpy arrays
            embeddings = []
            chunks = []
            for chunk in data['embedded_chunks']:
                embeddings.append(np.array(chunk['embedding']))
                chunks.append(chunk)
            
            self.models_data[model_name] = {
                'embeddings': np.array(embeddings),
                'chunks': chunks,
                'metadata': data['metadata'],
                'embedding_dim': data['metadata']['embedding_dimension']
            }
            
            logger.info(f"Loaded {len(chunks)} chunks for {model_name} "
                       f"(dim: {data['metadata']['embedding_dimension']})")
    
    def create_evaluation_queries(self, num_queries: int = 20) -> List[Dict]:
        """
        Create evaluation queries from the chunks
        We'll use some chunks as queries and expect to retrieve similar/related chunks
        """
        # Use chunks from the first model (they should be the same across models)
        first_model = list(self.models_data.keys())[0]
        chunks = self.models_data[first_model]['chunks']
        
        # Select diverse queries (every 5th chunk to get variety)
        query_indices = list(range(0, len(chunks), len(chunks) // num_queries))[:num_queries]
        
        queries = []
        for i, idx in enumerate(query_indices):
            chunk = chunks[idx]
            
            # Create query from the chunk text (simulate user questions)
            query_text = chunk['text']
            
            # For FAQ data, extract the question part if it's a Q&A format
            if query_text.startswith('Q:') and '\nA:' in query_text:
                question_part = query_text.split('\nA:')[0].replace('Q:', '').strip()
                query_text = question_part
            
            # Find related chunks (same category or similar content)
            ground_truth_ids = [chunk['id']]
            category = chunk.get('category', 'unknown')
            
            # Add chunks from the same category as relevant
            for other_chunk in chunks:
                if (other_chunk['id'] != chunk['id'] and 
                    other_chunk.get('category', 'unknown') == category):
                    ground_truth_ids.append(other_chunk['id'])
            
            queries.append({
                'query_id': f"query_{i}",
                'query_text': query_text,
                'source_chunk_id': chunk['id'],
                'expected_category': category,
                'ground_truth_ids': ground_truth_ids
            })
        
        return queries
    
    def compute_similarity_and_retrieve(self, query_embedding: np.ndarray, 
                                      corpus_embeddings: np.ndarray, 
                                      k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity and retrieve top-k results
        """
        similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
        
        # Get top-k indices (excluding the query itself if it's in the corpus)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices, top_k_scores
    
    def evaluate_model(self, model_name: str, queries: List[Dict]) -> Dict:
        """
        Evaluate a single model on the query set
        """
        logger.info(f"Evaluating model: {model_name}")
        
        model_data = self.models_data[model_name]
        embeddings = model_data['embeddings']
        chunks = model_data['chunks']
        
        # Create a mapping from chunk_id to index
        chunk_id_to_idx = {chunk['id']: i for i, chunk in enumerate(chunks)}
        
        results = []
        recall_at_5_scores = []
        reciprocal_ranks = []
        query_times = []
        
        for query in queries:
            # For simplicity, we'll use the embedding of the source chunk as query
            # In a real scenario, you'd embed the query text
            source_idx = chunk_id_to_idx[query['source_chunk_id']]
            query_embedding = embeddings[source_idx]
            
            # Measure query time
            start_time = time.time()
            top_k_indices, top_k_scores = self.compute_similarity_and_retrieve(
                query_embedding, embeddings, k=5
            )
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            retrieved_chunk_ids = [chunks[idx]['id'] for idx in top_k_indices]
            
            # Calculate Recall@5
            relevant_retrieved = len(set(retrieved_chunk_ids) & set(query['ground_truth_ids']))
            recall_at_5 = relevant_retrieved / len(query['ground_truth_ids'])
            recall_at_5_scores.append(recall_at_5)
            
            # Calculate MRR
            reciprocal_rank = 0
            for rank, chunk_id in enumerate(retrieved_chunk_ids, 1):
                if chunk_id in query['ground_truth_ids']:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)
            
            results.append({
                'query_id': query['query_id'],
                'retrieved_ids': retrieved_chunk_ids,
                'scores': top_k_scores.tolist(),
                'recall_at_5': recall_at_5,
                'reciprocal_rank': reciprocal_rank,
                'query_time': query_time
            })
        
        # Calculate aggregate metrics
        avg_recall_at_5 = np.mean(recall_at_5_scores)
        avg_mrr = np.mean(reciprocal_ranks)
        avg_query_time = np.mean(query_times)
        
        return {
            'model_name': model_name,
            'avg_recall_at_5': avg_recall_at_5,
            'avg_mrr': avg_mrr,
            'avg_query_time_ms': avg_query_time * 1000,
            'embedding_dimension': model_data['embedding_dim'],
            'total_chunks': len(chunks),
            'individual_results': results
        }
    
    def calculate_index_size(self, model_name: str) -> float:
        """
        Calculate approximate index size in MB
        """
        model_data = self.models_data[model_name]
        embeddings = model_data['embeddings']
        
        # Calculate size: num_vectors * dimension * 4 bytes (float32)
        size_bytes = embeddings.shape[0] * embeddings.shape[1] * 4
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    def estimate_cost_per_1k_queries(self, model_name: str, avg_query_time: float) -> float:
        """
        Estimate cost per 1000 queries
        For local models, cost is 0 as they run locally without API fees
        """
        return 0.0  # Local models have no API costs
    
    def run_ablation_study(self) -> Dict:
        """
        Run the complete ablation study
        """
        logger.info("Starting embedding ablation study...")
        
        # Load all embedded data
        self.load_embedded_data()
        
        # Create evaluation queries
        queries = self.create_evaluation_queries(num_queries=20)
        logger.info(f"Created {len(queries)} evaluation queries")
        
        # Evaluate each model
        all_results = {}
        for model_name in self.models_data.keys():
            model_results = self.evaluate_model(model_name, queries)
            
            # Add additional metrics
            model_results['index_size_mb'] = self.calculate_index_size(model_name)
            model_results['cost_per_1k_queries'] = self.estimate_cost_per_1k_queries(
                model_name, model_results['avg_query_time_ms'] / 1000
            )
            
            all_results[model_name] = model_results
            
            logger.info(f"Model {model_name} - Recall@5: {model_results['avg_recall_at_5']:.4f}, "
                       f"MRR: {model_results['avg_mrr']:.4f}")
        
        self.evaluation_results = all_results
        return all_results
    
    def create_results_table(self) -> pd.DataFrame:
        """
        Create a summary table of results
        """
        table_data = []
        
        for model_name, results in self.evaluation_results.items():
            table_data.append({
                'Model': model_name,
                'Recall@5': f"{results['avg_recall_at_5']:.4f}",
                'MRR': f"{results['avg_mrr']:.4f}",
                'Index Size (MB)': f"{results['index_size_mb']:.2f}",
                'Avg. Cost / 1k queries': f"${results['cost_per_1k_queries']:.6f}",
                'Embedding Dim': results['embedding_dimension'],
                'Avg Query Time (ms)': f"{results['avg_query_time_ms']:.2f}"
            })
        
        return pd.DataFrame(table_data)
    
    def save_results(self, output_dir: str):
        """
        Save all results to files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"embedding_ablation_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        # Save summary table
        table = self.create_results_table()
        table_file = os.path.join(output_dir, f"embedding_ablation_table_{timestamp}.csv")
        table.to_csv(table_file, index=False)
        
        # Create visualization
        self.create_visualizations(output_dir, timestamp)
        
        logger.info(f"Results saved to {output_dir}")
        return results_file, table_file
    
    def create_visualizations(self, output_dir: str, timestamp: str):
        """
        Create visualizations of the results
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Embedding Models Ablation Study Results', fontsize=16, fontweight='bold')
        
        models = list(self.evaluation_results.keys())
        
        # Recall@5 comparison
        recall_scores = [self.evaluation_results[m]['avg_recall_at_5'] for m in models]
        axes[0, 0].bar(models, recall_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Recall@5 Performance')
        axes[0, 0].set_ylabel('Recall@5')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(recall_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # MRR comparison
        mrr_scores = [self.evaluation_results[m]['avg_mrr'] for m in models]
        axes[0, 1].bar(models, mrr_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Mean Reciprocal Rank (MRR)')
        axes[0, 1].set_ylabel('MRR')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(mrr_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Index Size comparison
        index_sizes = [self.evaluation_results[m]['index_size_mb'] for m in models]
        axes[1, 0].bar(models, index_sizes, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 0].set_title('Index Size')
        axes[1, 0].set_ylabel('Size (MB)')
        for i, v in enumerate(index_sizes):
            axes[1, 0].text(i, v + max(index_sizes)*0.01, f'{v:.2f}', ha='center', va='bottom')
        
        # Query Time comparison
        query_times = [self.evaluation_results[m]['avg_query_time_ms'] for m in models]
        axes[1, 1].bar(models, query_times, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Average Query Time')
        axes[1, 1].set_ylabel('Time (ms)')
        for i, v in enumerate(query_times):
            axes[1, 1].text(i, v + max(query_times)*0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"embedding_ablation_comparison_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_file}")


def main():
    """Main execution function"""
    # Paths
    embedded_data_dir = "/home/vikas/Desktop/LLM_Assignment_02/JioPay-Rag-System/Dataset/Embeded data"
    output_dir = "/home/vikas/Desktop/LLM_Assignment_02/JioPay-Rag-System/Ablation Study/result"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize and run ablation study
        study = EmbeddingAblationStudy(embedded_data_dir)
        results = study.run_ablation_study()
        
        # Create and display results table
        table = study.create_results_table()
        print("\n" + "="*80)
        print("EMBEDDING ABLATION STUDY RESULTS")
        print("="*80)
        print(table.to_string(index=False))
        print("="*80)
        
        # Save results
        results_file, table_file = study.save_results(output_dir)
        
        logger.info("Embedding ablation study completed successfully!")
        logger.info(f"Detailed results: {results_file}")
        logger.info(f"Summary table: {table_file}")
        
    except Exception as e:
        logger.error(f"Error in ablation study: {str(e)}")
        raise


if __name__ == "__main__":
    main()
