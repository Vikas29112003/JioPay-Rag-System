#!/usr/bin/env python3
"""
Chunking Ablation Study for JioPay RAG System

This module implements and evaluates different chunking strategies:
1. Fixed: size ∈ {256, 512, 1024} tokens; overlap ∈ {0, 64, 128}
2. Semantic: sentence/paragraph splits with similarity-thresholded merges
3. Structural: split by headings/HTML tags; preserve hierarchy
4. Recursive: fallback from large structural blocks to smaller semantic/fixed chunks
5. LLM-based: intelligent chunking using Google Gemini API with fallback to sentence-based chunking

Each strategy is evaluated on:
- Top-k retrieval accuracy
- Answer F1 score
- Latency (ms)
"""

import json
import time
import re
import logging
import os
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies"""
    strategy_name: str
    size: Optional[int] = None  # In tokens
    overlap: Optional[int] = None  # In tokens
    similarity_threshold: Optional[float] = None
    preserve_hierarchy: bool = False
    max_depth: Optional[int] = None

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    id: str
    text: str
    tokens: int
    source_faq_id: str
    category: str
    chunk_type: str  # 'fixed', 'semantic', 'structural', 'recursive'
    metadata: Dict[str, Any]

@dataclass
class ChunkingResult:
    """Results from chunking evaluation"""
    strategy: str
    config: ChunkingConfig
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    processing_time_ms: float
    chunks: List[Chunk]

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Chunk text according to strategy"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting using word tokenization"""
        return len(word_tokenize(text))

class FixedChunking(ChunkingStrategy):
    """Fixed-size chunking with configurable overlap"""
    
    def chunk_text(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap"""
        tokens = word_tokenize(text)
        chunks = []
        
        if len(tokens) <= config.size:
            # Text is smaller than chunk size, return as single chunk
            chunk = Chunk(
                id=f"{faq_id}_chunk_0",
                text=text,
                tokens=len(tokens),
                source_faq_id=faq_id,
                category=category,
                chunk_type="fixed",
                metadata={"size": config.size, "overlap": config.overlap}
            )
            return [chunk]
        
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            end = min(start + config.size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = ' '.join(chunk_tokens)
            
            chunk = Chunk(
                id=f"{faq_id}_chunk_{chunk_idx}",
                text=chunk_text,
                tokens=len(chunk_tokens),
                source_faq_id=faq_id,
                category=category,
                chunk_type="fixed",
                metadata={
                    "size": config.size,
                    "overlap": config.overlap,
                    "start_token": start,
                    "end_token": end
                }
            )
            chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - config.overlap if config.overlap > 0 else end
            chunk_idx += 1
            
            # Prevent infinite loop
            if start >= len(tokens):
                break
        
        return chunks

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking using sentence boundaries and similarity"""
    
    def __init__(self):
        # Load sentence transformer for similarity computation
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.model = None
    
    def chunk_text(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Split text using semantic boundaries"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            # Single sentence, return as one chunk
            chunk = Chunk(
                id=f"{faq_id}_semantic_0",
                text=text,
                tokens=self.count_tokens(text),
                source_faq_id=faq_id,
                category=category,
                chunk_type="semantic",
                metadata={"method": "single_sentence"}
            )
            return [chunk]
        
        # Group sentences by semantic similarity
        chunks = []
        if self.model and config.similarity_threshold:
            chunks = self._similarity_based_chunking(sentences, faq_id, category, config)
        else:
            # Fallback to paragraph-based chunking
            chunks = self._paragraph_based_chunking(text, faq_id, category, config)
        
        return chunks
    
    def _similarity_based_chunking(self, sentences: List[str], faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Group sentences based on semantic similarity"""
        if not self.model:
            return []
        
        # Encode sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_embedding = embeddings[0:1]
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current sentence and chunk centroid
            chunk_centroid = np.mean(current_embedding, axis=0)
            similarity = cosine_similarity([embeddings[i]], [chunk_centroid])[0][0]
            
            if similarity >= config.similarity_threshold:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                current_embedding = np.vstack([current_embedding, embeddings[i:i+1]])
            else:
                # Create new chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = Chunk(
                    id=f"{faq_id}_semantic_{len(chunks)}",
                    text=chunk_text,
                    tokens=self.count_tokens(chunk_text),
                    source_faq_id=faq_id,
                    category=category,
                    chunk_type="semantic",
                    metadata={
                        "similarity_threshold": config.similarity_threshold,
                        "sentences_count": len(current_chunk_sentences)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_embedding = embeddings[i:i+1]
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = Chunk(
                id=f"{faq_id}_semantic_{len(chunks)}",
                text=chunk_text,
                tokens=self.count_tokens(chunk_text),
                source_faq_id=faq_id,
                category=category,
                chunk_type="semantic",
                metadata={
                    "similarity_threshold": config.similarity_threshold,
                    "sentences_count": len(current_chunk_sentences)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_based_chunking(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Fallback to paragraph-based chunking"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunk = Chunk(
                    id=f"{faq_id}_semantic_{i}",
                    text=paragraph.strip(),
                    tokens=self.count_tokens(paragraph.strip()),
                    source_faq_id=faq_id,
                    category=category,
                    chunk_type="semantic",
                    metadata={"method": "paragraph_based"}
                )
                chunks.append(chunk)
        
        return chunks

class StructuralChunking(ChunkingStrategy):
    """Structural chunking based on hierarchy and formatting"""
    
    def chunk_text(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Split text based on structural elements"""
        chunks = []
        
        # Look for structural markers
        structural_patterns = [
            r'\n#{1,6}\s+(.+)',  # Markdown headers
            r'\n(.+)\n[=-]{3,}',  # Underlined headers
            r'\n\d+\.\s+(.+)',  # Numbered lists
            r'\n[•\-\*]\s+(.+)',  # Bullet points
            r'\n([A-Z][A-Z\s]+):',  # ALL CAPS labels
        ]
        
        # Find all structural boundaries
        boundaries = [0]
        for pattern in structural_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                boundaries.append(match.start())
        
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))
        
        # Create chunks based on structural boundaries
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) > 10:  # Minimum chunk size
                chunk = Chunk(
                    id=f"{faq_id}_structural_{i}",
                    text=chunk_text,
                    tokens=self.count_tokens(chunk_text),
                    source_faq_id=faq_id,
                    category=category,
                    chunk_type="structural",
                    metadata={
                        "preserve_hierarchy": config.preserve_hierarchy,
                        "structural_level": self._detect_structure_level(chunk_text)
                    }
                )
                chunks.append(chunk)
        
        # If no structural elements found, fallback to semantic chunking
        if not chunks:
            semantic_chunker = SemanticChunking()
            semantic_config = ChunkingConfig(
                strategy_name="semantic_fallback",
                similarity_threshold=0.7
            )
            chunks = semantic_chunker.chunk_text(text, faq_id, category, semantic_config)
        
        return chunks
    
    def _detect_structure_level(self, text: str) -> int:
        """Detect the structural level of a chunk"""
        if re.match(r'^#{1,6}\s+', text):
            return len(re.match(r'^(#{1,6})', text).group(1))
        elif re.match(r'^[A-Z][A-Z\s]+:', text):
            return 1  # Top level
        elif re.match(r'^\d+\.\s+', text):
            return 2  # Second level
        elif re.match(r'^[•\-\*]\s+', text):
            return 3  # Third level
        else:
            return 4  # Content level

class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking with fallback strategies"""
    
    def __init__(self):
        self.structural_chunker = StructuralChunking()
        self.semantic_chunker = SemanticChunking()
        self.fixed_chunker = FixedChunking()
    
    def chunk_text(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Apply recursive chunking with fallbacks"""
        chunks = []
        
        # Step 1: Try structural chunking first
        structural_config = ChunkingConfig(
            strategy_name="structural",
            preserve_hierarchy=True
        )
        structural_chunks = self.structural_chunker.chunk_text(text, faq_id, category, structural_config)
        
        # Step 2: Process each structural chunk
        for struct_chunk in structural_chunks:
            if struct_chunk.tokens <= 512:  # Good size, keep as is
                struct_chunk.chunk_type = "recursive"
                struct_chunk.metadata["recursive_level"] = "structural"
                chunks.append(struct_chunk)
            
            elif struct_chunk.tokens <= 1024:  # Try semantic splitting
                semantic_config = ChunkingConfig(
                    strategy_name="semantic",
                    similarity_threshold=0.6
                )
                semantic_chunks = self.semantic_chunker.chunk_text(
                    struct_chunk.text, 
                    struct_chunk.id, 
                    category, 
                    semantic_config
                )
                
                for sem_chunk in semantic_chunks:
                    sem_chunk.chunk_type = "recursive"
                    sem_chunk.metadata["recursive_level"] = "semantic"
                    sem_chunk.source_faq_id = faq_id
                    chunks.append(sem_chunk)
            
            else:  # Too large, use fixed chunking
                fixed_config = ChunkingConfig(
                    strategy_name="fixed",
                    size=512,
                    overlap=64
                )
                fixed_chunks = self.fixed_chunker.chunk_text(
                    struct_chunk.text,
                    struct_chunk.id,
                    category,
                    fixed_config
                )
                
                for fix_chunk in fixed_chunks:
                    fix_chunk.chunk_type = "recursive"
                    fix_chunk.metadata["recursive_level"] = "fixed"
                    fix_chunk.source_faq_id = faq_id
                    chunks.append(fix_chunk)
        
        return chunks

class LLMChunking(ChunkingStrategy):
    """LLM-based intelligent chunking using Google Gemini"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set your Gemini API key.")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise RuntimeError(f"Could not initialize Gemini API: {e}")
    
    def chunk_text(self, text: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """LLM-based intelligent chunking using Gemini API"""
        # Create prompt for intelligent chunking
        prompt = f"""
You are an expert at breaking down text into semantically coherent chunks for a RAG system.

Please analyze the following text and split it into logical chunks that:
1. Maintain semantic coherence
2. Are suitable for retrieval-augmented generation
3. Preserve important context within each chunk
4. Are roughly 200-800 words each

Text to chunk:
{text}

Please return the chunks separated by clear markers. Each chunk should be complete and self-contained.

Format:
CHUNK_1_START
[First chunk content]
CHUNK_1_END

CHUNK_2_START
[Second chunk content]
CHUNK_2_END

And so on...
"""
        
        # Call Gemini API with rate limiting (free tier: 15 requests/minute)
        time.sleep(4.5)  # Wait 4.5 seconds between requests to stay under 15/minute limit
        response = self.model.generate_content(prompt)
        
        if not response.text:
            raise RuntimeError("Empty response from Gemini API")
            
        return self._parse_llm_response(response.text, faq_id, category, config)
    
    def _parse_llm_response(self, response: str, faq_id: str, category: str, config: ChunkingConfig) -> List[Chunk]:
        """Parse LLM response into chunks"""
        chunks = []
        
        # Extract chunks using regex pattern
        chunk_pattern = r'CHUNK_\d+_START\s*(.*?)\s*CHUNK_\d+_END'
        matches = re.findall(chunk_pattern, response, re.DOTALL)
        
        if not matches:
            raise ValueError("LLM response did not follow expected format. No valid chunks found.")
        
        for i, chunk_text in enumerate(matches):
            chunk_text = chunk_text.strip()
            if chunk_text and len(chunk_text) > 10:  # Minimum chunk size
                chunk = Chunk(
                    id=f"{faq_id}_llm_{i}",
                    text=chunk_text,
                    tokens=self.count_tokens(chunk_text),
                    source_faq_id=faq_id,
                    category=category,
                    chunk_type="llm",
                    metadata={
                        "llm_model": "gemini-1.5-flash",
                        "chunk_index": i,
                        "total_chunks": len(matches)
                    }
                )
                chunks.append(chunk)
        
        if not chunks:
            raise ValueError("No valid chunks could be extracted from LLM response")
        
        return chunks


class ChunkingEvaluator:
    """Evaluates different chunking strategies"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.faqs = self._load_dataset()
        self.strategies = {
            'fixed': FixedChunking(),
            'semantic': SemanticChunking(),
            'structural': StructuralChunking(),
            'recursive': RecursiveChunking(),
            'llm': LLMChunking()
        }
    
    def _load_dataset(self) -> List[Dict]:
        """Load the normalized FAQ dataset"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['faqs']
    
    def run_ablation_study(self) -> Dict[str, List[ChunkingResult]]:
        """Run comprehensive chunking ablation study"""
        results = {}
        
        # Fixed chunking configurations
        fixed_configs = []
        for size in [256, 512, 1024]:
            for overlap in [0, 64, 128]:
                fixed_configs.append(ChunkingConfig(
                    strategy_name="fixed",
                    size=size,
                    overlap=overlap
                ))
        
        results['fixed'] = [self._evaluate_strategy('fixed', config) for config in fixed_configs]
        
        # Semantic chunking configurations
        semantic_configs = [
            ChunkingConfig(strategy_name="semantic", similarity_threshold=0.6),
            ChunkingConfig(strategy_name="semantic", similarity_threshold=0.7),
            ChunkingConfig(strategy_name="semantic", similarity_threshold=0.8),
        ]
        results['semantic'] = [self._evaluate_strategy('semantic', config) for config in semantic_configs]
        
        # Structural chunking configuration
        structural_config = ChunkingConfig(
            strategy_name="structural",
            preserve_hierarchy=True
        )
        results['structural'] = [self._evaluate_strategy('structural', structural_config)]
        
        # Recursive chunking configuration
        recursive_config = ChunkingConfig(
            strategy_name="recursive",
            max_depth=3
        )
        results['recursive'] = [self._evaluate_strategy('recursive', recursive_config)]
        
        # LLM chunking configurations
        llm_configs = [
            ChunkingConfig(strategy_name="llm", size=512, overlap=50),
            ChunkingConfig(strategy_name="llm", size=768, overlap=100),
            ChunkingConfig(strategy_name="llm", size=1024, overlap=128),
        ]
        results['llm'] = [self._evaluate_strategy('llm', config) for config in llm_configs]
        
        return results
    
    def _evaluate_strategy(self, strategy_name: str, config: ChunkingConfig) -> ChunkingResult:
        """Evaluate a single chunking strategy"""
        logger.info(f"Evaluating {strategy_name} chunking with config: {config}")
        
        start_time = time.time()
        all_chunks = []
        
        strategy = self.strategies[strategy_name]
        
        for i, faq in enumerate(self.faqs):
            # Combine question and answer for chunking
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            faq_id = f"faq_{i}"
            
            chunks = strategy.chunk_text(text, faq_id, faq['category'], config)
            all_chunks.extend(chunks)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Calculate statistics
        chunk_sizes = [chunk.tokens for chunk in all_chunks]
        
        result = ChunkingResult(
            strategy=strategy_name,
            config=config,
            total_chunks=len(all_chunks),
            avg_chunk_size=np.mean(chunk_sizes) if chunk_sizes else 0,
            min_chunk_size=min(chunk_sizes) if chunk_sizes else 0,
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            processing_time_ms=processing_time_ms,
            chunks=all_chunks
        )
        
        # Save chunked data to Dataset/Chunked data folder
        self._save_chunked_data(result)
        
        return result
    
    def _save_chunked_data(self, result: ChunkingResult):
        """Save chunked data to Dataset/Chunked data folder"""
        # Create directory structure
        chunked_data_dir = Path("../Dataset/Chunked data")
        chunked_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with strategy and config details
        config_str = f"size_{result.config.size}_overlap_{result.config.overlap}"
        if result.config.similarity_threshold:
            config_str += f"_sim_{result.config.similarity_threshold}"
        if result.config.max_depth:
            config_str += f"_depth_{result.config.max_depth}"
            
        filename = f"{result.strategy}_{config_str}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = chunked_data_dir / filename
        
        # Prepare data for saving
        chunked_data = {
            "metadata": {
                "strategy": result.strategy,
                "config": {
                    "strategy_name": result.config.strategy_name,
                    "size": result.config.size,
                    "overlap": result.config.overlap,
                    "similarity_threshold": result.config.similarity_threshold,
                    "preserve_hierarchy": result.config.preserve_hierarchy,
                    "max_depth": result.config.max_depth
                },
                "statistics": {
                    "total_chunks": result.total_chunks,
                    "avg_chunk_size": result.avg_chunk_size,
                    "min_chunk_size": result.min_chunk_size,
                    "max_chunk_size": result.max_chunk_size,
                    "processing_time_ms": result.processing_time_ms
                },
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "chunks": []
        }
        
        # Convert chunks to serializable format
        for chunk in result.chunks:
            chunk_data = {
                "id": chunk.id,
                "text": chunk.text,
                "tokens": chunk.tokens,
                "source_faq_id": chunk.source_faq_id,
                "category": chunk.category,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata
            }
            chunked_data["chunks"].append(chunk_data)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunked_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(result.chunks)} chunks to {filepath}")

    def generate_report(self, results: Dict[str, List[ChunkingResult]]) -> str:
        """Generate detailed ablation study report"""
        report = []
        report.append("=" * 60)
        report.append("CHUNKING ABLATION STUDY REPORT")
        report.append("=" * 60)
        report.append(f"Dataset: {self.dataset_path}")
        report.append(f"Total FAQs: {len(self.faqs)}")
        report.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table header
        report.append("STRATEGY COMPARISON")
        report.append("-" * 60)
        report.append(f"{'Strategy':<15} {'Size':<6} {'Overlap':<7} {'Chunks':<8} {'Avg Size':<10} {'Time(ms)':<10}")
        report.append("-" * 60)
        
        # Add results for each strategy
        for strategy_name, strategy_results in results.items():
            for result in strategy_results:
                size = result.config.size or "N/A"
                overlap = result.config.overlap or "N/A"
                
                report.append(f"{strategy_name:<15} {size:<6} {overlap:<7} "
                            f"{result.total_chunks:<8} {result.avg_chunk_size:<10.1f} "
                            f"{result.processing_time_ms:<10.1f}")
        
        report.append("")
        report.append("DETAILED ANALYSIS")
        report.append("=" * 40)
        
        for strategy_name, strategy_results in results.items():
            report.append(f"\n{strategy_name.upper()} CHUNKING:")
            report.append("-" * 30)
            
            for result in strategy_results:
                config_str = f"Size: {result.config.size}, Overlap: {result.config.overlap}"
                if result.config.similarity_threshold:
                    config_str = f"Similarity: {result.config.similarity_threshold}"
                
                report.append(f"  Config: {config_str}")
                report.append(f"  Total chunks: {result.total_chunks}")
                report.append(f"  Avg chunk size: {result.avg_chunk_size:.1f} tokens")
                report.append(f"  Size range: {result.min_chunk_size} - {result.max_chunk_size} tokens")
                report.append(f"  Processing time: {result.processing_time_ms:.1f} ms")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, List[ChunkingResult]], output_dir: str = "."):
        """Save chunking results and chunks to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save summary report
        report = self.generate_report(results)
        with open(output_path / "chunking_ablation_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results as JSON
        json_results = {}
        for strategy_name, strategy_results in results.items():
            json_results[strategy_name] = []
            for result in strategy_results:
                json_results[strategy_name].append({
                    'config': {
                        'strategy_name': result.config.strategy_name,
                        'size': result.config.size,
                        'overlap': result.config.overlap,
                        'similarity_threshold': result.config.similarity_threshold,
                        'preserve_hierarchy': result.config.preserve_hierarchy,
                        'max_depth': result.config.max_depth
                    },
                    'metrics': {
                        'total_chunks': result.total_chunks,
                        'avg_chunk_size': result.avg_chunk_size,
                        'min_chunk_size': result.min_chunk_size,
                        'max_chunk_size': result.max_chunk_size,
                        'processing_time_ms': result.processing_time_ms
                    },
                    'chunks': [
                        {
                            'id': chunk.id,
                            'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                            'tokens': chunk.tokens,
                            'source_faq_id': chunk.source_faq_id,
                            'category': chunk.category,
                            'chunk_type': chunk.chunk_type,
                            'metadata': chunk.metadata
                        }
                        for chunk in result.chunks[:5]  # Save only first 5 chunks for brevity
                    ]
                })
        
        with open(output_path / "chunking_ablation_results.json", 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main function to run chunking ablation study"""
    dataset_path = "../Dataset/Normalized_data.json"
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    logger.info("Starting Chunking Ablation Study for JioPay RAG System")
    
    evaluator = ChunkingEvaluator(dataset_path)
    results = evaluator.run_ablation_study()
    
    # Print report
    print(evaluator.generate_report(results))
    
    # Save results
    evaluator.save_results(results, ".")
    
    logger.info("Chunking ablation study completed!")

if __name__ == "__main__":
    main()
