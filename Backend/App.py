# Add this near the top of your file, before any imports
import os
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

from flask_cors import CORS
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
from groq import Groq
import json
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

# MongoDB configuration
MONGO_URL = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DATABASE")
NORMALIZED_COLLECTION = os.getenv("MONGODB_COLLECTION")
PROCESSED_COLLECTION = "Data"

# GROQ API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")

# Initialize MongoDB client
client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# Initialize GROQ client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize best embedding model (MiniLM based on your ablation study results)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir)

class StructuralChunker:
    """
    Structural chunking for FAQ data
    """
    
    def chunk_faq_data(self, faqs):
        """
        Apply structural chunking to FAQ data
        """
        chunks = []
        
        for i, faq in enumerate(faqs):
            # Create structural chunk
            chunk = {
                "id": f"faq_{i}_structural",
                "text": f"Q: {faq['question']}\nA: {faq['answer']}",
                "question": faq['question'],
                "answer": faq['answer'],
                "category": faq.get('category', 'Unknown'),
                "source_faq_id": f"faq_{i}",
                "chunk_type": "structural",
                "question_length": faq.get('question_length', len(faq['question'])),
                "answer_length": faq.get('answer_length', len(faq['answer'])),
                "metadata": {
                    "preserve_hierarchy": True,
                    "structural_level": 4,
                    "normalized_timestamp": faq.get('normalized_timestamp', ''),
                    "chunk_created_at": datetime.now().isoformat()
                }
            }
            chunks.append(chunk)
        
        return chunks

def generate_llm_response(question, context_docs, groq_client, temperature=0.3):
    """
    Generate response using GROQ LLM with retrieved context
    """
    try:
        import time
        start_time = time.time()
        
        # Prepare context from retrieved documents
        context = ""
        citations = []
        
        for i, doc in enumerate(context_docs[:3]):  # Use top 3 most relevant
            context += f"Context {i+1}:\n"
            context += f"Category: {doc['category']}\n"
            context += f"Q: {doc['question']}\n"
            context += f"A: {doc['answer']}\n\n"
            
            # Prepare citation data
            citations.append({
                'id': doc['id'],
                'snippet': doc['answer'][:200] + "..." if len(doc['answer']) > 200 else doc['answer'],
                'category': doc['category'],
                'source_question': doc['question'],
                'similarity_score': doc['similarity_score']
            })
        
        # Create prompt for GROQ
        prompt = f"""You are a helpful JioPay customer support assistant. Answer questions based ONLY on the provided context. 
        If the information is not in the context, say "I don't have information about that in my knowledge base."
        Always cite your sources using [Source X] format.
        Be concise and helpful.

Context Information:
{context}

User Question: {question}

Answer:"""

        # Generate response using GROQ
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract token usage if available
        token_usage = {
            'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
            'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
            'total_tokens': getattr(response.usage, 'total_tokens', 0)
        } if hasattr(response, 'usage') and response.usage else {
            'prompt_tokens': len(prompt.split()) * 1.3,  # Rough estimate
            'completion_tokens': len(response.choices[0].message.content.split()) * 1.3,
            'total_tokens': 0
        }
        
        # Calculate estimated cost (approximate for GROQ)
        estimated_cost = (token_usage['prompt_tokens'] * 0.0000015 + 
                         token_usage['completion_tokens'] * 0.000002)  # GROQ pricing estimate
        
        return {
            'response': response.choices[0].message.content.strip(),
            'citations': citations,
            'latency_ms': round(latency * 1000, 2),
            'token_usage': token_usage,
            'estimated_cost_usd': round(estimated_cost, 6),
            'temperature_used': temperature
        }
        
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        return None

@app.route('/setup', methods=['POST'])
def setup_rag_system():
    """
    Setup route: Fetch normalized data, apply structural chunking, 
    create embeddings, and set up search indexes
    """
    try:
        logger.info("Starting RAG system setup...")
        
        # Step 1: Fetch normalized data from MongoDB
        logger.info("Fetching normalized data from MongoDB...")
        normalized_collection = db[NORMALIZED_COLLECTION]
        normalized_docs = list(normalized_collection.find())
        
        if not normalized_docs:
            return jsonify({'error': 'No normalized data found in MongoDB'}), 400
        
        logger.info(f"Found {len(normalized_docs)} normalized documents")
        
        # Step 2: Apply structural chunking
        logger.info("Applying structural chunking...")
        chunker = StructuralChunker()
        chunks = chunker.chunk_faq_data(normalized_docs)
        
        logger.info(f"Created {len(chunks)} structural chunks")
        
        # Step 3: Generate embeddings using best model (MiniLM)
        logger.info("Generating embeddings with MiniLM model...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
        
        # Step 4: Prepare documents for MongoDB with embeddings
        logger.info("Preparing documents for MongoDB storage...")
        processed_docs = []
        for i, chunk in enumerate(chunks):
            doc = {
                **chunk,
                "embedding": embeddings[i].tolist(),
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": embeddings.shape[1],
                "setup_timestamp": datetime.now().isoformat()
            }
            processed_docs.append(doc)
        
        # Step 5: Clear and store processed data
        logger.info("Storing processed data in MongoDB...")
        processed_collection = db[PROCESSED_COLLECTION]
        
        # Clear existing processed data
        processed_collection.delete_many({})
        
        # Insert new processed data
        result = processed_collection.insert_many(processed_docs)
        logger.info(f"Inserted {len(result.inserted_ids)} processed documents")
        
        # Step 6: Create search index
        logger.info("Creating search indexes...")
        try:
            # Create vector search index
            index_name = f"{PROCESSED_COLLECTION}_vector_index"
            
            # Check if index already exists
            existing_indexes = list(processed_collection.list_search_indexes())
            index_exists = any(idx.get("name") == index_name for idx in existing_indexes)
            
            if not index_exists:
                processed_collection.create_search_index(
                    {
                        "definition": {
                            "mappings": {
                                "dynamic": True,
                                "fields": {
                                    "embedding": {
                                        "type": "knnVector",
                                        "dimensions": int(embeddings.shape[1]),
                                        "similarity": "cosine"
                                    }
                                }
                            }
                        },
                        "name": index_name
                    }
                )
                logger.info(f"Created vector search index: {index_name}")
            else:
                logger.info(f"Vector search index already exists: {index_name}")
            
            # Create text search index
            text_index_name = f"{PROCESSED_COLLECTION}_text_index"
            try:
                processed_collection.create_index([
                    ('question', 'text'),
                    ('answer', 'text'),
                    ('category', 'text')
                ], name=text_index_name)
                logger.info(f"Created text search index: {text_index_name}")
            except Exception as e:
                logger.warning(f"Text index may already exist: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Index creation warning: {str(e)}")
        
        # Step 7: Return setup summary
        setup_summary = {
            'status': 'success',
            'message': 'RAG system setup completed successfully',
            'statistics': {
                'normalized_documents': len(normalized_docs),
                'structural_chunks': len(chunks),
                'embeddings_generated': len(embeddings),
                'embedding_dimension': int(embeddings.shape[1]),
                'embedding_model': 'all-MiniLM-L6-v2',
                'database': DB_NAME,
                'processed_collection': PROCESSED_COLLECTION,
                'setup_timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info("RAG system setup completed successfully!")
        return jsonify(setup_summary)
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Setup failed: {str(e)}'
        }), 500

@app.route('/query', methods=['POST'])
def query_rag_system():
    """
    Query route: Handle RAG queries using processed data with temperature support
    """
    try:
        import time
        query_start_time = time.time()
        
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question']
        top_k = data.get('top_k', 5)
        temperature = data.get('temperature', 0.3)  # Get temperature from request
        
        logger.info(f"Processing query: {question} (temperature: {temperature})")
        
        # Step 1: Generate query embedding
        embedding_start = time.time()
        query_embedding = embedding_model.encode([question], normalize_embeddings=True)[0]
        embedding_time = time.time() - embedding_start
        
        # Step 2: Retrieve processed documents
        retrieval_start = time.time()
        processed_collection = db[PROCESSED_COLLECTION]
        all_docs = list(processed_collection.find())
        
        if not all_docs:
            return jsonify({'error': 'No processed data found. Please run /setup first'}), 400
        
        # Step 3: Compute similarities
        doc_embeddings = np.array([doc['embedding'] for doc in all_docs])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        retrieval_time = time.time() - retrieval_start
        
        # Step 4: Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = all_docs[idx]
            result = {
                'id': doc['id'],
                'question': doc['question'],
                'answer': doc['answer'],
                'category': doc['category'],
                'similarity_score': float(similarities[idx]),
                'chunk_type': doc['chunk_type'],
                'snippet': doc['answer'][:150] + "..." if len(doc['answer']) > 150 else doc['answer']
            }
            results.append(result)
        
        # Step 5: Generate response using GROQ LLM
        llm_result = None
        if results and results[0]['similarity_score'] > 0.3:
            logger.info("Generating response using GROQ LLM...")
            llm_result = generate_llm_response(question, results, groq_client, temperature)
        
        query_end_time = time.time()
        total_latency = query_end_time - query_start_time
        
        # Prepare comprehensive response for frontend
        if llm_result:
            response = {
                'status': 'success',
                'query': question,
                'answer': llm_result['response'],
                'llm_generated': True,
                'temperature': temperature,
                'top_k_results': results,
                'citations': llm_result['citations'],
                'performance_metrics': {
                    'total_latency_ms': round(total_latency * 1000, 2),
                    'embedding_latency_ms': round(embedding_time * 1000, 2),
                    'retrieval_latency_ms': round(retrieval_time * 1000, 2),
                    'llm_latency_ms': llm_result['latency_ms']
                },
                'token_usage': llm_result['token_usage'],
                'estimated_cost_usd': llm_result['estimated_cost_usd'],
                'model_info': {
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'llm_model': 'openai/gpt-oss-120b',
                    'temperature_used': temperature,
                    'top_k_retrieved': top_k
                },
                'confidence_score': float(results[0]['similarity_score']),
                'source_category': results[0]['category'],
                'timestamp': datetime.now().isoformat()
            }
        else:
            response = {
                'status': 'no_match',
                'query': question,
                'answer': 'I apologize, but I could not find relevant information in the JioPay knowledge base to answer your question.',
                'llm_generated': False,
                'temperature': temperature,
                'top_k_results': results,
                'citations': [],
                'performance_metrics': {
                    'total_latency_ms': round(total_latency * 1000, 2),
                    'embedding_latency_ms': round(embedding_time * 1000, 2),
                    'retrieval_latency_ms': round(retrieval_time * 1000, 2),
                    'llm_latency_ms': 0
                },
                'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'estimated_cost_usd': 0,
                'model_info': {
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'llm_model': 'openai/gpt-oss-120b',
                    'temperature_used': temperature,
                    'top_k_retrieved': top_k
                },
                'confidence_score': float(results[0]['similarity_score']) if results else 0,
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Query processed successfully. Total latency: {total_latency:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Query failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'JioPay RAG System is running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)