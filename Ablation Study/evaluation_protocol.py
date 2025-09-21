#!/usr/bin/env python3
"""
Comprehensive Evaluation Protocol for JioPay RAG System
Evaluates retrieval performance, answer quality, and system performance
according to the specified evaluation criteria.
"""

import requests
import json
import time
import statistics
import psutil
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "https://vikas2900-jio-rag.hf.space"
OUTPUT_DIR = "/home/vikas/Desktop/LLM_Assignment_02/JioPay-Rag-System/Ablation Study/evaluation_results"

# Initialize GROQ client for LLM-as-judge evaluation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    judge_client = Groq(api_key=GROQ_API_KEY)
else:
    print("Warning: GROQ_API_KEY not found. LLM-as-judge evaluation will be disabled.")
    judge_client = None

@dataclass
class EvaluationResult:
    """Data class to store evaluation results"""
    query: str
    category: str
    response_time: float
    answer: str
    sources: List[str]
    confidence: float
    retrieval_rank: int  # Rank of the correct answer in retrieved sources
    exact_match: bool
    faithfulness_score: float
    human_rating: int  # 1-5 scale
    performance_metrics: Dict[str, float]  # API performance metrics
    # LLM-as-judge metrics
    llm_judge_relevance: float  # 0-1 score
    llm_judge_accuracy: float   # 0-1 score
    llm_judge_completeness: float  # 0-1 score
    llm_judge_faithfulness: float  # 0-1 score
    llm_judge_overall: float    # 0-1 score
    judge_reasoning: str        # LLM's reasoning

class JioPayEvaluator:
    """Comprehensive evaluator for JioPay RAG system"""
    
    def __init__(self):
        self.test_queries = self._load_test_queries()
        self.ground_truth = self._load_ground_truth()
        self.results = []
        
    def _load_test_queries(self) -> List[Dict[str, str]]:
        """Load test queries spanning different categories"""
        return [
            # ONBOARDING (3 queries)
            {
                "query": "How do I register for JioPay account?",
                "category": "onboarding",
                "expected_keywords": ["register", "account", "signup", "mobile", "verification"]
            },
            {
                "query": "What documents are required for JioPay KYC verification?",
                "category": "onboarding", 
                "expected_keywords": ["KYC", "documents", "verification", "Aadhaar", "PAN"]
            },
            {
                "query": "How to download and install JioPay app?",
                "category": "onboarding",
                "expected_keywords": ["download", "install", "app", "Play Store", "iOS"]
            },
            
            # PAYMENTS (3 queries)
            {
                "query": "How do I make a UPI payment using JioPay?",
                "category": "payments",
                "expected_keywords": ["UPI", "payment", "transfer", "money", "send"]
            },
            {
                "query": "What are the daily transaction limits on JioPay?",
                "category": "payments",
                "expected_keywords": ["limit", "daily", "transaction", "maximum", "amount"]
            },
            {
                "query": "How to pay bills using JioPay wallet?",
                "category": "payments",
                "expected_keywords": ["bills", "payment", "wallet", "electricity", "mobile"]
            },
            
            # SECURITY (2 queries)
            {
                "query": "How secure are JioPay transactions?",
                "category": "security",
                "expected_keywords": ["secure", "encryption", "safety", "protection", "fraud"]
            },
            {
                "query": "What should I do if my JioPay account is compromised?",
                "category": "security",
                "expected_keywords": ["compromised", "block", "report", "security", "fraud"]
            },
            
            # REFUNDS (2 queries)
            {
                "query": "How do I request a refund for failed transaction?",
                "category": "refunds",
                "expected_keywords": ["refund", "failed", "transaction", "request", "money"]
            },
            {
                "query": "How long does it take to process refunds in JioPay?",
                "category": "refunds",
                "expected_keywords": ["refund", "process", "time", "duration", "days"]
            },
            
            # PRICING (1 query)
            {
                "query": "What are the charges for JioPay transactions?",
                "category": "pricing",
                "expected_keywords": ["charges", "fees", "cost", "transaction", "free"]
            },
            
            # API/INTEGRATION (1 query)
            {
                "query": "How can merchants integrate JioPay payment gateway?",
                "category": "api_integration",
                "expected_keywords": ["merchant", "integrate", "API", "gateway", "developer"]
            }
        ]
    
    def _load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth answers - simplified without MongoDB dependency"""
        # For evaluation, we'll use the LLM judge to assess quality
        # rather than exact ground truth matching
        return {
            "How do I register for JioPay account?": "Download app, enter mobile, verify OTP, complete KYC",
            "What documents are required for JioPay KYC verification?": "Aadhaar card, PAN card, bank details required",
            "How to download and install JioPay app?": "Available on Google Play Store and Apple App Store",
            "How do I make a UPI payment using JioPay?": "Open app, select UPI, enter amount, scan QR or enter VPA",
            "What are the daily transaction limits on JioPay?": "Varies by KYC level and bank policies",
            "How to pay bills using JioPay wallet?": "Select bill payment, choose utility, enter details, pay from wallet",
            "How secure are JioPay transactions?": "Uses encryption, secure authentication, fraud protection",
            "What should I do if my JioPay account is compromised?": "Immediately block account, contact support, change passwords",
            "How do I request a refund for failed transaction?": "Contact support with transaction details, provide screenshots",
            "How long does it take to process refunds in JioPay?": "Usually 3-5 business days depending on bank",
            "What are the charges for JioPay transactions?": "Most transactions are free, some premium services may have charges",
            "How can merchants integrate JioPay payment gateway?": "Contact business team, API documentation available"
        }
        return {
            "How do I register for JioPay account?": "Download JioPay app, enter mobile number, verify OTP, complete KYC",
            "What documents are required for JioPay KYC verification?": "Aadhaar card, PAN card, and bank account details",
            "How to download and install JioPay app?": "Download from Google Play Store or Apple App Store",
            # Add more ground truth as needed
        }
    
    def _measure_system_resources(self) -> Dict[str, float]:
        """Measure current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024)
        }
    
    def _query_api(self, query: str) -> Tuple[Dict[str, Any], float]:
        """Query the API and measure response time"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={"question": query},
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                return response.json(), response_time
            else:
                return {"error": f"Status {response.status_code}"}, response_time
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {"error": str(e)}, response_time
    
    def _calculate_precision_at_1(self, sources: List[str], expected_keywords: List[str]) -> float:
        """Calculate Precision@1 - whether the first source contains expected content"""
        if not sources:
            return 0.0
        
        first_source = sources[0].lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in first_source)
        return matches / len(expected_keywords)
    
    def _calculate_recall_at_k(self, sources: List[str], expected_keywords: List[str], k: int = 3) -> float:
        """Calculate Recall@k - how many expected keywords appear in top-k sources"""
        if not sources:
            return 0.0
        
        top_k_sources = " ".join(sources[:k]).lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in top_k_sources)
        return matches / len(expected_keywords)
    
    def _calculate_mrr(self, sources: List[str], expected_keywords: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, source in enumerate(sources):
            source_lower = source.lower()
            if any(keyword.lower() in source_lower for keyword in expected_keywords):
                return 1.0 / (i + 1)
        return 0.0
    
    def _check_exact_match(self, answer: str, query: str, expected_keywords: List[str]) -> bool:
        """Enhanced exact match using LLM judge for semantic similarity"""
        if not judge_client:
            # Fallback to keyword matching
            answer_words = set(answer.lower().split())
            keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
            return keyword_matches >= len(expected_keywords) * 0.6
        
        try:
            keywords_str = ", ".join(expected_keywords)
            
            match_prompt = f"""Evaluate if the given answer adequately addresses the query and contains the expected information.

QUERY: {query}
ANSWER: {answer}
EXPECTED TOPICS/KEYWORDS: {keywords_str}

Does the answer adequately cover the expected topics and provide a satisfactory response to the query?
Respond with only "YES" or "NO" followed by a brief reason.

Example: "YES - Answer covers payment methods and transaction limits as expected"
Example: "NO - Answer lacks information about KYC requirements"
"""

            response = judge_client.chat.completions.create(
                model="openai/gpt-oss-120b",  # Use OSS model
                messages=[{"role": "user", "content": match_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            judge_response = response.choices[0].message.content.strip().upper()
            return judge_response.startswith("YES")
            
        except Exception as e:
            print(f"LLM exact match evaluation failed: {e}")
            # Fallback to keyword matching
            answer_words = set(answer.lower().split())
            keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
            return keyword_matches >= len(expected_keywords) * 0.6
    
    def _evaluate_faithfulness(self, answer: str, sources: List[str]) -> float:
        """Simple faithfulness evaluation - how well answer is supported by sources"""
        if not sources:
            return 0.0
        
        answer_words = set(answer.lower().split())
        sources_text = " ".join(sources).lower()
        sources_words = set(sources_text.split())
        
        # Calculate overlap between answer and sources
        overlap = len(answer_words.intersection(sources_words))
        return min(overlap / len(answer_words), 1.0) if answer_words else 0.0
    
    def _human_rating_simulation(self, answer: str, query: str) -> int:
        """Simulate human rating (1-5 scale) based on answer quality heuristics"""
        # This is a simplified simulation - in practice, this would be done by humans
        if not answer or len(answer) < 20:
            return 1
        elif len(answer) < 50:
            return 2
        elif len(answer) < 100:
            return 3
        elif len(answer) < 200:
            return 4
        else:
            return 5
    
    def _llm_as_judge_evaluation(self, query: str, answer: str, sources: List[str], expected_keywords: List[str]) -> Dict[str, Any]:
        """Use GROQ LLM as judge to evaluate answer quality"""
        if not judge_client:
            # Fallback to simple heuristics if GROQ not available
            return {
                'relevance': 0.5,
                'accuracy': 0.5,
                'completeness': 0.5,
                'faithfulness': 0.5,
                'overall': 0.5,
                'reasoning': 'LLM judge not available - using fallback scoring'
            }
        
        try:
            # Prepare sources context
            sources_text = "\n".join([f"Source {i+1}: {source}" for i, source in enumerate(sources[:3])])
            expected_keywords_str = ", ".join(expected_keywords)
            
            judge_prompt = f"""You are an expert evaluator of question-answering systems. Evaluate the following answer based on multiple criteria.

QUERY: {query}

ANSWER TO EVALUATE: {answer}

RETRIEVED SOURCES:
{sources_text}

EXPECTED KEYWORDS/TOPICS: {expected_keywords_str}

Please evaluate the answer on the following criteria (score 0.0 to 1.0):

1. RELEVANCE: How well does the answer address the specific question asked?
2. ACCURACY: Is the information in the answer factually correct based on the sources?
3. COMPLETENESS: Does the answer provide sufficient detail to be helpful?
4. FAITHFULNESS: Is the answer well-supported by the provided sources?
5. OVERALL: Overall quality of the answer considering all factors.

Respond in the following JSON format:
{{
    "relevance": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "faithfulness": 0.0-1.0,
    "overall": 0.0-1.0,
    "reasoning": "Brief explanation of your scoring rationale"
}}

Focus on being objective and consistent in your evaluation."""

            response = judge_client.chat.completions.create(
                model="openai/gpt-oss-120b",  # Use OSS model
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the JSON response
            judge_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                if judge_response.startswith("```json"):
                    judge_response = judge_response.replace("```json", "").replace("```", "").strip()
                elif judge_response.startswith("```"):
                    judge_response = judge_response.replace("```", "").strip()
                
                scores = json.loads(judge_response)
                
                # Validate scores are in range
                for key in ['relevance', 'accuracy', 'completeness', 'faithfulness', 'overall']:
                    if key in scores:
                        scores[key] = max(0.0, min(1.0, float(scores[key])))
                    else:
                        scores[key] = 0.5  # Default if missing
                
                return scores
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract numbers from text
                print(f"JSON parsing failed for judge response: {judge_response}")
                scores = {
                    'relevance': 0.5,
                    'accuracy': 0.5,
                    'completeness': 0.5,
                    'faithfulness': 0.5,
                    'overall': 0.5,
                    'reasoning': 'Failed to parse judge response'
                }
                
                # Try to extract numeric scores from text
                import re
                numbers = re.findall(r'[\d.]+', judge_response)
                if len(numbers) >= 5:
                    try:
                        scores['relevance'] = max(0.0, min(1.0, float(numbers[0])))
                        scores['accuracy'] = max(0.0, min(1.0, float(numbers[1])))
                        scores['completeness'] = max(0.0, min(1.0, float(numbers[2])))
                        scores['faithfulness'] = max(0.0, min(1.0, float(numbers[3])))
                        scores['overall'] = max(0.0, min(1.0, float(numbers[4])))
                    except (ValueError, IndexError):
                        pass
                
                return scores
                
        except Exception as e:
            print(f"LLM judge evaluation failed: {e}")
            return {
                'relevance': 0.5,
                'accuracy': 0.5,
                'completeness': 0.5,
                'faithfulness': 0.5,
                'overall': 0.5,
                'reasoning': f'Evaluation failed: {str(e)}'
            }
    
    def evaluate_single_query(self, test_item: Dict[str, str]) -> EvaluationResult:
        """Evaluate a single query"""
        query = test_item["query"]
        category = test_item["category"]
        expected_keywords = test_item["expected_keywords"]
        
        print(f"\n🔍 Evaluating: {query}")
        
        # Measure system resources before query
        resources_before = self._measure_system_resources()
        
        # Query the API
        response_data, response_time = self._query_api(query)
        
        # Measure system resources after query
        resources_after = self._measure_system_resources()
        
        # Extract response components
        answer = response_data.get("answer", "")
        # API returns citations and top_k_results, not sources
        citations = response_data.get("citations", [])
        top_k_results = response_data.get("top_k_results", [])
        sources = [item.get("snippet", "") for item in citations] if citations else [item.get("snippet", "") for item in top_k_results]
        confidence = response_data.get("confidence_score", 0.0)
        
        # Calculate metrics
        p_at_1 = self._calculate_precision_at_1(sources, expected_keywords)
        recall_at_3 = self._calculate_recall_at_k(sources, expected_keywords, k=3)
        mrr = self._calculate_mrr(sources, expected_keywords)
        
        # Answer quality metrics using LLM judge
        exact_match = self._check_exact_match(answer, query, expected_keywords)
        
        # Get LLM judge evaluation
        judge_scores = self._llm_as_judge_evaluation(query, answer, sources, expected_keywords)
        
        # Legacy faithfulness calculation (keep for comparison)
        legacy_faithfulness = self._evaluate_faithfulness(answer, sources)
        human_rating = self._human_rating_simulation(answer, query)
        
        # Find rank of relevant answer
        retrieval_rank = 0
        for i, source in enumerate(sources):
            if any(keyword.lower() in source.lower() for keyword in expected_keywords):
                retrieval_rank = i + 1
                break
        
        # Extract performance metrics from API response
        api_performance = response_data.get("performance_metrics", {})
        
        result = EvaluationResult(
            query=query,
            category=category,
            response_time=response_time,
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_rank=retrieval_rank,
            exact_match=exact_match,
            faithfulness_score=legacy_faithfulness,
            human_rating=human_rating,
            performance_metrics=api_performance,
            # LLM judge scores
            llm_judge_relevance=judge_scores['relevance'],
            llm_judge_accuracy=judge_scores['accuracy'],
            llm_judge_completeness=judge_scores['completeness'],
            llm_judge_faithfulness=judge_scores['faithfulness'],
            llm_judge_overall=judge_scores['overall'],
            judge_reasoning=judge_scores['reasoning']
        )
        
        # Print immediate results
        print(f"  ⏱️  Response Time: {response_time:.2f}s")
        print(f"  🎯 P@1: {p_at_1:.2f}, Recall@3: {recall_at_3:.2f}, MRR: {mrr:.2f}")
        print(f"  🤖 LLM Judge - Overall: {judge_scores['overall']:.2f}, Faithfulness: {judge_scores['faithfulness']:.2f}")
        print(f"  ⭐ Human Rating: {human_rating}/5")
        print(f"  🔢 Tokens: {response_data.get('token_usage', {}).get('total_tokens', 'N/A')}")
        print(f"  📊 Sources Found: {len(sources)}")
        
        return result
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation protocol"""
        print("🚀 Starting Comprehensive JioPay RAG Evaluation")
        print("=" * 60)
        
        # Check API availability
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=30)
            if health_response.status_code != 200:
                raise Exception("API health check failed")
        except Exception as e:
            print(f"❌ API not accessible: {e}")
            return {}
        
        print("✅ API is accessible")
        
        # Run evaluation for each query
        start_time = time.time()
        
        for test_item in self.test_queries:
            try:
                result = self.evaluate_single_query(test_item)
                self.results.append(result)
                time.sleep(1)  # Small delay between requests
            except Exception as e:
                print(f"❌ Error evaluating query '{test_item['query']}': {e}")
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics()
        
        # Generate report
        report = self._generate_report(metrics, total_time)
        
        # Save results
        self._save_results(report)
        
        return report
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all queries"""
        if not self.results:
            return {}
        
        # Performance metrics
        response_times = [r.response_time for r in self.results]
        
        # Retrieval metrics
        p_at_1_scores = []
        recall_at_3_scores = []
        mrr_scores = []
        
        for result in self.results:
            test_item = next(t for t in self.test_queries if t["query"] == result.query)
            expected_keywords = test_item["expected_keywords"]
            
            p_at_1_scores.append(self._calculate_precision_at_1(result.sources, expected_keywords))
            recall_at_3_scores.append(self._calculate_recall_at_k(result.sources, expected_keywords, k=3))
            mrr_scores.append(self._calculate_mrr(result.sources, expected_keywords))
        
        # Answer quality metrics
        exact_matches = [r.exact_match for r in self.results]
        faithfulness_scores = [r.faithfulness_score for r in self.results]
        human_ratings = [r.human_rating for r in self.results]
        
        # LLM judge metrics
        llm_relevance_scores = [r.llm_judge_relevance for r in self.results]
        llm_accuracy_scores = [r.llm_judge_accuracy for r in self.results]
        llm_completeness_scores = [r.llm_judge_completeness for r in self.results]
        llm_faithfulness_scores = [r.llm_judge_faithfulness for r in self.results]
        llm_overall_scores = [r.llm_judge_overall for r in self.results]
        
        return {
            # Performance
            "latency_p50": statistics.median(response_times),
            "latency_p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0],
            "avg_response_time": statistics.mean(response_times),
            
            # Retrieval
            "precision_at_1": statistics.mean(p_at_1_scores),
            "recall_at_3": statistics.mean(recall_at_3_scores),
            "mrr": statistics.mean(mrr_scores),
            
            # Answer Quality
            "exact_match_rate": sum(exact_matches) / len(exact_matches),
            "avg_faithfulness": statistics.mean(faithfulness_scores),
            "avg_human_rating": statistics.mean(human_ratings),
            
            # LLM Judge Metrics
            "llm_judge_relevance": statistics.mean(llm_relevance_scores),
            "llm_judge_accuracy": statistics.mean(llm_accuracy_scores),
            "llm_judge_completeness": statistics.mean(llm_completeness_scores),
            "llm_judge_faithfulness": statistics.mean(llm_faithfulness_scores),
            "llm_judge_overall": statistics.mean(llm_overall_scores),
            
            # Category breakdown
            "category_performance": self._calculate_category_metrics()
        }
    
    def _calculate_category_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by category"""
        category_results = {}
        
        for result in self.results:
            if result.category not in category_results:
                category_results[result.category] = []
            category_results[result.category].append(result)
        
        metrics_by_category = {}
        for category, results in category_results.items():
            if results:
                metrics_by_category[category] = {
                    "avg_response_time": statistics.mean([r.response_time for r in results]),
                    "avg_faithfulness": statistics.mean([r.faithfulness_score for r in results]),
                    "avg_human_rating": statistics.mean([r.human_rating for r in results]),
                    "llm_judge_overall": statistics.mean([r.llm_judge_overall for r in results]),
                    "llm_judge_relevance": statistics.mean([r.llm_judge_relevance for r in results]),
                    "llm_judge_accuracy": statistics.mean([r.llm_judge_accuracy for r in results])
                }
        
        return metrics_by_category
    
    def _generate_report(self, metrics: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            "evaluation_metadata": {
                "timestamp": timestamp,
                "total_queries": len(self.test_queries),
                "successful_queries": len(self.results),
                "total_evaluation_time": total_time,
                "api_endpoint": API_BASE_URL
            },
            "retrieval_metrics": {
                "precision_at_1": round(metrics.get("precision_at_1", 0), 3),
                "recall_at_3": round(metrics.get("recall_at_3", 0), 3),
                "mrr": round(metrics.get("mrr", 0), 3)
            },
            "answer_quality_metrics": {
                "exact_match_rate": round(metrics.get("exact_match_rate", 0), 3),
                "avg_faithfulness_score": round(metrics.get("avg_faithfulness", 0), 3),
                "avg_human_rating": round(metrics.get("avg_human_rating", 0), 1)
            },
            "llm_judge_metrics": {
                "overall_quality": round(metrics.get("llm_judge_overall", 0), 3),
                "relevance": round(metrics.get("llm_judge_relevance", 0), 3),
                "accuracy": round(metrics.get("llm_judge_accuracy", 0), 3),
                "completeness": round(metrics.get("llm_judge_completeness", 0), 3),
                "faithfulness": round(metrics.get("llm_judge_faithfulness", 0), 3)
            },
            "performance_metrics": {
                "latency_p50_ms": round(metrics.get("latency_p50", 0) * 1000, 2),
                "latency_p95_ms": round(metrics.get("latency_p95", 0) * 1000, 2),
                "avg_response_time_ms": round(metrics.get("avg_response_time", 0) * 1000, 2)
            },
            "category_breakdown": metrics.get("category_performance", {}),
            "detailed_results": [
                {
                    "query": r.query,
                    "category": r.category,
                    "response_time_ms": round(r.response_time * 1000, 2),
                    "confidence": r.confidence,
                    "exact_match": r.exact_match,
                    "faithfulness_score": round(r.faithfulness_score, 3),
                    "human_rating": r.human_rating,
                    "sources_count": len(r.sources),
                    "api_metrics": r.performance_metrics,
                    "llm_judge": {
                        "overall": round(r.llm_judge_overall, 3),
                        "relevance": round(r.llm_judge_relevance, 3),
                        "accuracy": round(r.llm_judge_accuracy, 3),
                        "completeness": round(r.llm_judge_completeness, 3),
                        "faithfulness": round(r.llm_judge_faithfulness, 3),
                        "reasoning": r.judge_reasoning
                    }
                }
                for r in self.results
            ]
        }
        
        return report
    
    def _save_results(self, report: Dict[str, Any]):
        """Save evaluation results to files"""
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = os.path.join(OUTPUT_DIR, f"evaluation_report_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        txt_file = os.path.join(OUTPUT_DIR, f"evaluation_summary_{timestamp}.txt")
        with open(txt_file, 'w') as f:
            f.write("JioPay RAG System - Evaluation Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {report['evaluation_metadata']['timestamp']}\n")
            f.write(f"Total Queries: {report['evaluation_metadata']['total_queries']}\n")
            f.write(f"Successful Queries: {report['evaluation_metadata']['successful_queries']}\n\n")
            
            f.write("RETRIEVAL METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Precision@1: {report['retrieval_metrics']['precision_at_1']}\n")
            f.write(f"Recall@3: {report['retrieval_metrics']['recall_at_3']}\n")
            f.write(f"MRR: {report['retrieval_metrics']['mrr']}\n\n")
            
            f.write("ANSWER QUALITY METRICS (Legacy)\n")
            f.write("-" * 35 + "\n")
            f.write(f"Avg Faithfulness: {report['answer_quality_metrics']['avg_faithfulness_score']}\n")
            f.write(f"Avg Human Rating: {report['answer_quality_metrics']['avg_human_rating']}/5\n\n")
            
            f.write("LLM JUDGE METRICS\n")
            f.write("-" * 18 + "\n")
            f.write(f"Overall Quality: {report['llm_judge_metrics']['overall_quality']}\n")
            f.write(f"Relevance: {report['llm_judge_metrics']['relevance']}\n")
            f.write(f"Accuracy: {report['llm_judge_metrics']['accuracy']}\n")
            f.write(f"Completeness: {report['llm_judge_metrics']['completeness']}\n")
            f.write(f"Faithfulness: {report['llm_judge_metrics']['faithfulness']}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Latency P50: {report['performance_metrics']['latency_p50_ms']}ms\n")
            f.write(f"Latency P95: {report['performance_metrics']['latency_p95_ms']}ms\n")
            f.write(f"Avg Response Time: {report['performance_metrics']['avg_response_time_ms']}ms\n\n")
            
            f.write("CATEGORY BREAKDOWN\n")
            f.write("-" * 18 + "\n")
            for category, metrics in report['category_breakdown'].items():
                f.write(f"{category.upper()}\n")
                f.write(f"  Avg Response Time: {metrics['avg_response_time']*1000:.2f}ms\n")
                f.write(f"  Legacy Faithfulness: {metrics['avg_faithfulness']:.3f}\n")
                f.write(f"  Human Rating: {metrics['avg_human_rating']:.1f}/5\n")
                f.write(f"  LLM Judge Overall: {metrics['llm_judge_overall']:.3f}\n")
                f.write(f"  LLM Judge Relevance: {metrics['llm_judge_relevance']:.3f}\n")
                f.write(f"  LLM Judge Accuracy: {metrics['llm_judge_accuracy']:.3f}\n\n")
        
        print(f"\n📊 Results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Summary: {txt_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print evaluation summary to console"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        print("\n🎯 RETRIEVAL PERFORMANCE")
        print(f"  Precision@1: {report['retrieval_metrics']['precision_at_1']:.3f}")
        print(f"  Recall@3: {report['retrieval_metrics']['recall_at_3']:.3f}")
        print(f"  MRR: {report['retrieval_metrics']['mrr']:.3f}")
        
        print("\n✅ ANSWER QUALITY (Legacy)")
        print(f"  Faithfulness Score: {report['answer_quality_metrics']['avg_faithfulness_score']:.3f}")
        print(f"  Human Rating: {report['answer_quality_metrics']['avg_human_rating']:.1f}/5")
        
        print("\n🤖 LLM JUDGE EVALUATION")
        print(f"  Overall Quality: {report['llm_judge_metrics']['overall_quality']:.3f}")
        print(f"  Relevance: {report['llm_judge_metrics']['relevance']:.3f}")
        print(f"  Accuracy: {report['llm_judge_metrics']['accuracy']:.3f}")
        print(f"  Completeness: {report['llm_judge_metrics']['completeness']:.3f}")
        print(f"  Faithfulness: {report['llm_judge_metrics']['faithfulness']:.3f}")
        
        print("\n⚡ PERFORMANCE")
        print(f"  P50 Latency: {report['performance_metrics']['latency_p50_ms']:.2f}ms")
        print(f"  P95 Latency: {report['performance_metrics']['latency_p95_ms']:.2f}ms")
        print(f"  Avg Response Time: {report['performance_metrics']['avg_response_time_ms']:.2f}ms")
        
        print("\n📈 CATEGORY PERFORMANCE")
        for category, metrics in report['category_breakdown'].items():
            print(f"  {category.upper()}: LLM Judge {metrics['llm_judge_overall']:.2f}, Human {metrics['avg_human_rating']:.1f}/5, {metrics['avg_response_time']*1000:.0f}ms")

def main():
    """Main evaluation function"""
    print("🔬 JioPay RAG System - Comprehensive Evaluation Protocol")
    print("Testing retrieval performance, answer quality, and system performance")
    print("=" * 70)
    
    evaluator = JioPayEvaluator()
    report = evaluator.run_evaluation()
    
    if report:
        evaluator.print_summary(report)
        print(f"\n🎉 Evaluation completed! Results saved to: {OUTPUT_DIR}")
    else:
        print("\n❌ Evaluation failed - please check API availability")

if __name__ == "__main__":
    main()
