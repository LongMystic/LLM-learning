"""
RAG Evaluation Script
Measures: grounding rate, answer correctness, latency
"""

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time
import json
from typing import List, Dict
import re

# Add current directory to path to import from basic_rag
sys.path.append(str(Path(__file__).parent))
from basic_rag import build_rag_system, query_rag


# ============================================
# Evaluation metrics
# ============================================

def check_grounding(answer: str, retrieved_chunks: List[Dict], 
                   expected_source: str = None) -> bool:
    """
    Check if answer is grounded in retrieved chunks
    
    Args:
        answer: Generated answer
        retrieved_chunks: List of retrieved chunks
        expected_source: Optional expected source document
    
    Returns:
        True if answer appears to be grounded in retrieved chunks
    """
    # Simple heuristic: check if key phrases from answer appear in chunks
    answer_lower = answer.lower()
    
    # Extract key words from answer (simple approach)
    answer_words = set(re.findall(r'\b\w{4,}\b', answer_lower))  # Words with 4+ chars
    
    # Check if any chunk contains multiple answer words
    for chunk in retrieved_chunks:
        chunk_text_lower = chunk['text'].lower()
        chunk_words = set(re.findall(r'\b\w{4,}\b', chunk_text_lower))
        
        # If significant overlap, consider it grounded
        overlap = len(answer_words & chunk_words)
        if overlap >= 3:  # At least 3 key words overlap
            if expected_source:
                # Check if source matches
                chunk_source = chunk.get('metadata', {}).get('source', '')
                if expected_source in chunk_source or chunk_source in expected_source:
                    return True
            else:
                return True
    
    return False


def calculate_f1_score(predicted: str, expected: str) -> float:
    """
    Calculate F1 score between predicted and expected answers
    
    Args:
        predicted: Generated answer
        expected: Expected answer
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    # Tokenize (simple word-based)
    pred_tokens = set(predicted.lower().split())
    exp_tokens = set(expected.lower().split())
    
    if len(pred_tokens) == 0 and len(exp_tokens) == 0:
        return 1.0
    
    if len(pred_tokens) == 0 or len(exp_tokens) == 0:
        return 0.0
    
    # Calculate precision and recall
    intersection = pred_tokens & exp_tokens
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(intersection) / len(exp_tokens) if len(exp_tokens) > 0 else 0.0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def exact_match(predicted: str, expected: str) -> bool:
    """
    Check if predicted answer exactly matches expected (case-insensitive)
    """
    return predicted.lower().strip() == expected.lower().strip()


# ============================================
# Evaluation dataset
# ============================================

def load_eval_dataset(filepath: str = None) -> List[Dict]:
    """
    Load evaluation dataset
    
    Format: List of dicts with:
    - "question": The question
    - "expected_answer": Expected answer
    - "expected_source": Optional source document
    - "context": Optional context for building RAG
    
    If filepath is None, returns a sample dataset
    """
    if filepath:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Sample dataset
    return [
        {
            "question": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "expected_source": "doc_0",
            "context": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions."
        },
        {
            "question": "How do transformers work?",
            "expected_answer": "Transformers use attention mechanisms to process sequences of data.",
            "expected_source": "doc_2",
            "context": "Transformers are a type of neural network architecture introduced in 2017. They use attention mechanisms to process sequences of data, making them particularly effective for natural language processing tasks."
        },
        {
            "question": "What is deep learning?",
            "expected_answer": "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
            "expected_source": "doc_1",
            "context": "Deep learning is a type of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns in data, such as recognizing images, understanding speech, or translating languages."
        }
    ]


# ============================================
# Evaluation pipeline
# ============================================

def evaluate_rag(collection, embedding_model, eval_dataset: List[Dict], 
                 model="llama3.2", top_k=3) -> Dict:
    """
    Evaluate RAG system on a dataset
    
    Args:
        collection: Chroma collection
        embedding_model: SentenceTransformer model
        eval_dataset: List of evaluation examples
        model: Ollama model name
        top_k: Number of chunks to retrieve
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "total_questions": len(eval_dataset),
        "grounded_answers": 0,
        "exact_matches": 0,
        "f1_scores": [],
        "latencies": [],
        "detailed_results": []
    }
    
    print("=" * 60)
    print("Evaluating RAG System")
    print("=" * 60)
    print(f"Total questions: {len(eval_dataset)}\n")
    
    for i, example in enumerate(eval_dataset):
        question = example["question"]
        expected_answer = example["expected_answer"]
        expected_source = example.get("expected_source")
        
        print(f"[{i+1}/{len(eval_dataset)}] Question: {question}")
        
        # Query RAG
        start_time = time.time()
        answer, retrieved_chunks = query_rag(
            collection, embedding_model, question, top_k=top_k, model=model
        )
        latency = time.time() - start_time
        
        # Evaluate
        is_grounded = check_grounding(answer, retrieved_chunks, expected_source)
        is_exact = exact_match(answer, expected_answer)
        f1 = calculate_f1_score(answer, expected_answer)
        
        # Update metrics
        if is_grounded:
            results["grounded_answers"] += 1
        if is_exact:
            results["exact_matches"] += 1
        
        results["f1_scores"].append(f1)
        results["latencies"].append(latency)
        
        results["detailed_results"].append({
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": answer,
            "is_grounded": is_grounded,
            "is_exact_match": is_exact,
            "f1_score": f1,
            "latency": latency,
            "retrieved_sources": [chunk.get('metadata', {}).get('source', 'unknown') 
                                 for chunk in retrieved_chunks]
        })
        
        print(f"  Grounded: {is_grounded}, Exact: {is_exact}, F1: {f1:.3f}, "
              f"Latency: {latency:.2f}s")
        print()
    
    # Calculate aggregate metrics
    results["grounding_rate"] = results["grounded_answers"] / results["total_questions"]
    results["exact_match_rate"] = results["exact_matches"] / results["total_questions"]
    results["avg_f1"] = sum(results["f1_scores"]) / len(results["f1_scores"])
    results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
    results["p95_latency"] = sorted(results["latencies"])[int(len(results["latencies"]) * 0.95)]
    
    return results


def print_evaluation_summary(results: Dict):
    """
    Print evaluation summary
    """
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total questions: {results['total_questions']}")
    print(f"\nAccuracy Metrics:")
    print(f"  Grounding rate: {results['grounding_rate']:.2%} "
          f"({results['grounded_answers']}/{results['total_questions']})")
    print(f"  Exact match rate: {results['exact_match_rate']:.2%} "
          f"({results['exact_matches']}/{results['total_questions']})")
    print(f"  Average F1 score: {results['avg_f1']:.3f}")
    print(f"\nLatency Metrics:")
    print(f"  Average latency: {results['avg_latency']:.2f}s")
    print(f"  P95 latency: {results['p95_latency']:.2f}s")
    print("=" * 60)


def save_evaluation_results(results: Dict, filepath: str = "rag_evaluation_results.json"):
    """
    Save evaluation results to JSON file
    """
    # Convert to JSON-serializable format
    output = {
        "summary": {
            "total_questions": results["total_questions"],
            "grounding_rate": results["grounding_rate"],
            "exact_match_rate": results["exact_match_rate"],
            "avg_f1": results["avg_f1"],
            "avg_latency": results["avg_latency"],
            "p95_latency": results["p95_latency"],
        },
        "detailed_results": results["detailed_results"]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")


# ============================================
# Example usage
# ============================================

if __name__ == "__main__":
    # Load evaluation dataset
    eval_dataset = load_eval_dataset()
    
    # Build documents from eval dataset
    documents = [ex["context"] for ex in eval_dataset]
    
    # Build RAG system
    print("Building RAG system for evaluation...")
    collection, embedding_model = build_rag_system(documents)
    
    # Evaluate
    results = evaluate_rag(collection, embedding_model, eval_dataset, top_k=3)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    save_evaluation_results(results)

