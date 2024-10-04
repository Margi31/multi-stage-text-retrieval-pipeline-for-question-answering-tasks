import sys
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from huggingface_hub import login

# Login to Hugging Face with the provided token
token = "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"
login(token)

# Define paths to the datasets
naturalqa_path = 'data/naturalqa/Natural-Questions-Filtered.csv'

def load_dataset(dataset_path):
    # Load dataset depending on the file type
    if dataset_path.endswith('.csv'):
        return pd.read_csv(dataset_path)
    else:
        raise ValueError("Unsupported dataset format. Only CSV is supported.")

def candidate_retrieval(query, dataset_path, top_k=10):
    # Load pre-trained embedding model
    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Model loaded successfully!")

    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Check the available columns
    print(f"Columns in dataset: {list(dataset.columns)}")
    
    if 'long_answers' in dataset.columns:
        # Limiting the dataset size for testing
        passages = dataset['long_answers'].tolist()[:100]  # Limit to 100 passages for testing
    else:
        raise KeyError("Column 'long_answers' not found in dataset.")
    
    # Encode query and passages
    print("Encoding query and passages...")
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)
    print("Encoding complete!")

    # Perform retrieval based on similarity
    print("Performing semantic search...")
    top_k_results = util.semantic_search(query_embedding, passage_embeddings, top_k=top_k)[0]
    print("Search complete!")

    # Retrieve top-k passages
    top_k_passages = [passages[hit['corpus_id']] for hit in top_k_results]
    
    return top_k_passages

def rerank(query, top_k_passages):
    # Dummy rerank function (this is where you would apply your reranking model)
    print("Reranking passages...")
    reranked_passages = sorted(top_k_passages, key=lambda x: len(x))  # Sort by length as a dummy rerank
    print("Reranking complete!")
    return reranked_passages

def evaluate(query, retrieved_passages, ground_truth):
    # Dummy evaluation function to compute NDCG@10 (simplified)
    print("Evaluating performance...")
    ndcg_score = 0.78  # Placeholder score (replace with actual calculation)
    print(f"NDCG@10 for the query '{query}': {ndcg_score}")

def main():
    # Example query
    query = "What is the capital of France?"

    # Stage 1: Candidate Retrieval
    try:
        top_k_passages = candidate_retrieval(query, naturalqa_path, top_k=10)
    except Exception as e:
        print(f"Error during candidate retrieval: {e}")
        return

    # Stage 2: Reranking
    try:
        reranked_passages = rerank(query, top_k_passages)
    except Exception as e:
        print(f"Error during reranking: {e}")
        return

    # Stage 3: Evaluation
    try:
        # Example ground truth for evaluation
        ground_truth = ['Paris is the capital of France.']  # Adjust based on your dataset
        evaluate(query, reranked_passages, ground_truth)
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
