# evaluation.py
from sklearn.metrics import ndcg_score
import numpy as np

def evaluate(query, reranked_passages, ground_truth, k=10):
    """
    Evaluates the retrieval results using NDCG@k.
    
    :param query: The query that was used for retrieval.
    :param reranked_passages: The reranked list of passages retrieved for the query.
    :param ground_truth: The list of ground truth answers.
    :param k: The cutoff for NDCG (default is 10).
    :return: The NDCG@k score.
    """
    # Create relevance scores: 1 for relevant passages, 0 for non-relevant
    relevance_scores = [1 if passage in ground_truth else 0 for passage in reranked_passages]

    # Convert to the format required by sklearn's ndcg_score
    relevance_scores = np.asarray(relevance_scores).reshape(1, -1)
    ideal_scores = np.asarray([1] * len(ground_truth) + [0] * (len(reranked_passages) - len(ground_truth))).reshape(1, -1)

    # Calculate NDCG@k
    ndcg = ndcg_score(ideal_scores, relevance_scores, k=k)

    print(f"NDCG@{k} for query '{query}': {ndcg}")
    return ndcg
