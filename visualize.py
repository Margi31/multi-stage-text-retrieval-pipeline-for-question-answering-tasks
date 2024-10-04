import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_retrieval_distribution(retrieved_passages):
    """Visualizes the length distribution of the retrieved passages."""
    passage_lengths = [len(passage.split()) for passage in retrieved_passages]
    plt.figure(figsize=(10, 6))
    sns.histplot(passage_lengths, bins=20, kde=True)
    plt.title('Distribution of Passage Lengths')
    plt.xlabel('Passage Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.show()

def visualize_reranking_effect(retrieved_passages, reranked_passages):
    """Compares the lengths of passages before and after reranking."""
    retrieved_lengths = [len(p.split()) for p in retrieved_passages]
    reranked_lengths = [len(p.split()) for p in reranked_passages]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(retrieved_lengths)), retrieved_lengths, label="Before Reranking", marker='o')
    plt.plot(range(len(reranked_lengths)), reranked_lengths, label="After Reranking", marker='x')
    plt.title("Effect of Reranking on Passage Lengths")
    plt.xlabel("Passage Index")
    plt.ylabel("Passage Length (Words)")
    plt.legend()
    plt.show()

def visualize_ndcg_scores(ndcg_scores):
    """Plots the NDCG@10 scores."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ndcg_scores)), ndcg_scores, marker='o', color='g')
    plt.title('NDCG@10 Scores for Queries')
    plt.xlabel('Query Index')
    plt.ylabel('NDCG@10 Score')
    plt.show()

def summarize_results(retrieved_passages, reranked_passages, ndcg_score):
    """Prints a summary of results."""
    print(f"Total number of passages retrieved: {len(retrieved_passages)}")
    print(f"Total number of passages after reranking: {len(reranked_passages)}")
    print(f"Average NDCG@10 score: {sum(ndcg_score)/len(ndcg_score):.2f}")

def main():
    # Sample data - Replace this with actual data from your app.py
    retrieved_passages = [
        "Paris is the capital of France.", 
        "The capital city of France is Paris.", 
        "Paris, France's capital, is known for the Eiffel Tower."
    ]
    
    reranked_passages = [
        "The capital city of France is Paris.", 
        "Paris, France's capital, is known for the Eiffel Tower.", 
        "Paris is the capital of France."
    ]
    
    ndcg_scores = [0.78, 0.80, 0.82]  # Placeholder NDCG@10 scores

    # Visualizations
    visualize_retrieval_distribution(retrieved_passages)
    visualize_reranking_effect(retrieved_passages, reranked_passages)
    visualize_ndcg_scores(ndcg_scores)

    # Summary of results
    summarize_results(retrieved_passages, reranked_passages, ndcg_scores)

if __name__ == "__main__":
    main()
