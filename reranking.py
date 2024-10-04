from sentence_transformers import CrossEncoder

def rerank(query, top_k_passages):
    # Load a cross-encoder model for reranking
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    # Create pairs of query and each passage
    query_passage_pairs = [[query, passage] for passage in top_k_passages]

    # Perform reranking
    reranked_scores = model.predict(query_passage_pairs)

    # Sort passages by their scores
    reranked_passages = [x for _, x in sorted(zip(reranked_scores, top_k_passages), reverse=True)]

    return reranked_passages
