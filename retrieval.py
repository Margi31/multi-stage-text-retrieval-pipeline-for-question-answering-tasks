import pandas as pd
from sentence_transformers import SentenceTransformer, util

def load_dataset(dataset_path):
    # Load the dataset based on its format
    if dataset_path.endswith('.csv'):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith('.json'):
        return pd.read_json(dataset_path)
    # Add other formats if necessary

def candidate_retrieval(query, dataset_path, top_k=10):
    # Load pre-trained embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Check the available columns
    print(f"Columns in dataset: {list(dataset.columns)}")
    
    # Adjust column name to match the one in your dataset (e.g., 'long_answers')
    if 'long_answers' in dataset.columns:
        passages = dataset['long_answers'].tolist()
    else:
        raise KeyError("Column 'long_answers' not found in dataset.")  # Adjust based on your dataset
    
    # Encode query and passages
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)

    # Perform retrieval based on similarity
    top_k_results = util.semantic_search(query_embedding, passage_embeddings, top_k=top_k)[0]

    # Retrieve top-k passages
    top_k_passages = [passages[hit['corpus_id']] for hit in top_k_results]
    
    return top_k_passages
