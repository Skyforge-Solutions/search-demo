import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def compute_embeddings(data, _model):
    return _model.encode([item['What I am looking for'] for item in data])

def semantic_search(query, data, embeddings, model, top_k=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(data[idx], similarities[idx]) for idx in top_indices]

def main():
    st.title('search demp')
    
    data = load_data('Konnect_data.json')
    model = load_model()
    embeddings = compute_embeddings(data, model)
    
    query = st.text_input('Enter your query')
    
    if query:
        results = semantic_search(query, data, embeddings, model)
        
        # Display results
        st.write(f"Top {len(results)} Matches:")
        for result, score in results:
            with st.expander(f"Match (Similarity: {score:.2f})", expanded=False):
                st.write(f"**Name:** {result.get('name', 'N/A')}")
                st.write(f"**Current Company:** {result.get('Current Company', 'N/A')}")
                st.write(f"**Experience Level:** {result.get('Experience Level', 'N/A')}")
                st.write(f"**Tech:** {result.get('Tech', 'N/A')}")
                st.write(f"**What I am Looking For:** {result.get('What I am looking for', 'N/A')}")
                st.write(f"**Booking Price (INR):** {result.get('Booking Price INR', 'N/A')}")

if __name__ == '__main__':
    main()
