import os
import PyPDF2
import faiss
import numpy as np
from together import Together
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer
import re
import pickle
import time

# Load environment variables
load_dotenv()

# Set TogetherAI API key
together_api_key = os.getenv('TOGETHER_API_KEY')
if not together_api_key:
    raise ValueError("TOGETHER_API_KEY not set in environment variables")
client = Together(api_key=together_api_key)

#http://localhost:5173
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": ["https://astrobot-frontend.onrender.com"]}})  # Allow frontend origin

# Global variables for RAG components
index = None
chunks = None

# Paths for caching embeddings
EMBEDDINGS_CACHE = 'embeddings_cache.pkl'
CHUNKS_CACHE = 'chunks_cache.pkl'

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    start_time = time.time()
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            # Remove index-like content and non-alphanumeric characters
            text = re.sub(r'\d+,\d+.*?(?=\n|$)', '', text)
            text = re.sub(r'[^\w\s.,!?]', '', text)
            print(f"Extracted text (first 500 chars): {text[:500]}")
            print(f"Text extraction took {time.time() - start_time:.2f} seconds.")
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        print(f"Text extraction took {time.time() - start_time:.2f} seconds.")
        return ''

# Function to split text into chunks by tokens
def split_text_into_chunks(text, max_tokens=32):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    words = text.split()
    chunks = []
    current_chunk = ''
    current_tokens = 0
    
    for word in words:
        word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
        if current_tokens + word_tokens <= max_tokens:
            current_chunk += word + ' '
            current_tokens += word_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + ' '
            current_tokens = word_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    print(f"Text splitting created {len(chunks)} chunks in {time.time() - start_time:.2f} seconds.")
    return chunks

# Function to create embeddings using TogetherAI
def create_embeddings(texts):
    start_time = time.time()
    valid_texts = []
    valid_indices = []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch_size = 10
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        for j, text in enumerate(batch_texts):
            global_idx = i + j
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            char_count = len(text)
            print(f"Chunk {global_idx}: {char_count} chars, {token_count} tokens, '{text}'")
            if token_count > 32:
                print(f"Skipping chunk {global_idx}: {token_count} tokens exceed limit")
                continue
            valid_texts.append(text)
            valid_indices.append(global_idx)
    
    if not valid_texts:
        print("No valid texts for embedding.")
        print(f"Embedding creation took {time.time() - start_time:.2f} seconds.")
        return np.array([]), []
    
    embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        batch_indices = valid_indices[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model='togethercomputer/m2-bert-80M-8k-retrieval',
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error creating embeddings for batch {i//batch_size}: {e}")
            for idx in batch_indices:
                if idx in valid_indices:
                    valid_idx = valid_indices.index(idx)
                    valid_texts.pop(valid_idx)
                    valid_indices.pop(valid_idx)
    
    if not embeddings:
        print("No embeddings created.")
        print(f"Embedding creation took {time.time() - start_time:.2f} seconds.")
        return np.array([]), []
    
    print(f"Created {len(embeddings)} embeddings in {time.time() - start_time:.2f} seconds.")
    return np.array(embeddings, dtype='float32'), valid_indices

# Function to build FAISS index
def build_faiss_index(embeddings):
    start_time = time.time()
    if embeddings.size == 0:
        print("No embeddings to build FAISS index.")
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built in {time.time() - start_time:.2f} seconds.")
    return index

# Function to cache embeddings and chunks
def save_cache(embeddings, chunks, valid_indices):
    start_time = time.time()
    try:
        # Ensure directory is writable
        cache_dir = os.path.dirname(EMBEDDINGS_CACHE) or '.'
        if not os.access(cache_dir, os.W_OK):
            print(f"No write permission in {cache_dir}")
            return
        with open(EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(CHUNKS_CACHE, 'wb') as f:
            pickle.dump({'chunks': chunks, 'indices': valid_indices}, f)
        print(f"Embeddings and chunks cached successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error saving cache: {e}")

# Function to load cached embeddings and chunks
def load_cache():
    start_time = time.time()
    try:
        if not (os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CHUNKS_CACHE)):
            print("Cache files not found.")
            return None, None, None
        with open(EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
        with open(CHUNKS_CACHE, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded cached embeddings and chunks in {time.time() - start_time:.2f} seconds.")
        return embeddings, cache_data['chunks'], cache_data['indices']
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None, None

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, k=3):
    if index is None:
        return []
    query_embedding = create_embeddings([query])[0]
    if query_embedding.size == 0:
        return []
    distances, indices = index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0] if idx < len(chunks)]

# Function to split response into chunks
def split_response_into_chunks(response, max_length=100):
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ' '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to generate response using TogetherAI
def generate_response(query, context_chunks):
    context = '\n\n'.join(context_chunks) if context_chunks else "No document context available."
    prompt = f"""You are a knowledgeable and concise AI assistant specializing in space technology, astrophysics, orbital mechanics, cosmology, and theoretical physics. 
    When the user asks a question in these areas, respond clearly and accurately in 50–100 words, focusing on key points. 
    Avoid unnecessary detail or repetition.
    If the question is vague, prompt the user with:
    "Could you please clarify your question?"
Context:
{context}

Query:
{query}

Answer:
"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=150,
            temperature=0.7
        )
        full_response = response.choices[0].message.content.strip()
        return split_response_into_chunks(full_response)
    except Exception as e:
        print(f"Error generating response: {e}")
        return ["Sorry, I couldn't generate a response. Please try again."]

# Initialize RAG system
def initialize_rag():
    global index, chunks
    pdf_path = 'Astronomy.pdf'
    start_time = time.time()
    
    # Try loading cached embeddings
    cached_embeddings, cached_chunks, cached_indices = load_cache()
    if cached_embeddings is not None and cached_chunks is not None:
        chunks = cached_chunks
        index = build_faiss_index(cached_embeddings)
        if index is not None:
            print(f"RAG system initialized from cache in {time.time() - start_time:.2f} seconds.")
            return
    
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("Warning: No text extracted from PDF. Proceeding with empty context.")
        chunks = []
        index = None
        print(f"RAG initialization took {time.time() - start_time:.2f} seconds.")
        return
    
    print("Splitting text into chunks...")
    chunks_temp = split_text_into_chunks(text)[:500]  # Limit to 500 chunks
    if not chunks_temp:
        print("Warning: No chunks created. Proceeding with empty context.")
        chunks = []
        index = None
        print(f"RAG initialization took {time.time() - start_time:.2f} seconds.")
        return
    
    print("Creating embeddings...")
    embeddings, valid_indices = create_embeddings(chunks_temp)
    if embeddings.size == 0:
        print("Warning: No embeddings created. Proceeding with empty context.")
        chunks = []
        index = None
        print(f"RAG initialization took {time.time() - start_time:.2f} seconds.")
        return
    
    print("Building FAISS index...")
    chunks = [chunks_temp[i] for i in valid_indices]
    index = build_faiss_index(embeddings)
    if index is None:
        print("Warning: Failed to build FAISS index. Proceeding with empty context.")
        chunks = []
        index = None
        print(f"RAG initialization took {time.time() - start_time:.2f} seconds.")
        return
    
    # Cache embeddings and chunks
    save_cache(embeddings, chunks, valid_indices)
    print(f"RAG system initialized in {time.time() - start_time:.2f} seconds.")

# Call initialization at app startup
try:
    initialize_rag()
except Exception as e:
    print(f"Failed to initialize RAG: {e}")
    chunks = []
    index = None

# API endpoint to handle queries
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Normalize query
    lowercase_query = query.lower().strip()
    
    # Handle greetings
    if lowercase_query in ['hi', 'hello', 'hey']:
        return jsonify({"response": ["Greetings, space explorer! Cooper here, ready to answer your cosmic questions. What's on your mind?"]})
    
    # Handle vague or non-question inputs
    vague_phrases = ['ok', 'okk', 'okay', 'sure', 'cool', 'astro', 'astrobot']
    is_vague = (lowercase_query in vague_phrases or 
                len(lowercase_query.split()) <= 3 and '?' not in query)
    
    if is_vague:
        return jsonify({"response": ["Bazinga! What's your cosmic question, space explorer? I'm ready to enlighten you!"]})
    
    # Process through RAG pipeline for valid queries
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    response_chunks = generate_response(query, relevant_chunks)
    return jsonify({"response": response_chunks})

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
