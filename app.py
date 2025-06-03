import os
import boto3
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

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": ["http://localhost:5173", "https://main.d123456.amplifyapp.com"]}})

# Set TogetherAI API key
together_api_key = os.getenv('TOGETHER_API_KEY')
if not together_api_key:
    raise ValueError("TOGETHER_API_KEY not set")
client = Together(api_key=together_api_key)

# Global variables
index = None
chunks = None
EMBEDDINGS_CACHE = '/tmp/embeddings_cache.pkl'
CHUNKS_CACHE = '/tmp/chunks_cache.pkl'

def download_pdf_from_s3(bucket_name, pdf_key, local_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, pdf_key, local_path)
        print(f"Downloaded {pdf_key} from S3 to {local_path}")
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise

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
            text = re.sub(r'\d+,\d+.*?(?=\n|$)', '', text)
            text = re.sub(r'[^\w\s.,!?]', '', text)
            print(f"Extracted text in {time.time() - start_time:.2f} seconds")
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ''

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
    print(f"Split into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
    return chunks

def create_embeddings(texts):
    start_time = time.time()
    valid_texts = [t for t in texts if len(t) > 0]
    if not valid_texts:
        return np.array([]), []
    try:
        response = client.embeddings.create(
            model='togethercomputer/m2-bert-80M-8k-retrieval',
            input=valid_texts
        )
        embeddings = [item.embedding for item in response.data]
        print(f"Created {len(embeddings)} embeddings in {time.time() - start_time:.2f} seconds")
        return np.array(embeddings, dtype='float32'), list(range(len(valid_texts)))
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return np.array([]), []

def build_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_cache(embeddings, chunks, valid_indices):
    try:
        with open(EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(CHUNKS_CACHE, 'wb') as f:
            pickle.dump({'chunks': chunks, 'indices': valid_indices}, f)
        print("Cached embeddings and chunks")
    except Exception as e:
        print(f"Error saving cache: {e}")

def load_cache():
    try:
        if not (os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CHUNKS_CACHE)):
            return None, None, None
        with open(EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
        with open(CHUNKS_CACHE, 'rb') as f:
            cache_data = pickle.load(f)
        return embeddings, cache_data['chunks'], cache_data['indices']
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None, None

def retrieve_relevant_chunks(query, index, chunks, k=3):
    if index is None:
        return []
    query_embedding = create_embeddings([query])[0]
    if query_embedding.size == 0:
        return []
    distances, indices = index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0] if idx < len(chunks)]

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

def generate_response(query, context_chunks):
    context = '\n\n'.join(context_chunks) if context_chunks else "No context available."
    prompt = f"""You are a knowledgeable AI specializing in astrophysics. 
    Respond clearly in 50–100 words. If vague, ask for clarification.
    Context: {context}
    Query: {query}
    Answer:"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=150
        )
        return split_response_into_chunks(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generating response: {e}")
        return ["Sorry, I couldn't generate a response."]

def initialize_rag():
    global index, chunks
    bucket_name = os.getenv('S3_BUCKET_NAME')
    pdf_key = 'Astronomy.pdf'
    local_pdf_path = '/tmp/Astronomy.pdf'
    download_pdf_from_s3(bucket_name, pdf_key, local_pdf_path)
    text = extract_text_from_pdf(local_pdf_path)
    if not text:
        chunks = []
        index = None
        return
    chunks_temp = split_text_into_chunks(text)[:500]
    embeddings, valid_indices = create_embeddings(chunks_temp)
    if embeddings.size == 0:
        chunks = []
        index = None
        return
    chunks = [chunks_temp[i] for i in valid_indices]
    index = build_faiss_index(embeddings)
    save_cache(embeddings, chunks, valid_indices)

try:
    initialize_rag()
except Exception as e:
    print(f"Failed to initialize RAG: {e}")
    chunks = []
    index = None

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    if query.lower().strip() in ['hi', 'hello']:
        return jsonify({"response": ["Greetings, space explorer! What's your question?"]})
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    response_chunks = generate_response(query, relevant_chunks)
    return jsonify({"response": response_chunks})

@app.route('/')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)