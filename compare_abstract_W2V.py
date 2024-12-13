import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from scipy.spatial.distance import pdist, squareform

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    tokens = word_tokenize(text)
    # Stopwords Bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))
  
    additional_stopwords = {
        'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 
        'dan', 'atau', 'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 
        'dia', 'mereka', 'kita', 'akan', 'bisa', 'ada', 'tidak', 'sampai'
    }
    stop_words.update(additional_stopwords)
    filtered_tokens = [
        word for word in tokens 
        if word not in stop_words 
        and word not in string.punctuation
    ]
    
    return filtered_tokens

def jaccard_similarity(doc1, doc2):
    """
    Calculate Jaccard similarity between two documents
    """
    intersection = len(set(doc1) & set(doc2))
    union = len(set(doc1) | set(doc2))
    return intersection / union if union != 0 else 0

def word2vec_similarity(docs, vector_size=100, window=5, min_count=1):
    """
    Calculate document similarity using Word2Vec embeddings
    """
    # Train Word2Vec model
    model = Word2Vec(sentences=docs, vector_size=vector_size, 
                     window=window, min_count=min_count, workers=4)
    
    # Calculate document vectors by averaging word vectors
    doc_vectors = []
    for doc in docs:
        # Filter out words not in vocabulary
        doc_words = [word for word in doc if word in model.wv.key_to_index]
        
        # If no words, use zero vector
        if not doc_words:
            doc_vector = np.zeros(vector_size)
        else:
            # Average word vectors
            doc_vector = np.mean([model.wv[word] for word in doc_words], axis=0)
        
        doc_vectors.append(doc_vector)
    
    # Calculate cosine similarity between document vectors
    doc_vectors = np.array(doc_vectors)
    cosine_sim = 1 - squareform(pdist(doc_vectors, metric='cosine'))
    
    return cosine_sim

def analyze_document_similarities(pdf_folder, method='jaccard'):
    """
    Analyze document similarities using specified method
    """
    # Step 1: Load PDFs from folder
    abstracts = []
    pdf_filenames = []
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            preprocessed_tokens = preprocess_text(text)
            abstracts.append(preprocessed_tokens)
            pdf_filenames.append(os.path.splitext(pdf_file)[0])
    
    # Check if we have enough documents
    if len(abstracts) < 2:
        print("Error: Minimal 2 file PDF di folder")
        return None
    
    # Step 2: Calculate Similarity
    if method == 'jaccard':
        # Calculate Jaccard similarity
        n = len(abstracts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = jaccard_similarity(abstracts[i], abstracts[j])
    
    elif method == 'word2vec':
        # Calculate Word2Vec similarity
        similarity_matrix = word2vec_similarity(abstracts)
    
    # Step 3: Analyze Similarities
    return similarity_matrix, pdf_filenames

def combine_similarities(jaccard_matrix, word2vec_matrix, weight_jaccard=0.5, weight_word2vec=0.5):
    """
    Combine Jaccard and Word2Vec similarity matrices using weighted sum.
    """
    # Ensure weights sum to 1
    if weight_jaccard + weight_word2vec != 1:
        raise ValueError("The sum of weights must equal 1.")
    
    # Combine matrices
    combined_matrix = (weight_jaccard * jaccard_matrix) + (weight_word2vec * word2vec_matrix)
    return combined_matrix

def analyze_combined_similarities(pdf_folder, weight_jaccard=0.5, weight_word2vec=0.5):
    """
    Analyze document similarities using combined Jaccard and Word2Vec methods.
    """
    # Calculate Jaccard and Word2Vec similarities
    jaccard_matrix, pdf_filenames = analyze_document_similarities(pdf_folder, method='jaccard')
    word2vec_matrix, _ = analyze_document_similarities(pdf_folder, method='word2vec')
    
    if jaccard_matrix is None or word2vec_matrix is None:
        return None
    
    # Combine similarities
    combined_matrix = combine_similarities(jaccard_matrix, word2vec_matrix, 
                                           weight_jaccard, weight_word2vec)
    
    # Visualize combined similarities
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=pdf_filenames,
                yticklabels=pdf_filenames)
    plt.title("Document Similarity - Combined Method")
    plt.tight_layout()
    plt.savefig("heatmap_combined.png", format='png')
    print("Heatmap saved as 'heatmap_combined.png'")
    
    # Print pairwise similarities
    print("Pairwise Similarities:")
    n = combined_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            similarity_percent = combined_matrix[i, j] * 100
            print(f"{pdf_filenames[i]} & {pdf_filenames[j]}: {similarity_percent:.2f}% similar")
    
    return combined_matrix

pdf_folder = "pdf/"

print("Combined Similarity Method:")
combined_result = analyze_combined_similarities(pdf_folder, weight_jaccard=0.6, weight_word2vec=0.4)
