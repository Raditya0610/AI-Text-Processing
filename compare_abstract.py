from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

# Utility functions
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
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

# Step 1: Load PDFs from a folder
pdf_folder = "pdf/"
abstracts = []
pdf_filenames = []
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text(text)
        abstracts.append(preprocessed_text)
        pdf_filenames.append(os.path.splitext(pdf_file)[0])

if len(abstracts) < 2:
    print("Error: Minimal memiliki 2 file PDF di folder 'pdf_abstrak'")
    exit(1)

# Step 2: TF-IDF Calculation
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(abstracts)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 3: Generate Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=pdf_filenames,
            yticklabels=pdf_filenames)
plt.savefig("heatmap.png", format='png')

# Step 4: Analyze Similarities
n = cosine_sim.shape[0]
similar_pairs = []
for i in range(n):
    for j in range(i + 1, n):
        similarity_percent = cosine_sim[i, j] * 100  # Convert to percentage
        similar_pairs.append({
            "pair": f"{pdf_filenames[i]} & {pdf_filenames[j]}",
            "similarity": similarity_percent
        })

print("Heatmap saved as 'heatmap.png'")
for pair in similar_pairs:
    print(f"{pair['pair']}: {pair['similarity']:.2f}% similar")