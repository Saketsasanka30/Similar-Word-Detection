# ==========================================================
# SIMILAR WORD PREDICTION USING NLP
# ==========================================================
import re
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# STEP 1: Create the Corpus
# ----------------------------------------------------------
corpus = [
    "computer technology software system data network",
    "computer software technology programming data system",
    "computer programming software code technology system",
    "computer network data internet technology system",
    "software programming code application computer technology",
    "data database analytics machine learning computer",
    "technology internet network computer digital system",
    "system software hardware computer network",
    "programming code software developer computer",
    "machine learning data analytics artificial intelligence",
    "database data storage system software",
    "internet network communication digital technology",
    "cybersecurity network system data security",
    "artificial intelligence machine learning data technology",
    "web application software programming computer system",
    "mobile application software technology computer",
    "cloud computing data storage network technology",
    "computer science programming algorithms data software",
]

# ----------------------------------------------------------
# STEP 2: Tokenisation
# ----------------------------------------------------------
tokenized = [re.findall(r"[a-z]+", sentence.lower()) for sentence in corpus]

# ----------------------------------------------------------
# STEP 3: Create Vocabulary
# ----------------------------------------------------------
vocab = sorted(set(word for sentence in tokenized for word in sentence))
word_to_index = {word: index for index, word in enumerate(vocab)}

# ----------------------------------------------------------
# STEP 4: Create Word-Context Vectors
# ----------------------------------------------------------
vectors = {word: [0] * len(vocab) for word in vocab}
window = 2

for sentence in tokenized:
    for i, word in enumerate(sentence):
        start = max(0, i - window)
        end = min(len(sentence), i + window + 1)
        for j in range(start, end):
            if i != j:
                context_word = sentence[j]
                vectors[word][word_to_index[context_word]] += 1

# ----------------------------------------------------------
# STEP 5: Take Input Word
# ----------------------------------------------------------
target = "computer"

if target not in vectors:
    print("Word not found in the vocabulary.")
else:
    # ------------------------------------------------------
    # STEP 6: Calculate Cosine Similarity
    # ------------------------------------------------------
    candidates = [word for word in vocab if word != target]
    candidate_vectors = [vectors[word] for word in candidates]
    scores = cosine_similarity([vectors[target]], candidate_vectors)[0]

    # ------------------------------------------------------
    # STEP 7: Rank Similar Words
    # ------------------------------------------------------
    results = list(zip(candidates, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------
    # STEP 8: Display Top 5 Results
    # ------------------------------------------------------
    print("==========================================")
    print(" SIMILAR WORD PREDICTION")
    print("==========================================")
    print("\nInput word:", target)
    print("\nTop 5 similar words:\n")
    for rank, (word, score) in enumerate(results[:5], 1):
        print(f"{rank}. {word:<15} Similarity: {score:.3f}")
