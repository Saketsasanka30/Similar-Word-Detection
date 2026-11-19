# SEMANTIQ — Intelligent Similar Word Detection System

SEMANTIQ is a full-stack Natural Language Processing (NLP) web application that trains **Word-Context Co-occurrence Vectors** (`window = 2`) on an uploaded text dataset or built-in corpus, predicts the most semantically similar words for a target input word using **Cosine Similarity**, and visualizes the resulting vector angles in 2D space.

---

## Features

- **Custom Text Dataset Upload & Built-In Corpus**:
  - Upload `.txt`, `.csv`, `.json`, or `.md` files (up to 500 lines) via drag-and-drop or file picker, or use the built-in 18-sentence technology training corpus.
- **8-Step Word-Context Vector Model**:
  1. **Corpus Creation**: Loads either the uploaded dataset or the default 18-sentence tech corpus.
  2. **Tokenisation**: Converts sentences to lowercase and extracts alphabetic tokens using `[a-z]+`.
  3. **Vocabulary Building**: Creates a sorted vocabulary of unique tokens.
  4. **Word-Context Vectors (`window = 2`)**: Counts co-occurring context words within 2 positions left and right of each target word.
  5. **Target Word Input**: Accepts any target word (e.g., `computer`) and selects Top 5 or Top 10 results.
  6. **Cosine Similarity**: Computes `cos(A, B) = (A · B) / (||A|| × ||B||)` against every other word in the vocabulary.
  7. **Ranking**: Sorts words in descending order of cosine similarity.
  8. **Prediction Output & CSV Export**: Displays ranked results (`1. code — Similarity: 0.805`, etc.) with one-click CSV download.
- **2D Vector Geometry Visualization**:
  - Interactive coordinate-plane graph plotting Target Vector **A**, Predicted Word Vector **B**, lowest-similarity Contrast Vector **C**, and separation angle **θ** (`θ = acos(cosine)`).
  - Click any predicted word chip above the diagram to plot and inspect its angle relative to the target word.

---

## Example Output (Input: `computer`)

```text
SIMILAR WORD PREDICTION
Input word: computer
Top 5 similar words:
1. code        — Similarity: 0.805
2. system      — Similarity: 0.739
3. application — Similarity: 0.736
4. programming — Similarity: 0.680
5. data        — Similarity: 0.664
```

---

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Motion, Lucide React
- **Backend**: Node.js, Express (`server.ts`), TypeScript (`tsx`)
- **Python Reference Script**: `backend/similar_word_prediction.py` (`re` + `scikit-learn` `cosine_similarity`)

---

## Project Structure

```text
├── backend/
│   └── similar_word_prediction.py   # Standalone Python reference script
├── server/
│   └── nlp/
│       └── contextCorpusModel.ts    # Word-context vector training & cosine similarity engine
├── src/
│   ├── components/
│   │   ├── DatasetAnalyzer.tsx      # Corpus upload, prediction results, & CSV export UI
│   │   ├── HeroSection.tsx          # Hero banner & interactive constellation
│   │   ├── LiquidEther.jsx          # WebGL background canvas
│   │   ├── Navbar.tsx               # Top navigation bar
│   │   └── VectorAngleDiagram.tsx   # 2D cosine vector & angle (θ) visualizer
│   ├── types/
│   │   └── nlp.ts                   # Shared TypeScript interfaces
│   └── App.tsx                      # Root application layout
├── server.ts                        # Express + Vite dev/production server (port 3000)
└── package.json
```

---

## API Reference

### `POST /api/corpus/predict`

Trains the Word-Context Co-occurrence Vector model (`window = 2`) on the supplied corpus (or defaults to the 18-sentence tech corpus) and returns the Top-K most similar words plus 2D vector geometry coordinates.

**Request Body**

```json
{
  "word": "computer",
  "top_k": 5,
  "window": 2,
  "corpus": [
    "computer technology software system data network"
  ]
}
```

**Response (`200 OK`)**

```json
{
  "found": true,
  "target": "computer",
  "topK": 5,
  "vocabularySize": 29,
  "corpusSentenceCount": 18,
  "windowSize": 2,
  "results": [
    { "rank": 1, "word": "code", "similarity": 0.805, "scorePercent": 80.5 },
    { "rank": 2, "word": "system", "similarity": 0.739, "scorePercent": 73.9 },
    { "rank": 3, "word": "application", "similarity": 0.736, "scorePercent": 73.6 },
    { "rank": 4, "word": "programming", "similarity": 0.68, "scorePercent": 68 },
    { "rank": 5, "word": "data", "similarity": 0.664, "scorePercent": 66.4 }
  ],
  "vectorGeometry": {
    "primaryLabel": "computer",
    "secondaryLabel": "code",
    "contrastLabel": "computing",
    "cosineSimilarity": 0.805,
    "thetaDegrees": 36.4
  }
}
```

---

## Running Locally

### Prerequisites

- Node.js 20+ and npm

### Start the Full-Stack App

```bash
npm install
npm run dev
```

The app and API run on `http://localhost:3000`.

### Build & Typecheck

```bash
npm run lint
npm run build
```

### Run Standalone Python Script (Optional)

```bash
pip install scikit-learn
python3 backend/similar_word_prediction.py
```
