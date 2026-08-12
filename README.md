# 🎬 Movie Recommendation System

A full-stack movie recommendation web app built with **FastAPI** (backend) and **Streamlit** (frontend), powered by **TF-IDF content-based filtering** and the **TMDB API**.

---

## 🚀 Live Demo

- **Frontend (Streamlit):** *[https://movie-recommendation-system-1207.streamlit.app](https://movie-recommendation-system-1207.streamlit.app)
- **Backend API (FastAPI):** [https://movie-recommendation-system-6puz.onrender.com](https://movie-recommendation-system-6puz.onrender.com)

---

## ✨ Features

- 🔍 **Movie Search** — Keyword-based search with autocomplete suggestions powered by TMDB
- 🎯 **TF-IDF Recommendations** — Content-based filtering using movie overviews (NLP)
- 🎭 **Genre Recommendations** — Similar movies discovered via TMDB genre matching
- 🎬 **Sequel Finder** — Automatically detects and displays sequels & franchise entries
- 🏠 **Home Feed** — Browse Trending, Popular, Top Rated, Now Playing, and Upcoming movies
- 📄 **Movie Details Page** — Full details including poster, backdrop, genres, and overview
- 🌐 **Responsive Grid Layout** — Adjustable column grid for movie posters

---

## 🏗️ Architecture

```
┌──────────────────────┐        HTTP Requests        ┌──────────────────────┐
│   Streamlit Frontend │  ─────────────────────────▶  │  FastAPI Backend     │
│      (app.py)        │                              │     (main.py)        │
└──────────────────────┘                              └──────────┬───────────┘
                                                                 │
                                              ┌──────────────────┴──────────────────┐
                                              │                                     │
                                    ┌─────────▼──────────┐              ┌──────────▼─────────┐
                                    │  TMDB API          │              │  Local TF-IDF       │
                                    │  (posters, details,│              │  (movies_metadata   │
                                    │   search, genres)  │              │   .csv + sklearn)   │
                                    └────────────────────┘              └────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer      | Technology                            |
|------------|---------------------------------------|
| Frontend   | Streamlit                             |
| Backend    | FastAPI + Uvicorn                     |
| ML/NLP     | scikit-learn (TF-IDF), pandas, numpy  |
| Data       | TMDB API, movies_metadata.csv         |
| Deployment | Render                                |

---

## 📁 Project Structure

```
movie-recommendation-system/
├── app.py                              # Streamlit frontend
├── main.py                             # FastAPI backend
├── movies_metadata.csv                 # Movie dataset (TMDB-based)
├── df.pkl                              # Preprocessed DataFrame (pickle)
├── indices.pkl                         # Title → index mapping (pickle)
├── requirements.txt                    # Python dependencies
├── render.yaml                         # Render deployment config
├── movie_reccomendation_system_executed.ipynb  # Jupyter notebook (EDA + model)
├── .env                                # API keys (not tracked in git)
└── .gitignore
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/karan1207-cse/Movie-recommendation-system.git
cd Movie-recommendation-system
```

### 2. Create & activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

> Get your free API key at [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)

### 5. Run the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Run the Streamlit frontend

In a separate terminal:

```bash
streamlit run app.py
```

App available at: [http://localhost:8501](http://localhost:8501)

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| GET    | `/health`             | Health check                                     |
| GET    | `/home`               | Home feed (trending, popular, top_rated, etc.)   |
| GET    | `/tmdb/search`        | Keyword search — returns TMDB results            |
| GET    | `/movie/id/{tmdb_id}` | Full movie details by TMDB ID                    |
| GET    | `/movie/search`       | Bundle: details + TF-IDF recs + genre recs       |
| GET    | `/recommend/genre`    | Genre-based recommendations by TMDB ID           |
| GET    | `/recommend/tfidf`    | TF-IDF recommendations by movie title            |

---

## 🧠 How the Recommendation Engine Works

1. **Data Loading** — On startup, `movies_metadata.csv` is loaded and cleaned.
2. **TF-IDF Vectorization** — Movie overviews are vectorized using `TfidfVectorizer` (sklearn), removing English stop words.
3. **Cosine Similarity** — For a given movie, cosine similarity is computed across the TF-IDF matrix to find the most content-similar movies.
4. **TMDB Enrichment** — Recommended titles are matched to TMDB entries to fetch posters and metadata.
5. **Genre Discovery** — A secondary TMDB Discover call finds popular movies in the same genre.

---

## 🚀 Deployment

The backend is deployed on **Render** using the configuration in [`render.yaml`](render.yaml):

```yaml
services:
  - type: web
    name: movie-recommendation-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 📦 Dependencies

```
fastapi==0.111.0
uvicorn==0.30.1
python-dotenv==1.0.1
httpx==0.27.0
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.4.2
streamlit==1.36.0
requests==2.34.2
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👨‍💻 Author

**Karan** — [@karan1207-cse](https://github.com/karan1207-cse)
