import os
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# =============================
# CONFIG
# =============================
API_BASE = "https://movie-recommendation-system-6puz.onrender.com"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
DEFAULT_TMDB_KEY = "c93c1394ade63533cd412d87433eda9"


def get_tmdb_key():
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY") or DEFAULT_TMDB_KEY


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES (minimal modern)
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# DIRECT ENGINE FALLBACK (Self-contained)
# =============================
@st.cache_resource
def load_local_engine():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "movies_metadata.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    
    df_clean = df[["id", "title", "overview", "poster_path", "release_date", "vote_average", "vote_count"]].dropna(subset=["title", "overview"]).copy()
    df_clean["title_str"] = df_clean["title"].astype(str)
    df_clean["vote_count_num"] = pd.to_numeric(df_clean["vote_count"], errors="coerce").fillna(0)
    
    tfidf_obj = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_obj.fit_transform(df_clean["overview"])
    
    indices = pd.Series(df_clean.index, index=df_clean["title_str"]).drop_duplicates()
    title_to_idx = {str(k).strip().lower(): int(v) for k, v in indices.items()}
    
    return df_clean, tfidf_matrix, title_to_idx


def tmdb_direct_get(endpoint: str, params: dict | None = None):
    p = dict(params or {})
    key = get_tmdb_key()
    p["api_key"] = key
    headers = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.get(f"{TMDB_BASE}{endpoint}", params=p, headers=headers, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def direct_home(category: str = "popular", limit: int = 24):
    endpoint = "/trending/movie/day" if category == "trending" else f"/movie/{category}"
    data = tmdb_direct_get(endpoint, {"language": "en-US", "page": 1})
    if data and "results" in data:
        cards = []
        for m in data["results"][:limit]:
            cards.append({
                "tmdb_id": int(m["id"]),
                "title": m.get("title") or m.get("name") or "Untitled",
                "poster_url": f"{TMDB_IMG}{m.get('poster_path')}" if m.get("poster_path") else None,
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average"),
            })
        if cards:
            return cards
    
    # Offline CSV fallback sorted by popularity
    df_clean, _, _ = load_local_engine()
    top_movies = df_clean.sort_values(by="vote_count_num", ascending=False).head(limit)
    cards = []
    for _, row in top_movies.iterrows():
        try:
            tmdb_id = int(row["id"])
        except Exception:
            tmdb_id = 0
        p_path = str(row["poster_path"]).strip() if pd.notna(row["poster_path"]) else ""
        cards.append({
            "tmdb_id": tmdb_id,
            "title": str(row["title"]),
            "poster_url": f"{TMDB_IMG}{p_path}" if p_path and p_path.startswith("/") else None,
            "release_date": str(row["release_date"]) if pd.notna(row["release_date"]) else "",
            "vote_average": float(row["vote_average"]) if pd.notna(row["vote_average"]) else None,
        })
    return cards


def direct_tmdb_search(query: str, page: int = 1):
    data = tmdb_direct_get("/search/movie", {"query": query, "include_adult": "false", "language": "en-US", "page": page})
    if data and "results" in data:
        return data
    
    df_clean, _, _ = load_local_engine()
    matched = df_clean[df_clean["title_str"].str.contains(query, case=False, na=False)].head(20)
    results = []
    for _, row in matched.iterrows():
        try:
            t_id = int(row["id"])
        except Exception:
            t_id = 0
        results.append({
            "id": t_id,
            "title": str(row["title"]),
            "poster_path": str(row["poster_path"]) if pd.notna(row["poster_path"]) else None,
            "release_date": str(row["release_date"]),
            "overview": str(row["overview"]),
        })
    return {"results": results}


def direct_movie_details(tmdb_id: int):
    data = tmdb_direct_get(f"/movie/{tmdb_id}", {"language": "en-US"})
    if data:
        return {
            "tmdb_id": int(data["id"]),
            "title": data.get("title") or "",
            "overview": data.get("overview"),
            "release_date": data.get("release_date"),
            "poster_url": f"{TMDB_IMG}{data.get('poster_path')}" if data.get("poster_path") else None,
            "backdrop_url": f"{TMDB_IMG}{data.get('backdrop_path')}" if data.get("backdrop_path") else None,
            "genres": data.get("genres", []) or [],
        }
    
    df_clean, _, _ = load_local_engine()
    row = df_clean[df_clean["id"] == str(tmdb_id)]
    if not row.empty:
        r = row.iloc[0]
        return {
            "tmdb_id": tmdb_id,
            "title": str(r["title"]),
            "overview": str(r["overview"]),
            "release_date": str(r["release_date"]),
            "poster_url": f"{TMDB_IMG}{r['poster_path']}" if pd.notna(r["poster_path"]) else None,
            "backdrop_url": None,
            "genres": [],
        }
    return {"tmdb_id": tmdb_id, "title": "Movie Details", "overview": "No overview available.", "genres": []}


def direct_tfidf_recommend(title: str, top_n: int = 12):
    df_clean, matrix, title_to_idx = load_local_engine()
    key = str(title).strip().lower()
    idx = title_to_idx.get(key)
    
    if idx is None:
        matches = [k for k in title_to_idx.keys() if key in k]
        if matches:
            idx = title_to_idx[matches[0]]
            
    if idx is None:
        return []
        
    qv = matrix[idx]
    scores = (matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    
    recs = []
    for i in order:
        if int(i) == int(idx):
            continue
        row = df_clean.iloc[int(i)]
        t = str(row["title"])
        s = float(scores[int(i)])
        p_path = row["poster_path"] if pd.notna(row["poster_path"]) else None
        
        try:
            t_id = int(row["id"])
        except Exception:
            t_id = 0
            
        recs.append({
            "title": t,
            "score": s,
            "tmdb": {
                "tmdb_id": t_id,
                "title": t,
                "poster_url": f"{TMDB_IMG}{p_path}" if p_path else None
            }
        })
        if len(recs) >= top_n:
            break
    return recs


def direct_genre_recommend(tmdb_id: int, limit: int = 12):
    details = direct_movie_details(tmdb_id)
    genres = details.get("genres", [])
    if genres:
        g_id = genres[0]["id"]
        discover = tmdb_direct_get("/discover/movie", {
            "with_genres": g_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1
        })
        if discover and "results" in discover:
            cards = []
            for m in discover["results"][:limit]:
                if int(m["id"]) != tmdb_id:
                    cards.append({
                        "tmdb_id": int(m["id"]),
                        "title": m.get("title") or "",
                        "poster_url": f"{TMDB_IMG}{m.get('poster_path')}" if m.get("poster_path") else None
                    })
            return cards
    return []


def direct_search_bundle(query: str, tfidf_top_n: int = 12, genre_limit: int = 12):
    search_res = direct_tmdb_search(query)
    results = search_res.get("results", [])
    if not results:
        return None
    best = results[0]
    tmdb_id = int(best["id"])
    details = direct_movie_details(tmdb_id)
    
    tfidf_items = direct_tfidf_recommend(details["title"], top_n=tfidf_top_n)
    genre_items = direct_genre_recommend(tmdb_id, limit=genre_limit)
    
    return {
        "query": query,
        "movie_details": details,
        "tfidf_recommendations": tfidf_items,
        "genre_recommendations": genre_items
    }

# =============================
# STATE + ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except Exception:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS (With Seamless Direct Fallback)
# =============================
@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    if API_BASE:
        try:
            r = requests.get(f"{API_BASE}{path}", params=params, timeout=4)
            if r.status_code == 200 and "application/json" in r.headers.get("content-type", ""):
                return r.json(), None
        except Exception:
            pass

    # Direct Execution Fallback
    params = params or {}
    try:
        if path == "/home":
            cat = params.get("category", "popular")
            lim = int(params.get("limit", 24))
            return direct_home(cat, lim), None
        elif path == "/tmdb/search":
            q = params.get("query", "")
            p = int(params.get("page", 1))
            return direct_tmdb_search(q, p), None
        elif path.startswith("/movie/id/"):
            tmdb_id = int(path.split("/")[-1])
            return direct_movie_details(tmdb_id), None
        elif path == "/movie/search":
            q = params.get("query", "")
            t_n = int(params.get("tfidf_top_n", 12))
            g_l = int(params.get("genre_limit", 12))
            res = direct_search_bundle(q, t_n, g_l)
            if res:
                return res, None
            return None, "No movie found"
        elif path == "/recommend/genre":
            t_id = int(params.get("tmdb_id", 0))
            lim = int(params.get("limit", 12))
            return direct_genre_recommend(t_id, lim), None
    except Exception as e:
        return None, f"Direct engine error: {e}"

    return None, "Endpoint not found"


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")
            
            is_valid_poster = (
                poster 
                and isinstance(poster, str) 
                and poster.startswith("http") 
                and not poster.endswith("None") 
                and not poster.endswith("nan")
            )

            with colset[c]:
                if is_valid_poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.markdown(
                        f"""
                        <div style="height:210px; background:#1e293b; border-radius:10px; display:flex; align-items:center; justify-content:center; text-align:center; padding:12px; color:#f8fafc; font-weight:600; font-size:0.85rem; margin-bottom:8px; border:1px solid #334155;">
                            🎬<br><br>{title}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True
                )


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


def search_sequels(base_title: str, limit: int = 6) -> list:
    sequels = []
    patterns = [
        f"{base_title} 2",
        f"{base_title} II",
        f"{base_title} Part 2",
        f"{base_title}: Part 2",
    ]

    for pattern in patterns:
        data, err = api_get_json("/tmdb/search", params={"query": pattern.strip(), "page": 1})
        if not err and data and "results" in data:
            for result in data.get("results", []):
                if result.get("id") and result.get("title"):
                    sequels.append({
                        "tmdb_id": int(result["id"]),
                        "title": result.get("title", ""),
                        "poster_url": f"{TMDB_IMG}{result.get('poster_path')}" if result.get("poster_path") else None,
                    })
        if sequels:
            break

    seen = set()
    unique_sequels = []
    for s in sequels:
        if s["tmdb_id"] not in seen:
            seen.add(s["tmdb_id"])
            unique_sequels.append(s)

    return unique_sequels[:limit]


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path and not str(poster_path).startswith("http") else poster_path,
                    "release_date": m.get("release_date", ""),
                }
            )

    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            poster_url = m.get("poster_url")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed (only home)")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Type keyword → dropdown suggestions + matching results → open → details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type: avenger, batman, love..."
    )

    st.divider()

    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)

                    if selected != "-- Select a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    st.markdown(f"### 🏠 Home — {home_category.replace('_',' ').title()}")

    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_column_width=True)
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        release = data.get("release_date") or "-"
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], use_column_width=True)

    st.divider()
    st.markdown("### ✅ Recommendations")

    title = (data.get("title") or "").strip()
    if title:
        st.markdown("#### 🎬 Sequels & Franchises")
        sequels = search_sequels(title, limit=6)
        if sequels:
            poster_grid(sequels, cols=grid_cols, key_prefix="sequels")
        else:
            st.caption("No sequels found for this movie.")

        st.divider()

        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=grid_cols, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")