import os
import re
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recomendador de Cursos Udemy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(135deg, #a435f0 0%, #6c12c7 100%);
            padding: 2rem 2rem 1.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
        .main-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1rem; }

        .course-card {
            background: #1e1e2e;
            border: 1px solid #2e2e42;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.8rem;
            transition: border-color 0.2s;
        }
        .course-card:hover { border-color: #a435f0; }
        .course-title { font-size: 1.05rem; font-weight: 600; color: #e0d7f8; }
        .course-meta  { font-size: 0.82rem; color: #9999bb; margin-top: 0.35rem; }

        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 5px;
        }
        .badge-subject  { background: #2a1550; color: #c084fc; border: 1px solid #7c3aed; }
        .badge-level    { background: #0f2a1e; color: #4ade80; border: 1px solid #16a34a; }
        .badge-free     { background: #1a2e0f; color: #86efac; border: 1px solid #22c55e; }
        .badge-paid     { background: #2e1a0f; color: #fdba74; border: 1px solid #f97316; }

        .rating-stars   { color: #facc15; font-size: 0.9rem; }
        .metric-card {
            background: #1e1e2e;
            border: 1px solid #2e2e42;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: #c084fc; }
        .metric-label { font-size: 0.8rem; color: #9999bb; margin-top: 0.2rem; }

        .login-box {
            max-width: 460px;
            margin: 4rem auto;
            background: #1e1e2e;
            border: 1px solid #2e2e42;
            border-radius: 14px;
            padding: 2.5rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #a435f0, #6c12c7);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover { opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT    UNIQUE NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT NOT NULL,
                keywords   TEXT NOT NULL,
                response   TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_feedback(email: str, keywords: str, response: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (email, keywords, response) VALUES (?, ?, ?)",
            (email, keywords, response),
        )
        conn.commit()


def register_email(email: str) -> tuple[bool, str]:
    """
    Insert the email.  Returns (is_new_user, message).
    Never stores passwords — only the e-mail address.
    """
    try:
        with _get_conn() as conn:
            conn.execute("INSERT INTO users (email) VALUES (?)", (email.strip().lower(),))
            conn.commit()
        return True, "✅ Cadastro realizado com sucesso! Bem-vindo."
    except sqlite3.IntegrityError:
        return False, "👋 E-mail já cadastrado. Bem-vindo de volta!"


def is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


# ---------------------------------------------------------------------------
# Data loading & model building (cached)
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "udemy_cursos_completos.csv")

EXTRA_STOP = {
    "learn", "learned", "how", "you", "your",
    "course", "complete", "guide", "part", "beginner", "level",
    "introduction", "intro", "basics", "tutorial", "bootcamp",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))

SUBJECT_LABELS = {
    "Business Finance": "Business Finance",
    "Graphic Design": "Graphic Design",
    "Musical Instruments": "Musical Instruments",
    "Subject: Web Development": "Web Development",
}

LEVEL_ORDER = ["All Levels", "Beginner Level", "Intermediate Level", "Expert Level"]


@st.cache_data(show_spinner="Carregando base de dados…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df[df["num_subscribers"] > 0].copy()
    df = df[df["course_title"].str.split().str.len() > 3]
    df = df[df["course_title"].apply(lambda t: bool(re.match(r"^[\x00-\x7F]+$", str(t))))]

    # Normalise subject label
    df["subject_display"] = df["subject"].map(SUBJECT_LABELS).fillna(df["subject"])

    # TF-IDF content field  (title repeated for weight, + subject + level)
    df["conteudo"] = (
        df["course_title"].str.lower() + " "
        + df["course_title"].str.lower() + " "
        + df["subject_display"].str.lower() + " "
        + df["level"].str.lower()
    )
    df["course_title_clean"] = df["course_title"].str.lower()
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="Construindo modelo de recomendação…")
def build_model(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        max_df=0.8,
        min_df=1,
        ngram_range=(1, 2),
        max_features=6000,
    )
    tfidf_matrix = vectorizer.fit_transform(df["conteudo"])
    return vectorizer, tfidf_matrix


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def recommend(
    keywords: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 10,
) -> pd.DataFrame:
    """Content-based filtering using TF-IDF + cosine similarity."""

    if not keywords.strip():
        return pd.DataFrame()

    # Repeat keywords to give them strong weight in the query
    query_vec = vectorizer.transform([" ".join([keywords.lower().strip()] * 5)])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    result = df.copy()
    result["_sim"] = sim_scores

    # Only keep courses with a non-zero similarity to the query
    result = result[result["_sim"] > 0]
    result = result.drop_duplicates(subset="course_title")

    if result.empty:
        return result

    # Score = 75% similarity (relevance first) + 15% popularity + 10% rating
    max_subs = result["num_subscribers"].max() or 1
    result["_score"] = (
        result["_sim"] * 0.75
        + (result["num_subscribers"] / max_subs) * 0.15
        + result["Rating"] * 0.10
    )
    result = result.sort_values("_score", ascending=False)

    return result.head(top_n)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def star_html(rating_01: float) -> str:
    """Convert 0-1 rating to ★ display (scaled to 5)."""
    score = rating_01 * 5
    full = int(score)
    half = 1 if (score - full) >= 0.4 else 0
    empty = 5 - full - half
    return (
        "★" * full
        + ("½" if half else "")
        + "☆" * empty
        + f" <span style='color:#ccc;font-size:0.78rem;'>{score:.1f}/5</span>"
    )


def render_course_card(rank: int, row: pd.Series) -> None:
    price_badge = (
        '<span class="badge badge-free">Gratuito</span>'
        if row["price"] == 0
        else f'<span class="badge badge-paid">$ {row["price"]:.2f}</span>'
    )
    subject_badge = f'<span class="badge badge-subject">{row["subject_display"]}</span>'
    level_badge   = f'<span class="badge badge-level">{row["level"]}</span>'

    subs  = f'{int(row["num_subscribers"]):,}'
    duration = (
        f'{row["content_duration"]:.1f}h' if pd.notna(row.get("content_duration")) else "—"
    )

    url = row.get("url", "#") or "#"

    st.markdown(
        f"""
        <div class="course-card">
            <div class="course-title">#{rank} &nbsp;
                <a href="{url}" target="_blank" style="color:#c084fc;text-decoration:none;">
                    {row['course_title'].title()}
                </a>
            </div>
            <div style="margin:0.5rem 0;">
                {subject_badge}{level_badge}{price_badge}
            </div>
            <div class="course-meta">
                👥 {subs} alunos
                &nbsp;&nbsp;⏱ {duration}
                &nbsp;&nbsp;📝 {int(row.get('num_lectures', 0)) if pd.notna(row.get('num_lectures')) else '—'} aulas
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    init_db()
    df = load_data()
    vectorizer, tfidf_matrix = build_model(df)

    # --- Session state ---
    if "registered" not in st.session_state:
        st.session_state.registered = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    if "last_keywords" not in st.session_state:
        st.session_state.last_keywords = ""
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    # ===================================================================
    # REGISTRATION GATE
    # ===================================================================
    if not st.session_state.registered:
        st.markdown(
            """
            <div style="text-align:center;margin-top:2rem;">
                <h1 style="font-size:2.5rem;">🎓 Recomendador de Cursos Udemy</h1>
                <p style="color:#9999bb;font-size:1.05rem;">
                    Descubra os melhores cursos com base nas suas preferências.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_l, col_center, col_r = st.columns([1, 1.4, 1])
        with col_center:
            st.markdown('<div class="login-box">', unsafe_allow_html=True)
            st.markdown("#### Informe seu e-mail para começar")
            st.markdown(
                "<p style='color:#9999bb;font-size:0.87rem;margin-top:-0.4rem;'>"
                "Seu e-mail é usado apenas para identificação. "
                "Nenhum dado é compartilhado com terceiros.</p>",
                unsafe_allow_html=True,
            )
            email_input = st.text_input(
                "E-mail",
                placeholder="voce@exemplo.com",
                label_visibility="collapsed",
                key="email_field",
            )
            if st.button("Entrar →", type="primary", use_container_width=True):
                if not email_input.strip():
                    st.error("Por favor, insira seu e-mail.")
                elif not is_valid_email(email_input):
                    st.error("E-mail inválido. Verifique e tente novamente.")
                else:
                    is_new, msg = register_email(email_input)
                    st.session_state.registered = True
                    st.session_state.user_email = email_input.strip().lower()
                    st.session_state.welcome_msg = msg
                    st.session_state.is_new_user = is_new
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Dataset quick stats below login
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for col, val, label in [
            (c1, f"{len(df):,}", "Cursos disponíveis"),
            (c2, f"{df['num_subscribers'].sum() / 1_000_000:.1f}M", "Matrículas totais"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        return  # stop here until user logs in

    # ===================================================================
    # MAIN APP
    # ===================================================================

    # Welcome toast (shown once)
    if st.session_state.get("welcome_msg"):
        if st.session_state.get("is_new_user"):
            st.success(st.session_state.welcome_msg)
        else:
            st.info(st.session_state.welcome_msg)
        st.session_state.welcome_msg = None

    # Header
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🎓 Recomendador de Cursos Udemy</h1>
            <p>Olá, <strong>{st.session_state.user_email}</strong>!
               Encontre os melhores cursos para você.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Logout button (top-right)
    col_space, col_logout = st.columns([5, 1])
    with col_logout:
        if st.button("Sair"):
            st.session_state.registered = False
            st.session_state.user_email = ""
            st.rerun()

    # ---- Search Form -------------------------------------------------------
    st.markdown("### 🔍 O que você quer aprender?")
    with st.form("search_form"):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            keywords = st.text_input(
                "Tópico ou palavras-chave",
                placeholder="Ex: python, machine learning, guitarra, web design…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("🚀 Buscar", type="primary", use_container_width=True)

    # ---- Results -----------------------------------------------------------
    if submitted:
        if not keywords.strip():
            st.warning("Digite ao menos uma palavra-chave para buscar.")
        else:
            st.markdown("---")
            with st.spinner("Buscando recomendações…"):
                results = recommend(keywords, df, vectorizer, tfidf_matrix, top_n=10)

            if results.empty:
                st.warning(
                    "Nenhum curso encontrado para essa busca. "
                    "Tente palavras-chave diferentes."
                )
            else:
                st.markdown(f"#### 📚 10 recomendações para **\"{keywords.strip()}\"**")
                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    render_course_card(rank, row)

                # Persist results in session state for the feedback step
                st.session_state.last_results = results
                st.session_state.last_keywords = keywords.strip()
                st.session_state.feedback_given = False

    # ---- Feedback step (shown after a successful search) -------------------
    if (
        st.session_state.last_results is not None
        and not st.session_state.last_results.empty
        and not submitted  # don't overlap with a new search in the same run
    ):
        if not st.session_state.feedback_given:
            st.markdown("---")
            st.markdown("#### 💬 As recomendações fizeram sentido?")
            st.markdown(
                "<p style='color:#9999bb;font-size:0.9rem;margin-top:-0.4rem;'>"
                "Sua resposta nos ajuda a melhorar o sistema.</p>",
                unsafe_allow_html=True,
            )
            with st.form("feedback_form"):
                resposta = st.radio(
                    "Avaliação",
                    options=["✅ Sim, fizeram sentido", "🔶 Parcialmente", "❌ Não fizeram sentido"],
                    label_visibility="collapsed",
                    horizontal=True,
                )
                enviado = st.form_submit_button("Enviar avaliação", type="primary")

            if enviado:
                save_feedback(
                    st.session_state.user_email,
                    st.session_state.last_keywords,
                    resposta,
                )
                st.session_state.feedback_given = True
                st.rerun()
        else:
            st.markdown("---")
            st.success("Obrigado pelo feedback! 🙏")

    elif not submitted and st.session_state.last_results is None:
        st.markdown(
            "<br><p style='text-align:center;color:#9999bb;'>"
            "Digite um tópico acima e clique em <strong>Buscar</strong> para ver sugestões.</p>",
            unsafe_allow_html=True,
        )

        # Teaser: top popular courses
        st.markdown("#### ⭐ Cursos mais populares da plataforma")
        top_popular = (
            df.sort_values(["num_subscribers", "Rating"], ascending=False)
            .drop_duplicates("course_title")
            .head(6)
        )
        for rank, (_, row) in enumerate(top_popular.iterrows(), start=1):
            render_course_card(rank, row)


if __name__ == "__main__":
    main()
