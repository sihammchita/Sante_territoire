import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
import warnings
import os

warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Santé & Territoires",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary: #1e3a5f;
        --accent: #2ecc71;
        --warning: #e74c3c;
        --neutral: #f0f4f8;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #2980b9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .kpi-card.danger { border-left-color: #e74c3c; }
    .kpi-card.warning { border-left-color: #f39c12; }
    .kpi-card.success { border-left-color: #27ae60; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .kpi-label { font-size: 0.8rem; color: #7f8c8d; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2980b9; padding-bottom: 6px;
        margin: 1.2rem 0 1rem;
    }
    .alert-box {
        border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    .alert-critical { background: #fdecea; border-left: 4px solid #e74c3c; color: #c0392b; }
    .alert-warning  { background: #fef9e7; border-left: 4px solid #f39c12; color: #d68910; }
    .alert-ok       { background: #eafaf1; border-left: 4px solid #27ae60; color: #1e8449; }
    [data-testid="stSidebar"] { background: #1e3a5f; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; }
    .stTab [data-baseweb="tab"] { font-weight: 600; }
    .score-badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 700;
    }
    .badge-red { background: #fdecea; color: #c0392b; }
    .badge-orange { background: #fef9e7; color: #d68910; }
    .badge-green { background: #eafaf1; color: #1e8449; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def load_all_data():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") + "/"

    def norm_dept(s):
        s = str(s).strip().zfill(2)
        return s

    # Population
    pop = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=11rOLt12iXUxbEQTRlZlbuil_AEp2jxue",
    sep=";")    
    pop.columns = [c.replace('\r\n', ' ').strip() for c in pop.columns]
    pop["dept"] = pop["code_departement"].apply(norm_dept)
    for col in ["Population municipale", "Densité de population (en km²)",
                "Part des moins  de 25 ans (en %)", "Part des  25 à 64 ans (en %)", "Part des  plus de 65 ans (en %)"]:
        if col in pop.columns:
            pop[col] = pop[col].astype(str).str.replace(" ", "").str.replace(",", ".").replace("nan", np.nan)
            try:
                pop[col] = pd.to_numeric(pop[col], errors="coerce")
            except:
                pass
    col_map = {}
    for c in pop.columns:
        if "25 ans" in c:   col_map[c] = "pct_moins_25"
        if "25 à 64" in c:  col_map[c] = "pct_25_64"
        if "65 ans" in c:   col_map[c] = "pct_plus_65"
        if "Population" in c: col_map[c] = "population"
        if "Densité" in c:    col_map[c] = "densite"
    pop = pop.rename(columns=col_map)

    # Professionnels santé
    pros = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1_wkO1vtWE2WO9aZmiI8lNPdbecO5V3pA",
    sep=";",
    low_memory=False)
    pros["dept"] = pros["code_departement"].apply(norm_dept)
    pros_dept = pros.groupby("dept").agg(
        nb_pros=("specialite_libelle", "count"),
        nb_med_gen=("specialite_libelle", lambda x: (x == "Médecin généraliste").sum()),
        nb_specialistes=("specialite_libelle", lambda x: x.isin(["Cardiologue","Pédiatre","Psychiatre","Gynécologue médical","Ophtalmologue"]).sum()),
        nb_infirmiers=("specialite_libelle", lambda x: (x == "Infirmier").sum()),
        nb_pharmaciens=("specialite_libelle", lambda x: (x == "Pharmacien").sum()),
    ).reset_index()

    # Établissements
    etabs = pd.read_csv("https://drive.google.com/uc?export=download&id=1hZ71udkcpyquNPgGowvSxUrjrmK-n-PC", sep=";")
    etabs["dept"] = etabs["code_departement"].apply(norm_dept)
    etabs_dept = etabs.groupby("dept").agg(
        nb_etabs=("Rslongue", "count"),
        nb_hopitaux=("categetab", lambda x: x.isin(["Centre Hospitalier (C.H.)","Centre Hospitalier Régional (C.H.R.)"]).sum()),
        nb_cliniques=("categetab", lambda x: x.str.contains("Clinique|privé", na=False, case=False).sum()),
    ).reset_index()

    # Temps d'accès
    temps = pd.read_csv("https://drive.google.com/uc?export=download&id=1BoP_S7BYOvDKpwOhpFSEscTM31ltPEEa", sep=";")
    temps["dept"] = temps["code_departement"].apply(norm_dept)
    temps_dept = temps.groupby("dept").agg(
        temps_acces_moyen=("temps_acces", "mean"),
        temps_acces_max=("temps_acces", "max"),
        nb_communes=("commune", "count"),
        nb_communes_critiques=("temps_acces", lambda x: (x > 15).sum()),
    ).reset_index()

    # Immobilier
    immo = pd.read_csv("https://drive.google.com/uc?export=download&id=1Psjk6nf41I_X4dnFE0kgXpCNN5is4s9n", sep=";", low_memory=False)
    immo["dept"] = immo["code_departement"].apply(norm_dept)
    immo_dept = immo.groupby("dept").agg(
        prix_m2_moyen=("prix_m2", "mean"),
        nb_transactions=("valeur_fonciere", "count"),
        surface_moy=("surface_m2", "mean"),
    ).reset_index()

    # Environnement santé (par région)
    env = pd.read_csv("https://drive.google.com/uc?export=download&id=1rfdxUJDSX5HzHStZgl5LTUPBoGF9V2i4", sep=";")
    env.columns = ["Code_region", "nom_region", "enviro_score"]
    env["enviro_score"] = env["enviro_score"].astype(str).str.replace(",", ".").replace("nan", np.nan)
    env["enviro_score"] = pd.to_numeric(env["enviro_score"], errors="coerce")

    # Médicaments
    medic = pd.read_csv("https://drive.google.com/uc?export=download&id=193dosn8DVXFgALvoWynssmxcs-8eNM82", sep=";")
    medic_summary = medic.groupby(["Statut", "Domaine(s) médical(aux)"]).size().reset_index(name="count")

    # ─── MASTER JOIN ───────────────────────────────────────────────────────────
    # Base : population
    master = pop[["dept", "Nom du département", "Nom de la région", "Code région",
                   "population", "densite", "pct_moins_25", "pct_25_64", "pct_plus_65"]].copy()
    master = master.merge(pros_dept, on="dept", how="left")
    master = master.merge(etabs_dept, on="dept", how="left")
    master = master.merge(temps_dept, on="dept", how="left")
    master = master.merge(immo_dept, on="dept", how="left")

    # Add enviro score via region
    env["Code_region"] = env["Code_region"].astype(str)
    master["Code région"] = master["Code région"].astype(str)
    master = master.merge(env[["Code_region", "enviro_score"]], left_on="Code région", right_on="Code_region", how="left")

    # Derived indicators
    master["population_num"] = pd.to_numeric(
        master["population"].astype(str).str.replace(" ","").str.replace(",","."), errors="coerce"
    )
    master["pros_pour_100k"] = master["nb_pros"] / (master["population_num"] / 100000)
    master["med_gen_pour_100k"] = master["nb_med_gen"] / (master["population_num"] / 100000)
    master["hopitaux_pour_100k"] = master["nb_hopitaux"] / (master["population_num"] / 100000)

    # ─── SCORING (aide à la décision) ─────────────────────────────────────────
    # Normalise 0-100 for each indicator (higher = better)
    def norm_inv(series):   # higher raw = worse → lower score
        mn, mx = series.min(), series.max()
        return 100 - (series - mn) / (mx - mn + 1e-9) * 100
    def norm(series):       # higher raw = better → higher score
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9) * 100

    master["score_acces"]     = norm_inv(master["temps_acces_moyen"].fillna(master["temps_acces_moyen"].median()))
    master["score_pros"]      = norm(master["pros_pour_100k"].fillna(master["pros_pour_100k"].median()))
    master["score_etabs"]     = norm(master["hopitaux_pour_100k"].fillna(master["hopitaux_pour_100k"].median()))
    master["score_env"]       = norm(master["enviro_score"].fillna(master["enviro_score"].median()))
    master["score_global"]    = (
        master["score_acces"] * 0.30 +
        master["score_pros"]  * 0.30 +
        master["score_etabs"] * 0.25 +
        master["score_env"]   * 0.15
    )

    def label(score):
        if score < 33:  return "🔴 Zone critique"
        if score < 66:  return "🟡 Zone intermédiaire"
        return "🟢 Zone favorable"

    master["zone"] = master["score_global"].apply(label)
    master["zone_short"] = master["score_global"].apply(
        lambda s: "Critique" if s < 33 else ("Intermédiaire" if s < 66 else "Favorable")
    )

    return master, medic, pros, immo, etabs, temps, env


master, medic, pros, immo, etabs, temps, env = load_all_data()


# ─── GEOJSON DÉPARTEMENTS ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement de la carte…")
def load_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    try:
        r = requests.get(url, timeout=15)
        geo = r.json()
        return geo
    except:
        return None

geojson = load_geojson()


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Filtres")
    st.markdown("---")

    regions_list = ["Toutes les régions"] + sorted(master["Nom de la région"].dropna().unique().tolist())
    selected_region = st.selectbox("Région", regions_list)

    if selected_region != "Toutes les régions":
        depts_list = master[master["Nom de la région"] == selected_region]["Nom du département"].dropna().tolist()
    else:
        depts_list = master["Nom du département"].dropna().tolist()

    selected_depts = st.multiselect(
        "Départements (multi-sélection)",
        options=sorted(depts_list),
        default=[]
    )

    st.markdown("---")
    st.markdown("### 🎚 Pondération du score global")
    w_acces  = st.slider("Poids Accès aux soins", 0, 100, 30, 5)
    w_pros   = st.slider("Poids Professionnels", 0, 100, 30, 5)
    w_etabs  = st.slider("Poids Établissements", 0, 100, 25, 5)
    w_env    = st.slider("Poids Environnement", 0, 100, 15, 5)
    total_w  = w_acces + w_pros + w_etabs + w_env
    if total_w > 0:
        master["score_global"] = (
            master["score_acces"] * (w_acces / total_w) +
            master["score_pros"]  * (w_pros  / total_w) +
            master["score_etabs"] * (w_etabs / total_w) +
            master["score_env"]   * (w_env   / total_w)
        )
    master["zone"] = master["score_global"].apply(
        lambda s: "🔴 Zone critique" if s < 33 else ("🟡 Zone intermédiaire" if s < 66 else "🟢 Zone favorable")
    )
    master["zone_short"] = master["score_global"].apply(
        lambda s: "Critique" if s < 33 else ("Intermédiaire" if s < 66 else "Favorable")
    )

    st.markdown("---")
    st.markdown("### 📊 Source des données")
    st.caption("• Population 2021 – INSEE\n• Pros santé – RPPS\n• Établissements – FINESS\n• Immo 2025 – DVF\n• Médicaments – ANSM\n• Enviro – Score régional")


# ─── FILTERED DATA ─────────────────────────────────────────────────────────────
df = master.copy()
if selected_region != "Toutes les régions":
    df = df[df["Nom de la région"] == selected_region]
if selected_depts:
    df = df[df["Nom du département"].isin(selected_depts)]


# ─── HEADER ────────────────────────────────────────────────────────────────────
scope_label = selected_region if selected_region != "Toutes les régions" else "France entière"
st.markdown(f"""
<div class="main-header">
  <h1>🏥 Dashboard Santé & Territoires — Aide à la Décision</h1>
  <p>Analyse croisée · {scope_label} · {len(df)} département(s) sélectionné(s) · Données 2021–2025</p>
</div>
""", unsafe_allow_html=True)


# ─── KPI ROW ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

n_crit   = int((df["zone_short"] == "Critique").sum())
n_inter  = int((df["zone_short"] == "Intermédiaire").sum())
n_fav    = int((df["zone_short"] == "Favorable").sum())
avg_acces = df["temps_acces_moyen"].mean()
avg_pros  = df["pros_pour_100k"].mean()

with k1:
    st.markdown(f"""<div class="kpi-card danger">
        <div class="kpi-value">{n_crit}</div>
        <div class="kpi-label">Zones critiques</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card warning">
        <div class="kpi-value">{n_inter}</div>
        <div class="kpi-label">Zones intermédiaires</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card success">
        <div class="kpi-value">{n_fav}</div>
        <div class="kpi-label">Zones favorables</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{avg_acces:.1f} min</div>
        <div class="kpi-label">Temps d'accès moyen</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{avg_pros:.0f}</div>
        <div class="kpi-label">Pros santé / 100k hab.</div></div>""", unsafe_allow_html=True)

st.markdown("")


# ─── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🗺️ Carte Territoriale", "📊 Analyse Comparative", "🔬 Croisement Données",
                "💊 Médicaments", "🏠 Immobilier & Santé", "🎯 Aide à la Décision"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CARTE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">🗺️ Carte des Départements — Score Santé Global</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        map_metric = st.selectbox("Indicateur cartographié", [
            "Score global santé",
            "Temps d'accès moyen (min)",
            "Professionnels pour 100k hab.",
            "Médecins généralistes pour 100k",
            "Prix immobilier moyen (€/m²)",
            "Part des +65 ans (%)",
        ])
    with c2:
        color_scale = st.selectbox("Palette", ["RdYlGn", "Blues", "Reds", "Plasma"])

    metric_map = {
        "Score global santé": ("score_global", True),
        "Temps d'accès moyen (min)": ("temps_acces_moyen", False),
        "Professionnels pour 100k hab.": ("pros_pour_100k", True),
        "Médecins généralistes pour 100k": ("med_gen_pour_100k", True),
        "Prix immobilier moyen (€/m²)": ("prix_m2_moyen", False),
        "Part des +65 ans (%)": ("pct_plus_65", False),
    }
    col_key, higher_better = metric_map[map_metric]

    if geojson:
        map_df = master.copy() if selected_region == "Toutes les régions" else df.copy()
        map_df["dept_code"] = map_df["dept"].astype(str)
        map_df["hover"] = (
            map_df["Nom du département"].fillna("") + "<br>" +
            "Zone: " + map_df["zone"] + "<br>" +
            "Score global: " + map_df["score_global"].round(1).astype(str) + "/100<br>" +
            "Temps accès: " + map_df["temps_acces_moyen"].round(1).astype(str) + " min<br>" +
            "Pros/100k: " + map_df["pros_pour_100k"].round(0).astype(str) + "<br>" +
            "Prix m²: " + map_df["prix_m2_moyen"].round(0).astype(str) + " €"
        )
        fig_map = px.choropleth(
            map_df,
            geojson=geojson,
            locations="dept_code",
            color=col_key,
            featureidkey="properties.code",
            hover_name="Nom du département",
            hover_data={"dept_code": False, col_key: ":.1f"},
            color_continuous_scale=color_scale if higher_better else color_scale + "_r",
            labels={col_key: map_metric},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            height=560, margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title=map_metric, thickness=12, len=0.7)
        )
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.warning("Carte non disponible (impossible de charger le GeoJSON). Veuillez vérifier votre connexion.")

    # Zone legend
    st.markdown("")
    l1, l2, l3 = st.columns(3)
    crit_dept  = df[df["zone_short"] == "Critique"].sort_values("score_global").head(5)
    inter_dept = df[df["zone_short"] == "Intermédiaire"].sort_values("score_global").head(5)
    fav_dept   = df[df["zone_short"] == "Favorable"].sort_values("score_global", ascending=False).head(5)

    with l1:
        st.markdown("#### 🔴 Zones les + critiques")
        for _, r in crit_dept.iterrows():
            st.markdown(f"**{r['Nom du département']}** — score {r['score_global']:.0f}/100")
    with l2:
        st.markdown("#### 🟡 Zones intermédiaires")
        for _, r in inter_dept.iterrows():
            st.markdown(f"**{r['Nom du département']}** — score {r['score_global']:.0f}/100")
    with l3:
        st.markdown("#### 🟢 Zones favorables")
        for _, r in fav_dept.iterrows():
            st.markdown(f"**{r['Nom du département']}** — score {r['score_global']:.0f}/100")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ANALYSE COMPARATIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">📊 Comparaison des Départements</div>', unsafe_allow_html=True)

    top_n = st.slider("Nombre de départements à afficher", 10, 50, 20)
    sort_by = st.radio("Trier par", ["Score global", "Temps d'accès", "Pros / 100k", "Prix immobilier"], horizontal=True)

    sort_map = {
        "Score global": ("score_global", False),
        "Temps d'accès": ("temps_acces_moyen", True),
        "Pros / 100k": ("pros_pour_100k", False),
        "Prix immobilier": ("prix_m2_moyen", True),
    }
    sort_col, asc = sort_map[sort_by]
    plot_df = df.sort_values(sort_col, ascending=asc).head(top_n)

    color_zones = {"Critique": "#e74c3c", "Intermédiaire": "#f39c12", "Favorable": "#27ae60"}

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_score = px.bar(
            plot_df.sort_values("score_global"),
            x="score_global", y="Nom du département",
            color="zone_short", color_discrete_map=color_zones,
            orientation="h",
            title="Score global santé (0–100)",
            labels={"score_global": "Score", "zone_short": "Zone"},
        )
        fig_score.add_vline(x=33, line_dash="dash", line_color="red", opacity=0.5)
        fig_score.add_vline(x=66, line_dash="dash", line_color="orange", opacity=0.5)
        fig_score.update_layout(height=500, showlegend=True, legend_title="Zone")
        st.plotly_chart(fig_score,width="stretch")

    with r1c2:
        fig_acces = px.bar(
            plot_df.sort_values("temps_acces_moyen", ascending=True),
            x="temps_acces_moyen", y="Nom du département",
            color="zone_short", color_discrete_map=color_zones,
            orientation="h",
            title="Temps d'accès moyen aux soins (min)",
            labels={"temps_acces_moyen": "Minutes", "zone_short": "Zone"},
        )
        st.plotly_chart(fig_acces, width="stretch")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_pros = px.scatter(
            df,
            x="temps_acces_moyen", y="pros_pour_100k",
            color="zone_short", color_discrete_map=color_zones,
            size="population_num", size_max=30,
            hover_name="Nom du département",
            text="dept",
            title="Accès aux soins vs Densité médicale",
            labels={"temps_acces_moyen": "Temps accès (min)", "pros_pour_100k": "Pros / 100k hab.", "zone_short": "Zone"},
        )
        fig_pros.update_traces(textposition="top center", textfont_size=7)
        fig_pros.update_layout(height=420)
        st.plotly_chart(fig_pros, width="stretch")

    with r2c2:
        radar_df = df[df["zone_short"].isin(["Critique", "Favorable"])].groupby("zone_short").agg(
            score_acces=("score_acces", "mean"),
            score_pros=("score_pros", "mean"),
            score_etabs=("score_etabs", "mean"),
            score_env=("score_env", "mean"),
        ).reset_index()

        categories = ["Accès aux soins", "Professionnels", "Établissements", "Environnement"]
        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            vals = [row["score_acces"], row["score_pros"], row["score_etabs"], row["score_env"]]
            vals += vals[:1]
            color = "#e74c3c" if row["zone_short"] == "Critique" else "#27ae60"
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself", name=row["zone_short"],
                line_color=color, fillcolor=color,
                opacity=0.35
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title="Profil : Zones Critiques vs Favorables",
            height=420,
        )
        st.plotly_chart(fig_radar, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CROISEMENT DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🔬 Croisement Multi-dimensionnel des Données</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("Axe X", ["temps_acces_moyen", "pros_pour_100k", "prix_m2_moyen", "pct_plus_65", "densite", "enviro_score", "med_gen_pour_100k"])
    with c2:
        y_axis = st.selectbox("Axe Y", ["pros_pour_100k", "score_global", "temps_acces_moyen", "pct_plus_65", "enviro_score", "prix_m2_moyen"])
    with c3:
        size_axis = st.selectbox("Taille bulles", ["population_num", "nb_etabs", "nb_hopitaux", "nb_transactions"])

    axis_labels = {
        "temps_acces_moyen": "Temps d'accès (min)",
        "pros_pour_100k": "Pros santé / 100k",
        "prix_m2_moyen": "Prix immo (€/m²)",
        "pct_plus_65": "Part +65 ans (%)",
        "densite": "Densité (hab/km²)",
        "enviro_score": "Score environnement",
        "score_global": "Score global",
        "med_gen_pour_100k": "Med. gen. / 100k",
        "population_num": "Population",
        "nb_etabs": "Nb établissements",
        "nb_hopitaux": "Nb hôpitaux",
        "nb_transactions": "Nb transactions immo",
    }

    fig_cross = px.scatter(
        df.dropna(subset=[x_axis, y_axis, size_axis]),
        x=x_axis, y=y_axis,
        size=size_axis, size_max=45,
        color="zone_short",
        color_discrete_map={"Critique": "#e74c3c", "Intermédiaire": "#f39c12", "Favorable": "#27ae60"},
        hover_name="Nom du département",
        hover_data={"dept": True, "zone_short": True, x_axis: ":.1f", y_axis: ":.1f"},
        labels={x_axis: axis_labels.get(x_axis, x_axis), y_axis: axis_labels.get(y_axis, y_axis), "zone_short": "Zone"},
        title=f"Croisement : {axis_labels.get(x_axis)} vs {axis_labels.get(y_axis)}",
    )
    fig_cross.update_layout(height=500)
    st.plotly_chart(fig_cross, width="stretch")

    st.markdown('<div class="section-title">🗂️ Tableau de bord consolidé</div>', unsafe_allow_html=True)

    display_cols = ["dept", "Nom du département", "Nom de la région", "zone",
                    "score_global", "temps_acces_moyen", "pros_pour_100k",
                    "med_gen_pour_100k", "nb_hopitaux", "prix_m2_moyen",
                    "pct_plus_65", "enviro_score"]
    display_df = df[display_cols].copy()
    display_df = display_df.rename(columns={
        "dept": "Code", "Nom du département": "Département", "Nom de la région": "Région",
        "zone": "Zone", "score_global": "Score /100",
        "temps_acces_moyen": "Temps accès (min)", "pros_pour_100k": "Pros/100k",
        "med_gen_pour_100k": "Med.Gen/100k", "nb_hopitaux": "Nb Hôpitaux",
        "prix_m2_moyen": "Prix m² (€)", "pct_plus_65": "+65 ans (%)", "enviro_score": "Enviro/20"
    })
    for num_col in ["Score /100", "Temps accès (min)", "Pros/100k", "Med.Gen/100k", "Prix m² (€)", "+65 ans (%)", "Enviro/20"]:
        if num_col in display_df.columns:
            display_df[num_col] = pd.to_numeric(display_df[num_col], errors="coerce").round(1)

    st.dataframe(
        display_df.sort_values("Score /100"),
        width="stretch",
        height=420,
    )

    csv_out = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger le tableau (CSV)", csv_out, "dashboard_sante_territoires.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MÉDICAMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">💊 Disponibilité des Médicaments</div>', unsafe_allow_html=True)

    # KPIs médicaments
    m1, m2, m3, m4 = st.columns(4)
    nb_ruptures  = int((medic["Statut"] == "Rupture de stock").sum())
    nb_tensions  = int((medic["Statut"] == "Tension d'approvisionnement").sum())
    nb_arrets    = int((medic["Statut"] == "Arrêt de commercialisation").sum())
    nb_remise    = int((medic["Statut"] == "Remise à disposition").sum())
    with m1:
        st.markdown(f"""<div class="kpi-card danger"><div class="kpi-value">{nb_ruptures}</div><div class="kpi-label">Ruptures de stock</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="kpi-card warning"><div class="kpi-value">{nb_tensions}</div><div class="kpi-label">Tensions approvisionnement</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-card danger"><div class="kpi-value">{nb_arrets}</div><div class="kpi-label">Arrêts de commercialisation</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="kpi-card success"><div class="kpi-value">{nb_remise}</div><div class="kpi-label">Remises à disposition</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    mc1, mc2 = st.columns(2)
    with mc1:
        fig_med_stat = px.pie(
            medic, names="Statut",
            color="Statut",
            color_discrete_map={
                "Rupture de stock": "#e74c3c",
                "Tension d'approvisionnement": "#f39c12",
                "Arrêt de commercialisation": "#8e44ad",
                "Remise à disposition": "#27ae60",
            },
            title="Répartition par statut",
            hole=0.4,
        )
        st.plotly_chart(fig_med_stat, width="stretch")

    with mc2:
        dom_counts = medic.groupby(["Domaine(s) médical(aux)", "Statut"]).size().reset_index(name="count")
        top_dom = medic["Domaine(s) médical(aux)"].value_counts().head(10).index
        dom_counts = dom_counts[dom_counts["Domaine(s) médical(aux)"].isin(top_dom)]
        fig_dom = px.bar(
            dom_counts,
            x="count", y="Domaine(s) médical(aux)",
            color="Statut",
            color_discrete_map={
                "Rupture de stock": "#e74c3c",
                "Tension d'approvisionnement": "#f39c12",
                "Arrêt de commercialisation": "#8e44ad",
                "Remise à disposition": "#27ae60",
            },
            orientation="h",
            title="Top 10 domaines médicaux touchés",
            labels={"count": "Nb médicaments", "Domaine(s) médical(aux)": ""},
        )
        st.plotly_chart(fig_dom, width="stretch"

    st.markdown('<div class="section-title">📋 Détail des médicaments en tension / rupture</div>', unsafe_allow_html=True)
    filter_statut = st.multiselect(
        "Filtrer par statut",
        options=medic["Statut"].unique().tolist(),
        default=["Rupture de stock", "Tension d'approvisionnement"]
    )
    filtered_medic = medic[medic["Statut"].isin(filter_statut)][
        ["Nom", "Statut", "Domaine(s) médical(aux)", "Produit(s) de santé",
         "Date de début d'incident", "Date de fin d'incident"]
    ].sort_values("Statut")
    st.dataframe(filtered_medic, width="stretch", height=350)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – IMMOBILIER & SANTÉ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🏠 Immobilier & Attractivité Territoriale</div>', unsafe_allow_html=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        fig_immo_zone = px.box(
            df.dropna(subset=["prix_m2_moyen", "zone_short"]),
            x="zone_short", y="prix_m2_moyen",
            color="zone_short",
            color_discrete_map={"Critique": "#e74c3c", "Intermédiaire": "#f39c12", "Favorable": "#27ae60"},
            title="Distribution des prix immobiliers par zone santé",
            labels={"prix_m2_moyen": "Prix moyen (€/m²)", "zone_short": "Zone santé"},
        )
        st.plotly_chart(fig_immo_zone, width="stretch")

    with ic2:
        fig_immo_acces = px.scatter(
            df.dropna(subset=["prix_m2_moyen", "temps_acces_moyen"]),
            x="temps_acces_moyen", y="prix_m2_moyen",
            color="zone_short",
            color_discrete_map={"Critique": "#e74c3c", "Intermédiaire": "#f39c12", "Favorable": "#27ae60"},
            hover_name="Nom du département",
            title="Prix immobilier vs Temps d'accès aux soins",
            labels={"prix_m2_moyen": "Prix (€/m²)", "temps_acces_moyen": "Temps accès (min)", "zone_short": "Zone"},
        )
        st.plotly_chart(fig_immo_acces,width="stretch")

    # Top 10 prix par type
    st.markdown('<div class="section-title">🏙️ Prix par département — Maisons vs Appartements</div>', unsafe_allow_html=True)
    immo_type = pd.read_csv("https://drive.google.com/uc?export=download&id=1Psjk6nf41I_X4dnFE0kgXpCNN5is4s9n", sep=";", low_memory=False)
    immo_type["dept"] = immo_type["code_departement"].astype(str).str.zfill(2)
    immo_type_dept = immo_type.groupby(["dept", "nom_departement", "type_local"])["prix_m2"].mean().reset_index()
    immo_type_dept = immo_type_dept.merge(df[["dept", "zone_short"]], on="dept", how="left")

    top_immo = immo_type_dept.groupby("dept")["prix_m2"].mean().nlargest(20).index
    plot_immo = immo_type_dept[immo_type_dept["dept"].isin(top_immo)]

    fig_immo_type = px.bar(
        plot_immo, x="nom_departement", y="prix_m2", color="type_local",
        barmode="group",
        title="Top 20 départements — Prix m² par type de bien",
        labels={"prix_m2": "Prix moyen (€/m²)", "nom_departement": "Département", "type_local": "Type"},
        color_discrete_sequence=["#2980b9", "#27ae60"],
    )
    fig_immo_type.update_xaxes(tickangle=45)
    st.plotly_chart(fig_immo_type, width="stretch")

    # Corrélation santé - immo
    st.markdown('<div class="section-title">📈 Corrélation Score Santé ↔ Marché Immobilier</div>', unsafe_allow_html=True)
    corr_cols = ["score_global", "prix_m2_moyen", "temps_acces_moyen", "pros_pour_100k",
                 "nb_hopitaux", "pct_plus_65", "enviro_score"]
    corr_df = df[corr_cols].dropna()
    corr_matrix = corr_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Matrice de corrélations",
        labels=dict(color="Corrélation"),
    )
    fig_corr.update_layout(height=420)
    st.plotly_chart(fig_corr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – AIDE À LA DÉCISION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">🎯 Aide à la Décision — Priorisation Territoriale</div>', unsafe_allow_html=True)

    st.markdown("""
    Ce module synthétise l'ensemble des indicateurs pour vous aider à **identifier les priorités d'intervention**,
    **évaluer l'attractivité d'un territoire** et **orienter vos décisions** en matière de politique de santé.
    """)

    st.markdown("#### 🔍 Sélectionnez un département à analyser en détail")
    dept_choice = st.selectbox(
        "Département",
        options=df.sort_values("score_global")["Nom du département"].dropna().tolist()
    )
    row = df[df["Nom du département"] == dept_choice].iloc[0]

    # Score badge
    score = row["score_global"]
    if score < 33:
        badge_class = "badge-red"
        verdict = "🔴 Zone Critique — Intervention prioritaire recommandée"
        alert_class = "alert-critical"
    elif score < 66:
        badge_class = "badge-orange"
        verdict = "🟡 Zone Intermédiaire — Surveillance et amélioration ciblée"
        alert_class = "alert-warning"
    else:
        badge_class = "badge-green"
        verdict = "🟢 Zone Favorable — Maintien des acquis recommandé"
        alert_class = "alert-ok"

    st.markdown(f"""
    <div class="alert-box {alert_class}">
      <strong>{dept_choice} ({row['dept']}) — {row.get('Nom de la région','')}</strong><br>
      {verdict}<br>
      Score global : <span class="score-badge {badge_class}">{score:.1f}/100</span>
    </div>
    """, unsafe_allow_html=True)

    # Detailed indicators
    st.markdown("#### 📊 Profil détaillé")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        ta = row.get("temps_acces_moyen", np.nan)
        lvl = "danger" if ta > 12 else ("warning" if ta > 7 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{ta:.1f} min</div>
            <div class="kpi-label">Temps d'accès moyen</div></div>""", unsafe_allow_html=True)
    with d2:
        pp = row.get("pros_pour_100k", np.nan)
        lvl = "danger" if pp < 200 else ("warning" if pp < 400 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{pp:.0f}</div>
            <div class="kpi-label">Pros / 100k hab.</div></div>""", unsafe_allow_html=True)
    with d3:
        pm = row.get("prix_m2_moyen", np.nan)
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{pm:.0f} €</div>
            <div class="kpi-label">Prix immo moyen /m²</div></div>""", unsafe_allow_html=True)
    with d4:
        es = row.get("enviro_score", np.nan)
        lvl = "danger" if es < 7 else ("warning" if es < 12 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{es:.1f}/20</div>
            <div class="kpi-label">Score environnemental</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # Radar individuel
    ra1, ra2 = st.columns([1, 1])
    with ra1:
        categories = ["Accès aux soins", "Professionnels", "Établissements", "Environnement"]
        vals_dept = [
            float(row.get("score_acces", 0) or 0),
            float(row.get("score_pros", 0) or 0),
            float(row.get("score_etabs", 0) or 0),
            float(row.get("score_env", 0) or 0),
        ]
        vals_nat = [
            float(master["score_acces"].mean()),
            float(master["score_pros"].mean()),
            float(master["score_etabs"].mean()),
            float(master["score_env"].mean()),
        ]
        fig_detail_radar = go.Figure()
        fig_detail_radar.add_trace(go.Scatterpolar(
            r=vals_dept + vals_dept[:1], theta=categories + [categories[0]],
            fill="toself", name=dept_choice,
            line_color="#2980b9", fillcolor="rgba(41,128,185,0.25)"
        ))
        fig_detail_radar.add_trace(go.Scatterpolar(
            r=vals_nat + vals_nat[:1], theta=categories + [categories[0]],
            fill="toself", name="Moyenne nationale",
            line_color="#7f8c8d", fillcolor="rgba(127,140,141,0.15)",
            line_dash="dash"
        ))
        fig_detail_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title=f"Profil de {dept_choice} vs Moyenne nationale",
            height=380,
        )
        st.plotly_chart(fig_detail_radar, width="stretch")

    with ra2:
        # Recommandations
        st.markdown("#### 💡 Recommandations stratégiques")
        reco_list = []
        if float(row.get("temps_acces_moyen", 0) or 0) > 12:
            reco_list.append(("🔴", "Accès aux soins critique", "Renforcer les maisons de santé pluridisciplinaires et les dispositifs de télémédecine."))
        elif float(row.get("temps_acces_moyen", 0) or 0) > 7:
            reco_list.append(("🟡", "Accès aux soins à surveiller", "Envisager des consultations avancées et des transports sanitaires renforcés."))
        else:
            reco_list.append(("🟢", "Bon accès aux soins", "Maintenir les structures existantes."))

        if float(row.get("pros_pour_100k", 0) or 0) < 200:
            reco_list.append(("🔴", "Désert médical", "Incitations fiscales et bourses pour l'installation de médecins généralistes."))
        elif float(row.get("pros_pour_100k", 0) or 0) < 400:
            reco_list.append(("🟡", "Densité médicale insuffisante", "Renforcer les partenariats avec les facultés de médecine locales."))
        else:
            reco_list.append(("🟢", "Bonne densité médicale", "Encourager la spécialisation et la formation continue."))

        env_score = row.get("enviro_score", 10) or 10
        if float(str(env_score).replace(",",".")) < 7:
            reco_list.append(("🔴", "Risque environnemental élevé", "Audit sanitaire environnemental recommandé (air, eau, sols)."))
        elif float(str(env_score).replace(",",".")) < 12:
            reco_list.append(("🟡", "Environnement à surveiller", "Suivi des indicateurs de pollution et prévention ciblée."))
        else:
            reco_list.append(("🟢", "Bon environnement santé", "Valoriser cet atout dans la communication territoriale."))

        for icon, title, text in reco_list:
            cls = "alert-critical" if icon == "🔴" else ("alert-warning" if icon == "🟡" else "alert-ok")
            st.markdown(f"""<div class="alert-box {cls}"><strong>{icon} {title}</strong><br>{text}</div>""", unsafe_allow_html=True)

        # Opportunité investissement
        st.markdown("#### 🏗️ Opportunité d'investissement")
        opp_score = (
            (100 - score) * 0.4 +   # plus c'est critique, plus ça nécessite
            min(float(row.get("prix_m2_moyen", 3000) or 3000) / 100, 100) * 0.3 +
            float(row.get("pct_plus_65", 15) or 15) * 2
        )
        opp_score = min(opp_score, 100)
        opp_label = "Forte" if opp_score > 60 else ("Modérée" if opp_score > 35 else "Faible")
        opp_color = "#e74c3c" if opp_score > 60 else ("#f39c12" if opp_score > 35 else "#27ae60")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=opp_score,
            title={"text": f"Besoin d'investissement santé : {opp_label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": opp_color},
                "steps": [
                    {"range": [0, 35], "color": "#eafaf1"},
                    {"range": [35, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fdecea"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": opp_score},
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

    # Classement comparatif
    st.markdown("---")
    st.markdown("#### 🏆 Classement — Top 10 priorités d'intervention")
    priority_df = df.sort_values("score_global").head(10)[
        ["dept", "Nom du département", "Nom de la région", "zone",
         "score_global", "temps_acces_moyen", "pros_pour_100k", "enviro_score"]
    ].copy()
    priority_df["Rang"] = range(1, len(priority_df)+1)
    priority_df = priority_df.rename(columns={
        "dept": "Code", "Nom du département": "Département", "Nom de la région": "Région",
        "zone": "Zone", "score_global": "Score /100",
        "temps_acces_moyen": "Accès (min)", "pros_pour_100k": "Pros/100k", "enviro_score": "Enviro/20"
    })
    for col in ["Score /100", "Accès (min)", "Pros/100k", "Enviro/20"]:
        if col in priority_df.columns:
            priority_df[col] = pd.to_numeric(priority_df[col], errors="coerce").round(1)
    st.dataframe(priority_df.set_index("Rang"), width="stretch")


# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#95a5a6; font-size:0.8rem;'>"
    "Dashboard Santé & Territoires · Données INSEE, RPPS, FINESS, DVF, ANSM · "
    "Croisement multi-sources · Aide à la décision territoriale"
    "</div>",
    unsafe_allow_html=True
)
