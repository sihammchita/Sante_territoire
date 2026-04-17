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
import gdown

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
    :root { --primary: #1e3a5f; --accent: #2ecc71; --warning: #e74c3c; --neutral: #f0f4f8; }
    .main-header { background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%); padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .kpi-card { background: white; border-radius: 10px; padding: 1rem 1.2rem; border-left: 5px solid #2980b9; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
    .kpi-card.danger { border-left-color: #e74c3c; }
    .kpi-card.warning { border-left-color: #f39c12; }
    .kpi-card.success { border-left-color: #27ae60; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .kpi-label { font-size: 0.8rem; color: #7f8c8d; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #1e3a5f; border-bottom: 2px solid #2980b9; padding-bottom: 6px; margin: 1.2rem 0 1rem; }
    .alert-box { border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
    .alert-critical { background: #fdecea; border-left: 4px solid #e74c3c; color: #c0392b; }
    .alert-warning  { background: #fef9e7; border-left: 4px solid #f39c12; color: #d68910; }
    .alert-ok       { background: #eafaf1; border-left: 4px solid #27ae60; color: #1e8449; }
    [data-testid="stSidebar"] { background: #1e3a5f; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; }
    .score-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .badge-red { background: #fdecea; color: #c0392b; }
    .badge-orange { background: #fef9e7; color: #d68910; }
    .badge-green { background: #eafaf1; color: #1e8449; }
</style>
""", unsafe_allow_html=True)


# ─── HELPER: téléchargement Drive robuste avec cache disque ───────────────────
def download_drive_csv(file_id, sep=";", usecols=None, low_memory=True):
    """Télécharge depuis Google Drive et met en cache dans /tmp/"""
    tmp_path = f"/tmp/gdrive_{file_id}.csv"
    if not os.path.exists(tmp_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, tmp_path, quiet=True)
    return pd.read_csv(tmp_path, sep=sep, usecols=usecols, low_memory=low_memory)


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Chargement des données…")
def load_all_data():

    def norm_dept(s):
        return str(s).strip().zfill(2)

    SPECIALISTES = [
        "Cardiologue","Gynécologue / Obstétricien","Pédiatre","Chirurgien urologue",
        "Chirurgien orthopédiste et traumatologue","Endocrinologue-diabétologue",
        "Chirurgien vasculaire","Oto-Rhino-Laryngologue (ORL) et chirurgien cervico-facial",
        "Anesthésiste réanimateur","Neurologue","Pneumologue","Néphrologue",
        "Médecin spécialiste en santé publique et médecine sociale","Ophtalmologiste",
        "Psychiatre","Radiologue","Dermatologue et vénéréologue","Rhumatologue",
        "Stomatologue","Spécialiste en médecine physique et de réadaptation",
        "Spécialiste en allergologie","Chirurgien général","Médecin biologiste",
        "Gastro-entérologue et hépatologue","Radiothérapeute","Chirurgien viscéral",
        "Chirurgien plasticien","Anatomo-Cyto-Pathologiste","Cancérologue radiothérapeute",
        "Chirurgien maxillo-facial et stomatologiste","Médecine d'urgence",
        "Médecine vasculaire","Chirurgien oral","Médecin spécialiste en médecine nucléaire",
        "Neurochirurgien","Cancérologue médical","Spécialiste en médecine interne",
        "Hématologue","Chirurgien thoracique et cardio-vasculaire","Réanimateur médical",
        "Gériatre","Chirurgien maxillo-facial","Neuropsychiatre","Chirurgien infantile",
        "Médecine des Maladies infectieuses et tropicales","Médecin généticien",
        "Médecine légale et expertises médicales"
    ]

    # ── Population ──────────────────────────────────────────────────────────────
    pop = download_drive_csv("11rOLt12iXUxbEQTRlZlbuil_AEp2jxue")
    pop.columns = [c.replace('\r\n', ' ').strip() for c in pop.columns]
    pop["dept"] = pop["code_departement"].apply(norm_dept)
    for col in pop.columns:
        if any(k in col for k in ["Population", "Densité", "25 ans", "25 à 64", "65 ans"]):
            pop[col] = pd.to_numeric(
                pop[col].astype(str).str.replace(" ", "").str.replace(",", "."),
                errors="coerce"
            )
    col_map = {}
    for c in pop.columns:
        if "25 ans" in c:    col_map[c] = "pct_moins_25"
        if "25 à 64" in c:   col_map[c] = "pct_25_64"
        if "65 ans" in c:    col_map[c] = "pct_plus_65"
        if "Population" in c: col_map[c] = "population"
        if "Densité" in c:    col_map[c] = "densite"
    pop = pop.rename(columns=col_map)

    # ── Professionnels santé — SEULEMENT les colonnes nécessaires ───────────────
    pros_raw = download_drive_csv(
        "1_wkO1vtWE2WO9aZmiI8lNPdbecO5V3pA",
        usecols=["code_departement", "specialite_libelle"],
        low_memory=False
    )
    pros_raw["dept"] = pros_raw["code_departement"].apply(norm_dept)
    pros_dept = pros_raw.groupby("dept").agg(
        nb_med_gen=("specialite_libelle", lambda x: (x == "Médecin généraliste").sum()),
        nb_specialistes=("specialite_libelle", lambda x: x.isin(SPECIALISTES).sum()),
        nb_infirmiers=("specialite_libelle", lambda x: (x == "Infirmier").sum()),
        nb_pharmaciens=("specialite_libelle", lambda x: (x == "Pharmacien").sum()),
    ).reset_index()
    pros_dept["nb_pros"] = pros_dept["nb_med_gen"] + pros_dept["nb_specialistes"]
    del pros_raw  # ← libère la RAM immédiatement

    # ── Établissements ───────────────────────────────────────────────────────────
    etabs_raw = download_drive_csv(
        "1hZ71udkcpyquNPgGowvSxUrjrmK-n-PC",
        usecols=["code_departement", "Rslongue", "categetab"]
    )
    etabs_raw["dept"] = etabs_raw["code_departement"].apply(norm_dept)
    etabs_dept = etabs_raw.groupby("dept").agg(
        nb_etabs=("Rslongue", "count"),
        nb_hopitaux=("categetab", lambda x: x.isin(["Centre Hospitalier (C.H.)","Centre Hospitalier Régional (C.H.R.)","Centre Hospitalier Spécialisé lutte Maladies Mentales","Centre hospitalier, ex Hôpital local","Centre de Lutte Contre Cancer","Etablissement de Soins Chirurgicaux","Etablissement Soins Obstétriques Chirurgico-Gynécologiques"]).sum()),
        nb_cliniques=("categetab", lambda x: x.str.contains("Clinique|privé", na=False, case=False).sum()),
    ).reset_index()
    del etabs_raw

    # ── Temps d'accès ─────────────────────────────────────────────────────────
    temps_raw = download_drive_csv(
        "1BoP_S7BYOvDKpwOhpFSEscTM31ltPEEa",
        usecols=["code_departement", "temps_acces", "commune"]
    )
    temps_raw["dept"] = temps_raw["code_departement"].apply(norm_dept)
    temps_dept = temps_raw.groupby("dept").agg(
        temps_acces_moyen=("temps_acces", "mean"),
        temps_acces_max=("temps_acces", "max"),
        nb_communes=("commune", "count"),
        nb_communes_critiques=("temps_acces", lambda x: (x > 15).sum()),
    ).reset_index()
    del temps_raw

    # ── Immobilier — SEULEMENT les colonnes nécessaires ──────────────────────
    immo_raw = download_drive_csv(
        "1Psjk6nf41I_X4dnFE0kgXpCNN5is4s9n",
        usecols=["code_departement", "prix_m2", "valeur_fonciere", "surface_m2", "nom_departement", "type_local"],
        low_memory=False
    )
    immo_raw["dept"] = immo_raw["code_departement"].astype(str).str.zfill(2)
    immo_dept = immo_raw.groupby("dept").agg(
        prix_m2_moyen=("prix_m2", "mean"),
        nb_transactions=("valeur_fonciere", "count"),
        surface_moy=("surface_m2", "mean"),
    ).reset_index()
    # Garder immo_raw en mémoire pour l'onglet immobilier (mais allégé)
    immo_type = immo_raw[["dept", "nom_departement", "type_local", "prix_m2"]].copy()
    del immo_raw

    # ── Environnement ────────────────────────────────────────────────────────
    env = download_drive_csv("1rfdxUJDSX5HzHStZgl5LTUPBoGF9V2i4")
    env.columns = ["Code_region", "nom_region", "enviro_score"]
    env["enviro_score"] = pd.to_numeric(
        env["enviro_score"].astype(str).str.replace(",", "."), errors="coerce"
    )

    # ── Médicaments ──────────────────────────────────────────────────────────
    medic = download_drive_csv("193dosn8DVXFgALvoWynssmxcs-8eNM82")

    # ── MASTER JOIN ──────────────────────────────────────────────────────────
    master = pop[["dept", "Nom du département", "Nom de la région", "Code région",
                  "population", "densite", "pct_moins_25", "pct_25_64", "pct_plus_65"]].copy()
    master = master.merge(pros_dept, on="dept", how="left")
    master = master.merge(etabs_dept, on="dept", how="left")
    master = master.merge(temps_dept, on="dept", how="left")
    master = master.merge(immo_dept,  on="dept", how="left")

    env["Code_region"] = env["Code_region"].astype(str)
    master["Code région"] = master["Code région"].astype(str)
    master = master.merge(env[["Code_region", "enviro_score"]], left_on="Code région", right_on="Code_region", how="left")

    master["population_num"] = pd.to_numeric(
        master["population"].astype(str).str.replace(" ", "").str.replace(",", "."), errors="coerce"
    )
    master["pros_pour_100k"]    = (master["nb_med_gen"] + master["nb_specialistes"]) / (master["population_num"] / 100000)
    master["med_gen_pour_100k"] = master["nb_med_gen"]    / (master["population_num"] / 100000)
    master["hopitaux_pour_100k"]= master["nb_hopitaux"]   / (master["population_num"] / 100000)

    # ── SCORING ──────────────────────────────────────────────────────────────
    def norm_inv(s): mn,mx=s.min(),s.max(); return 100-(s-mn)/(mx-mn+1e-9)*100
    def norm(s):     mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-9)*100

    master["score_acces"]  = norm_inv(master["temps_acces_moyen"].fillna(master["temps_acces_moyen"].median()))
    master["score_pros"]   = norm(master["pros_pour_100k"].fillna(master["pros_pour_100k"].median()))
    master["score_etabs"]  = norm(master["hopitaux_pour_100k"].fillna(master["hopitaux_pour_100k"].median()))
    master["score_env"]    = norm(master["enviro_score"].fillna(master["enviro_score"].median()))
    master["score_global"] = (
        master["score_acces"] * 0.30 +
        master["score_pros"]  * 0.30 +
        master["score_etabs"] * 0.25 +
        master["score_env"]   * 0.15
    )

    def label(s):
        if s < 33: return "🔴 Zone critique"
        if s < 66: return "🟡 Zone intermédiaire"
        return "🟢 Zone favorable"

    master["zone"]       = master["score_global"].apply(label)
    master["zone_short"] = master["score_global"].apply(
        lambda s: "Critique" if s < 33 else ("Intermédiaire" if s < 66 else "Favorable")
    )

    return master, medic, immo_type, env


master, medic, immo_type, env = load_all_data()


# ─── GEOJSON ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement de la carte…")
def load_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    try:
        return requests.get(url, timeout=15).json()
    except:
        return None

geojson = load_geojson()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Filtres")
    st.markdown("---")
    regions_list = ["Toutes les régions"] + sorted(master["Nom de la région"].dropna().unique().tolist())
    selected_region = st.selectbox("Région", regions_list)

    if selected_region != "Toutes les régions":
        depts_list = master[master["Nom de la région"] == selected_region]["Nom du département"].dropna().tolist()
    else:
        depts_list = master["Nom du département"].dropna().tolist()

    selected_depts = st.multiselect("Départements (multi-sélection)", options=sorted(depts_list), default=[])

    st.markdown("---")
    st.markdown("### 📊 Source des données")
    st.caption("• Population 2021 – INSEE\n• Pros santé – RPPS\n• Établissements – FINESS\n• Immo 2025 – DVF\n• Médicaments – ANSM\n• Enviro – Score régional")


# ─── FILTERED DATA ────────────────────────────────────────────────────────────
df = master.copy()
if selected_region != "Toutes les régions":
    df = df[df["Nom de la région"] == selected_region]
if selected_depts:
    df = df[df["Nom du département"].isin(selected_depts)]


# ─── HEADER ───────────────────────────────────────────────────────────────────
scope_label = selected_region if selected_region != "Toutes les régions" else "France entière"
st.markdown(f"""
<div class="main-header">
  <h1>🏥 Dashboard Santé & Territoires — Aide à la Décision</h1>
  <p>Analyse croisée · {scope_label} · {len(df)} département(s) sélectionné(s) · Données 2021–2025</p>
</div>
""", unsafe_allow_html=True)


# ─── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
n_crit  = int((df["zone_short"] == "Critique").sum())
n_inter = int((df["zone_short"] == "Intermédiaire").sum())
n_fav   = int((df["zone_short"] == "Favorable").sum())
avg_acces = df["temps_acces_moyen"].mean()
avg_pros  = df["pros_pour_100k"].mean()

with k1:
    st.markdown(f'<div class="kpi-card danger"><div class="kpi-value">{n_crit}</div><div class="kpi-label">Zones critiques</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card warning"><div class="kpi-value">{n_inter}</div><div class="kpi-label">Zones intermédiaires</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card success"><div class="kpi-value">{n_fav}</div><div class="kpi-label">Zones favorables</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_acces:.1f} min</div><div class="kpi-label">Généralistes le + proche(min)</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_pros:.0f}</div><div class="kpi-label">Pros santé / 100k hab.</div></div>', unsafe_allow_html=True)

st.markdown("")


# ─── TABS ─────────────────────────────────────────────────────────────────────
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
            "Score global santé","Généralistes le plus proche (min)","Professionnels pour 100k hab.",
            "Médecins généralistes pour 100k","Prix immobilier moyen (€/m²)","Part des +65 ans (%)",
        ])
    with c2:
        color_scale = st.selectbox("Palette", ["RdYlGn", "Blues", "Reds", "Plasma"])

    metric_map = {
        "Score global santé":               ("score_global",       True),
        "Généralistes le plus proche (min)":        ("temps_acces_moyen",  False),
        "Professionnels pour 100k hab.":    ("pros_pour_100k",     True),
        "Médecins généralistes pour 100k":  ("med_gen_pour_100k",  True),
        "Prix immobilier moyen (€/m²)":     ("prix_m2_moyen",      False),
        "Part des +65 ans (%)":             ("pct_plus_65",        False),
    }
    col_key, higher_better = metric_map[map_metric]

    if geojson:
        map_df = master.copy() if selected_region == "Toutes les régions" else df.copy()
        map_df["dept_code"] = map_df["dept"].astype(str)
        fig_map = px.choropleth(
            map_df, geojson=geojson, locations="dept_code", color=col_key,
            featureidkey="properties.code", hover_name="Nom du département",
            hover_data={"dept_code": False, col_key: ":.1f"},
            color_continuous_scale=color_scale if higher_better else color_scale + "_r",
            labels={col_key: map_metric},
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=560, margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_colorbar=dict(title=map_metric, thickness=12, len=0.7))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Carte non disponible — vérifiez votre connexion.")

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
    top_n   = st.slider("Nombre de départements à afficher", 10, 50, 20)
    sort_by = st.radio("Trier par", ["Score global","Temps d'accès","Pros / 100k","Prix immobilier"], horizontal=True)
    sort_map = {
        "Score global":    ("score_global",     False),
        "Temps d'accès":   ("temps_acces_moyen",True),
        "Pros / 100k":     ("pros_pour_100k",   False),
        "Prix immobilier": ("prix_m2_moyen",    True),
    }
    sort_col, asc = sort_map[sort_by]
    plot_df = df.sort_values(sort_col, ascending=asc).head(top_n)
    color_zones = {"Critique": "#e74c3c", "Intermédiaire": "#f39c12", "Favorable": "#27ae60"}

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_score = px.bar(
            plot_df.sort_values("score_global"), x="score_global", y="Nom du département",
            color="zone_short", color_discrete_map=color_zones, orientation="h",
            title="Score global santé (0–100)",
            labels={"score_global": "Score", "zone_short": "Zone"},
        )
        fig_score.add_vline(x=33, line_dash="dash", line_color="red",    opacity=0.5)
        fig_score.add_vline(x=66, line_dash="dash", line_color="orange", opacity=0.5)
        fig_score.update_layout(height=500)
        st.plotly_chart(fig_score, use_container_width=True)
    with r1c2:
        fig_acces = px.bar(
            plot_df.sort_values("temps_acces_moyen"), x="temps_acces_moyen", y="Nom du département",
            color="zone_short", color_discrete_map=color_zones, orientation="h",
            title="Temps d'accès moyen aux soins (min)",
            labels={"temps_acces_moyen": "Minutes", "zone_short": "Zone"},
        )
        st.plotly_chart(fig_acces, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_pros = px.scatter(
            df, x="temps_acces_moyen", y="pros_pour_100k",
            color="zone_short", color_discrete_map=color_zones,
            size="population_num", size_max=30,
            hover_name="Nom du département", text="dept",
            title="Accès aux soins vs Densité médicale",
            labels={"temps_acces_moyen":"Temps accès (min)","pros_pour_100k":"Pros / 100k hab.","zone_short":"Zone"},
        )
        fig_pros.update_traces(textposition="top center", textfont_size=7)
        fig_pros.update_layout(height=420)
        st.plotly_chart(fig_pros, use_container_width=True)
    with r2c2:
        radar_df = df[df["zone_short"].isin(["Critique","Favorable"])].groupby("zone_short").agg(
            score_acces=("score_acces","mean"), score_pros=("score_pros","mean"),
            score_etabs=("score_etabs","mean"), score_env=("score_env","mean"),
        ).reset_index()
        categories = ["Accès aux soins","Professionnels","Établissements","Environnement"]
        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            vals = [row["score_acces"],row["score_pros"],row["score_etabs"],row["score_env"]]
            color = "#e74c3c" if row["zone_short"] == "Critique" else "#27ae60"
            fig_radar.add_trace(go.Scatterpolar(
                r=vals+vals[:1], theta=categories+[categories[0]],
                fill="toself", name=row["zone_short"],
                line_color=color, fillcolor=color, opacity=0.35
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
            title="Profil : Zones Critiques vs Favorables", height=420)
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CROISEMENT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🔬 Croisement Multi-dimensionnel des Données</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("Axe X", ["temps_acces_moyen","pros_pour_100k","prix_m2_moyen","pct_plus_65","densite","enviro_score","med_gen_pour_100k"])
    with c2:
        y_axis = st.selectbox("Axe Y", ["pros_pour_100k","score_global","temps_acces_moyen","pct_plus_65","enviro_score","prix_m2_moyen"])
    with c3:
        size_axis = st.selectbox("Taille bulles", ["population_num","nb_etabs","nb_hopitaux","nb_transactions"])

    axis_labels = {
        "temps_acces_moyen":"Temps d'accès (min)","pros_pour_100k":"Pros santé / 100k",
        "prix_m2_moyen":"Prix immo (€/m²)","pct_plus_65":"Part +65 ans (%)",
        "densite":"Densité (hab/km²)","enviro_score":"Score environnement",
        "score_global":"Score global","med_gen_pour_100k":"Med. gen. / 100k",
        "population_num":"Population","nb_etabs":"Nb établissements",
        "nb_hopitaux":"Nb hôpitaux","nb_transactions":"Nb transactions immo",
    }
    fig_cross = px.scatter(
        df.dropna(subset=[x_axis, y_axis, size_axis]),
        x=x_axis, y=y_axis, size=size_axis, size_max=45,
        color="zone_short",
        color_discrete_map={"Critique":"#e74c3c","Intermédiaire":"#f39c12","Favorable":"#27ae60"},
        hover_name="Nom du département",
        labels={x_axis:axis_labels.get(x_axis,x_axis), y_axis:axis_labels.get(y_axis,y_axis), "zone_short":"Zone"},
        title=f"Croisement : {axis_labels.get(x_axis)} vs {axis_labels.get(y_axis)}",
    )
    fig_cross.update_layout(height=500)
    st.plotly_chart(fig_cross, use_container_width=True)

    st.markdown('<div class="section-title">🗂️ Tableau de bord consolidé</div>', unsafe_allow_html=True)
    display_cols = ["dept","Nom du département","Nom de la région","zone","score_global",
                    "temps_acces_moyen","pros_pour_100k","med_gen_pour_100k","nb_hopitaux",
                    "prix_m2_moyen","pct_plus_65","enviro_score"]
    display_df = df[display_cols].copy().rename(columns={
        "dept":"Code","Nom du département":"Département","Nom de la région":"Région",
        "zone":"Zone","score_global":"Score /100","temps_acces_moyen":"Temps accès (min)",
        "pros_pour_100k":"Pros/100k","med_gen_pour_100k":"Med.Gen/100k","nb_hopitaux":"Nb Hôpitaux",
        "prix_m2_moyen":"Prix m² (€)","pct_plus_65":"+65 ans (%)","enviro_score":"Enviro/20"
    })
    for col in ["Score /100","Temps accès (min)","Pros/100k","Med.Gen/100k","Prix m² (€)","+65 ans (%)","Enviro/20"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(1)
    st.dataframe(display_df.sort_values("Score /100"), use_container_width=True, height=420)
    st.download_button("⬇️ Télécharger le tableau (CSV)",
        display_df.to_csv(index=False).encode("utf-8"),
        "dashboard_sante_territoires.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MÉDICAMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">💊 Disponibilité des Médicaments</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    nb_ruptures = int((medic["Statut"] == "Rupture de stock").sum())
    nb_tensions = int((medic["Statut"] == "Tension d'approvisionnement").sum())
    nb_arrets   = int((medic["Statut"] == "Arrêt de commercialisation").sum())
    nb_remise   = int((medic["Statut"] == "Remise à disposition").sum())
    with m1: st.markdown(f'<div class="kpi-card danger"><div class="kpi-value">{nb_ruptures}</div><div class="kpi-label">Ruptures de stock</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="kpi-card warning"><div class="kpi-value">{nb_tensions}</div><div class="kpi-label">Tensions approvisionnement</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="kpi-card danger"><div class="kpi-value">{nb_arrets}</div><div class="kpi-label">Arrêts de commercialisation</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="kpi-card success"><div class="kpi-value">{nb_remise}</div><div class="kpi-label">Remises à disposition</div></div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        fig_med_stat = px.pie(medic, names="Statut", color="Statut", hole=0.4,
            color_discrete_map={"Rupture de stock":"#e74c3c","Tension d'approvisionnement":"#f39c12",
                "Arrêt de commercialisation":"#8e44ad","Remise à disposition":"#27ae60"},
            title="Répartition par statut")
        st.plotly_chart(fig_med_stat, use_container_width=True)
    with mc2:
        top_dom = medic["Domaine(s) médical(aux)"].value_counts().head(10).index
        dom_counts = medic[medic["Domaine(s) médical(aux)"].isin(top_dom)].groupby(
            ["Domaine(s) médical(aux)","Statut"]).size().reset_index(name="count")
        fig_dom = px.bar(dom_counts, x="count", y="Domaine(s) médical(aux)", color="Statut",
            color_discrete_map={"Rupture de stock":"#e74c3c","Tension d'approvisionnement":"#f39c12",
                "Arrêt de commercialisation":"#8e44ad","Remise à disposition":"#27ae60"},
            orientation="h", title="Top 10 domaines médicaux touchés",
            labels={"count":"Nb médicaments","Domaine(s) médical(aux)":""})
        st.plotly_chart(fig_dom, use_container_width=True)

    st.markdown('<div class="section-title">📋 Détail des médicaments en tension / rupture</div>', unsafe_allow_html=True)
    filter_statut = st.multiselect("Filtrer par statut", options=medic["Statut"].unique().tolist(),
        default=["Rupture de stock","Tension d'approvisionnement"])
    st.dataframe(
        medic[medic["Statut"].isin(filter_statut)][
            ["Nom","Statut","Domaine(s) médical(aux)","Produit(s) de santé",
             "Date de début d'incident","Date de fin d'incident"]
        ].sort_values("Statut"),
        use_container_width=True, height=350)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – IMMOBILIER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🏠 Immobilier & Attractivité Territoriale</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        fig_immo_zone = px.box(
            df.dropna(subset=["prix_m2_moyen","zone_short"]),
            x="zone_short", y="prix_m2_moyen", color="zone_short",
            color_discrete_map={"Critique":"#e74c3c","Intermédiaire":"#f39c12","Favorable":"#27ae60"},
            title="Distribution des prix immobiliers par zone santé",
            labels={"prix_m2_moyen":"Prix moyen (€/m²)","zone_short":"Zone santé"})
        st.plotly_chart(fig_immo_zone, use_container_width=True)
    with ic2:
        fig_immo_acces = px.scatter(
            df.dropna(subset=["prix_m2_moyen","temps_acces_moyen"]),
            x="temps_acces_moyen", y="prix_m2_moyen", color="zone_short",
            color_discrete_map={"Critique":"#e74c3c","Intermédiaire":"#f39c12","Favorable":"#27ae60"},
            hover_name="Nom du département",
            title="Prix immobilier vs Temps d'accès aux soins",
            labels={"prix_m2_moyen":"Prix (€/m²)","temps_acces_moyen":"Temps accès (min)","zone_short":"Zone"})
        st.plotly_chart(fig_immo_acces, use_container_width=True)

    st.markdown('<div class="section-title">🏙️ Prix par département — Maisons vs Appartements</div>', unsafe_allow_html=True)
    immo_type_dept = immo_type.groupby(["dept","nom_departement","type_local"])["prix_m2"].mean().reset_index()
    immo_type_dept = immo_type_dept.merge(df[["dept","zone_short"]], on="dept", how="left")
    top_immo = immo_type_dept.groupby("dept")["prix_m2"].mean().nlargest(20).index
    fig_immo_type = px.bar(
        immo_type_dept[immo_type_dept["dept"].isin(top_immo)],
        x="nom_departement", y="prix_m2", color="type_local", barmode="group",
        title="Top 20 départements — Prix m² par type de bien",
        labels={"prix_m2":"Prix moyen (€/m²)","nom_departement":"Département","type_local":"Type"},
        color_discrete_sequence=["#2980b9","#27ae60"])
    fig_immo_type.update_xaxes(tickangle=45)
    st.plotly_chart(fig_immo_type, use_container_width=True)

    st.markdown('<div class="section-title">📈 Corrélation Score Santé ↔ Marché Immobilier</div>', unsafe_allow_html=True)
    corr_cols = ["score_global","prix_m2_moyen","temps_acces_moyen","pros_pour_100k",
                 "nb_hopitaux","pct_plus_65","enviro_score"]
    fig_corr = px.imshow(df[corr_cols].dropna().corr(), text_auto=".2f",
        color_continuous_scale="RdBu", zmin=-1, zmax=1,
        title="Matrice de corrélations", labels=dict(color="Corrélation"))
    fig_corr.update_layout(height=420)
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – AIDE À LA DÉCISION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">🎯 Aide à la Décision — Priorisation Territoriale</div>', unsafe_allow_html=True)
    st.markdown("Ce module synthétise l'ensemble des indicateurs pour vous aider à **identifier les priorités d'intervention**, **évaluer l'attractivité d'un territoire** et **orienter vos décisions** en matière de politique de santé.")

    dept_choice = st.selectbox("Département", options=df.sort_values("score_global")["Nom du département"].dropna().tolist())
    row = df[df["Nom du département"] == dept_choice].iloc[0]
    score = row["score_global"]

    if score < 33:   badge_class, verdict, alert_class = "badge-red",    "🔴 Zone Critique — Intervention prioritaire recommandée",    "alert-critical"
    elif score < 66: badge_class, verdict, alert_class = "badge-orange", "🟡 Zone Intermédiaire — Surveillance et amélioration ciblée","alert-warning"
    else:            badge_class, verdict, alert_class = "badge-green",  "🟢 Zone Favorable — Maintien des acquis recommandé",          "alert-ok"

    st.markdown(f"""
    <div class="alert-box {alert_class}">
      <strong>{dept_choice} ({row['dept']}) — {row.get('Nom de la région','')}</strong><br>
      {verdict}<br>
      Score global : <span class="score-badge {badge_class}">{score:.1f}/100</span>
    </div>""", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    ta = float(row.get("temps_acces_moyen", np.nan) or 0)
    pp = float(row.get("pros_pour_100k",    np.nan) or 0)
    pm = float(row.get("prix_m2_moyen",     np.nan) or 0)
    es = float(str(row.get("enviro_score",  10) or 10).replace(",","."))

    with d1: st.markdown(f'<div class="kpi-card {"danger" if ta>12 else "warning" if ta>7 else "success"}"><div class="kpi-value">{ta:.1f} min</div><div class="kpi-label">Temps d\'accès moyen</div></div>', unsafe_allow_html=True)
    with d2: st.markdown(f'<div class="kpi-card {"danger" if pp<200 else "warning" if pp<400 else "success"}"><div class="kpi-value">{pp:.0f}</div><div class="kpi-label">Pros / 100k hab.</div></div>', unsafe_allow_html=True)
    with d3: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{pm:.0f} €</div><div class="kpi-label">Prix immo moyen /m²</div></div>', unsafe_allow_html=True)
    with d4: st.markdown(f'<div class="kpi-card {"danger" if es<7 else "warning" if es<12 else "success"}"><div class="kpi-value">{es:.1f}/20</div><div class="kpi-label">Score environnemental</div></div>', unsafe_allow_html=True)

    ra1, ra2 = st.columns(2)
    with ra1:
        categories = ["Accès aux soins","Professionnels","Établissements","Environnement"]
        vals_dept = [float(row.get(k,0) or 0) for k in ["score_acces","score_pros","score_etabs","score_env"]]
        vals_nat  = [float(master[k].mean()) for k in ["score_acces","score_pros","score_etabs","score_env"]]
        fig_detail_radar = go.Figure()
        fig_detail_radar.add_trace(go.Scatterpolar(r=vals_dept+vals_dept[:1], theta=categories+[categories[0]],
            fill="toself", name=dept_choice, line_color="#2980b9", fillcolor="rgba(41,128,185,0.25)"))
        fig_detail_radar.add_trace(go.Scatterpolar(r=vals_nat+vals_nat[:1], theta=categories+[categories[0]],
            fill="toself", name="Moyenne nationale", line_color="#7f8c8d", fillcolor="rgba(127,140,141,0.15)", line_dash="dash"))
        fig_detail_radar.update_layout(polar=dict(radialaxis=dict(range=[0,100])),
            title=f"Profil de {dept_choice} vs Moyenne nationale", height=380)
        st.plotly_chart(fig_detail_radar, use_container_width=True)

    with ra2:
        st.markdown("#### 💡 Recommandations stratégiques")
        reco_list = []
        if ta > 12:  reco_list.append(("🔴","Accès aux soins critique","Renforcer les maisons de santé pluridisciplinaires et les dispositifs de télémédecine."))
        elif ta > 7: reco_list.append(("🟡","Accès aux soins à surveiller","Envisager des consultations avancées et des transports sanitaires renforcés."))
        else:        reco_list.append(("🟢","Bon accès aux soins","Maintenir les structures existantes."))

        if pp < 200:  reco_list.append(("🔴","Désert médical","Incitations fiscales et bourses pour l'installation de médecins généralistes."))
        elif pp < 400:reco_list.append(("🟡","Densité médicale insuffisante","Renforcer les partenariats avec les facultés de médecine locales."))
        else:         reco_list.append(("🟢","Bonne densité médicale","Encourager la spécialisation et la formation continue."))

        if es < 7:    reco_list.append(("🔴","Risque environnemental élevé","Audit sanitaire environnemental recommandé (air, eau, sols)."))
        elif es < 12: reco_list.append(("🟡","Environnement à surveiller","Suivi des indicateurs de pollution et prévention ciblée."))
        else:         reco_list.append(("🟢","Bon environnement santé","Valoriser cet atout dans la communication territoriale."))

        for icon, title, text in reco_list:
            cls = "alert-critical" if icon=="🔴" else ("alert-warning" if icon=="🟡" else "alert-ok")
            st.markdown(f'<div class="alert-box {cls}"><strong>{icon} {title}</strong><br>{text}</div>', unsafe_allow_html=True)

        st.markdown("#### 🏗️ Opportunité d'investissement")
        opp_score = min((100-score)*0.4 + min(pm/100,100)*0.3 + float(row.get("pct_plus_65",15) or 15)*2, 100)
        opp_label = "Forte" if opp_score>60 else ("Modérée" if opp_score>35 else "Faible")
        opp_color = "#e74c3c" if opp_score>60 else ("#f39c12" if opp_score>35 else "#27ae60")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=opp_score,
            title={"text": f"Besoin d'investissement santé : {opp_label}"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":opp_color},
                   "steps":[{"range":[0,35],"color":"#eafaf1"},{"range":[35,60],"color":"#fef9e7"},{"range":[60,100],"color":"#fdecea"}],
                   "threshold":{"line":{"color":"black","width":2},"thickness":0.75,"value":opp_score}}
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏆 Classement — Top 10 priorités d'intervention")
    priority_df = df.sort_values("score_global").head(10)[
        ["dept","Nom du département","Nom de la région","zone","score_global","temps_acces_moyen","pros_pour_100k","enviro_score"]
    ].copy()
    priority_df["Rang"] = range(1, len(priority_df)+1)
    priority_df = priority_df.rename(columns={
        "dept":"Code","Nom du département":"Département","Nom de la région":"Région","zone":"Zone",
        "score_global":"Score /100","temps_acces_moyen":"Accès (min)","pros_pour_100k":"Pros/100k","enviro_score":"Enviro/20"
    })
    for col in ["Score /100","Accès (min)","Pros/100k","Enviro/20"]:
        if col in priority_df.columns:
            priority_df[col] = pd.to_numeric(priority_df[col], errors="coerce").round(1)
    st.dataframe(priority_df.set_index("Rang"), use_container_width=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#95a5a6; font-size:0.8rem;'>"
    "Dashboard Santé & Territoires · Données INSEE, RPPS, FINESS, DVF, ANSM · "
    "Croisement multi-sources · Aide à la décision territoriale</div>",
    unsafe_allow_html=True)
