import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings

warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Santé & Territoires",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ID Google Drive du fichier pathologies (format AMELI/SNDS : annee;patho_niv1;dept;Ntop;Npop;prev;...)
PATHO_DRIVE_ID = "1cBaxRy-hQl-qo8EDYRm7VYztXRJNRGuO"

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --primary: #1e3a5f; --accent: #2ecc71; --warning: #e74c3c; --neutral: #f0f4f8; }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .kpi-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 5px solid #2980b9; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    }
    .kpi-card.danger  { border-left-color: #e74c3c; }
    .kpi-card.warning { border-left-color: #f39c12; }
    .kpi-card.success { border-left-color: #27ae60; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .kpi-label { font-size: 0.8rem; color: #7f8c8d; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2980b9; padding-bottom: 6px; margin: 1.2rem 0 1rem;
    }
    .user-story {
        border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 1rem;
        background: #eaf4fb; border-left: 4px solid #2980b9; font-size: 0.88rem; color: #1e3a5f;
    }
    .alert-box { border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
    .alert-critical { background: #fdecea; border-left: 4px solid #e74c3c; color: #c0392b; }
    .alert-warning  { background: #fef9e7; border-left: 4px solid #f39c12; color: #d68910; }
    .alert-ok       { background: #eafaf1; border-left: 4px solid #27ae60; color: #1e8449; }
    [data-testid="stSidebar"] { background: #1e3a5f; }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; }
    .stTab [data-baseweb="tab"] { font-weight: 600; }
    .score-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .badge-red    { background: #fdecea; color: #c0392b; }
    .badge-orange { background: #fef9e7; color: #d68910; }
    .badge-green  { background: #eafaf1; color: #1e8449; }
</style>
""", unsafe_allow_html=True)


# ─── CHARGEMENT DES DONNÉES ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def load_all_data():
    def norm_dept(s):
        return str(s).strip().zfill(2)

    # ── Population (INSEE) ────────────────────────────────────────────────────
    pop = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=11rOLt12iXUxbEQTRlZlbuil_AEp2jxue",
        sep=";"
    )
    pop.columns = [c.replace('\r\n', ' ').strip() for c in pop.columns]
    pop["dept"] = pop["code_departement"].apply(norm_dept)
    for col in ["Population municipale", "Densité de population (en km²)",
                "Part des moins  de 25 ans (en %)", "Part des  25 à 64 ans (en %)",
                "Part des  plus de 65 ans (en %)"]:
        if col in pop.columns:
            pop[col] = pop[col].astype(str).str.replace(" ", "").str.replace(",", ".").replace("nan", np.nan)
            try:
                pop[col] = pd.to_numeric(pop[col], errors="coerce")
            except Exception:
                pass
    col_map = {}
    for c in pop.columns:
        if "25 ans"    in c: col_map[c] = "pct_moins_25"
        if "25 à 64"   in c: col_map[c] = "pct_25_64"
        if "65 ans"    in c: col_map[c] = "pct_plus_65"
        if "Population" in c: col_map[c] = "population"
        if "Densité"    in c: col_map[c] = "densite"
    pop = pop.rename(columns=col_map)

    # ── Professionnels de santé (RPPS) — détail par spécialité ───────────────
    pros = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1_wkO1vtWE2WO9aZmiI8lNPdbecO5V3pA",
        sep=";", low_memory=False
    )
    pros["dept"] = pros["code_departement"].apply(norm_dept)
    pros_dept = pros.groupby("dept").agg(
        nb_pros=("specialite_libelle", "count"),
        nb_med_gen=("specialite_libelle",    lambda x: (x == "Médecin généraliste").sum()),
        nb_infirmiers=("specialite_libelle", lambda x: (x == "Infirmier").sum()),
        nb_pharmaciens=("specialite_libelle",lambda x: (x == "Pharmacien").sum()),
        nb_cardio=("specialite_libelle",     lambda x: (x == "Cardiologue").sum()),
        nb_ophtalmo=("specialite_libelle",   lambda x: (x == "Ophtalmologue").sum()),
        nb_psychiatre=("specialite_libelle", lambda x: (x == "Psychiatre").sum()),
        nb_gyneco=("specialite_libelle",     lambda x: (x == "Gynécologue médical").sum()),
        nb_pediatre=("specialite_libelle",   lambda x: (x == "Pédiatre").sum()),
    ).reset_index()
    pros_dept["nb_specialistes"] = (
        pros_dept["nb_cardio"] + pros_dept["nb_ophtalmo"] +
        pros_dept["nb_psychiatre"] + pros_dept["nb_gyneco"] + pros_dept["nb_pediatre"]
    )

    # ── Établissements (FINESS) ───────────────────────────────────────────────
    etabs = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1hZ71udkcpyquNPgGowvSxUrjrmK-n-PC",
        sep=";"
    )
    etabs["dept"] = etabs["code_departement"].apply(norm_dept)
    etabs_dept = etabs.groupby("dept").agg(
        nb_etabs=("Rslongue", "count"),
        nb_hopitaux=("categetab", lambda x: x.isin([
            "Centre Hospitalier (C.H.)", "Centre Hospitalier Régional (C.H.R.)"
        ]).sum()),
        nb_cliniques=("categetab", lambda x: x.str.contains("Clinique|privé", na=False, case=False).sum()),
    ).reset_index()

    # ── Temps d'accès aux soins (commune par commune) ─────────────────────────
    temps = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1BoP_S7BYOvDKpwOhpFSEscTM31ltPEEa",
        sep=";"
    )
    temps["dept"] = temps["code_departement"].apply(norm_dept)
    temps_dept = temps.groupby("dept").agg(
        temps_acces_moyen=("temps_acces", "mean"),
        temps_acces_max=("temps_acces", "max"),
        nb_communes=("commune", "count"),
        nb_communes_critiques=("temps_acces", lambda x: (x > 15).sum()),
    ).reset_index()

    # ── Immobilier (DVF) ──────────────────────────────────────────────────────
    immo = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1Psjk6nf41I_X4dnFE0kgXpCNN5is4s9n",
        sep=";", low_memory=False
    )
    immo["dept"] = immo["code_departement"].astype(str).str.zfill(2)
    immo_dept = immo.groupby("dept").agg(
        prix_m2_moyen=("prix_m2", "mean"),
        nb_transactions=("valeur_fonciere", "count"),
        surface_moy=("surface_m2", "mean"),
    ).reset_index()
    # Évite le double chargement dans le tab Immobilier
    immo_type_dept = immo.groupby(["dept", "nom_departement", "type_local"])["prix_m2"].mean().reset_index()

    # ── Score environnement santé (par région) ────────────────────────────────
    env = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1rfdxUJDSX5HzHStZgl5LTUPBoGF9V2i4",
        sep=";"
    )
    env.columns = ["Code_region", "nom_region", "enviro_score"]
    env["enviro_score"] = env["enviro_score"].astype(str).str.replace(",", ".").replace("nan", np.nan)
    env["enviro_score"] = pd.to_numeric(env["enviro_score"], errors="coerce")

    # ── Médicaments (ANSM) ────────────────────────────────────────────────────
    medic = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=193dosn8DVXFgALvoWynssmxcs-8eNM82",
        sep=";"
    )

    # ── Pathologies (AMELI/SNDS) — chargement conditionnel ───────────────────
    patho_dept   = pd.DataFrame()
    patho_senior = pd.DataFrame()
    if PATHO_DRIVE_ID != "VOTRE_ID_GOOGLE_DRIVE_PATHOLOGIES":
        try:
            patho_raw = pd.read_csv(
                f"https://drive.google.com/uc?export=download&id={PATHO_DRIVE_ID}",
                sep=";"
            )
            patho_raw["dept"] = patho_raw["dept"].astype(str).str.zfill(2)
            for col in ["prev", "Ntop", "Npop"]:
                patho_raw[col] = pd.to_numeric(
                    patho_raw[col].astype(str).str.replace(",", "."), errors="coerce"
                )
            # Agrégation toutes tranches d'âge : ratio réel = Σ(Ntop)/Σ(Npop) * 100
            patho_dept = patho_raw.groupby(["dept", "patho_niv1"]).agg(
                nb_patients=("Ntop", "sum"),
                population_patho=("Npop", "sum"),
            ).reset_index()
            patho_dept["prevalence_pct"] = (
                patho_dept["nb_patients"] / patho_dept["population_patho"] * 100
            ).round(3)
            # Sous-ensemble 65 ans et + pour User 3
            if "groupe_age" in patho_raw.columns:
                patho_senior_raw = patho_raw[patho_raw["groupe_age"].astype(str).str.contains("65", na=False)]
                patho_senior = patho_senior_raw.groupby(["dept", "patho_niv1"]).agg(
                    nb_patients_senior=("Ntop", "sum"),
                    population_senior=("Npop", "sum"),
                ).reset_index()
                patho_senior["prevalence_pct"] = (
                    patho_senior["nb_patients_senior"] / patho_senior["population_senior"] * 100
                ).round(3)
        except Exception:
            patho_dept   = pd.DataFrame()
            patho_senior = pd.DataFrame()

    # ── Jointure maître ───────────────────────────────────────────────────────
    master = pop[[
        "dept", "Nom du département", "Nom de la région", "Code région",
        "population", "densite", "pct_moins_25", "pct_25_64", "pct_plus_65"
    ]].copy()
    master = master.merge(pros_dept,  on="dept", how="left")
    master = master.merge(etabs_dept, on="dept", how="left")
    master = master.merge(temps_dept, on="dept", how="left")
    master = master.merge(immo_dept,  on="dept", how="left")

    env["Code_region"]    = env["Code_region"].astype(str)
    master["Code région"] = master["Code région"].astype(str)
    master = master.merge(
        env[["Code_region", "enviro_score"]],
        left_on="Code région", right_on="Code_region", how="left"
    )

    # ── Indicateurs dérivés (tous en /100 000 hab.) ───────────────────────────
    master["population_num"] = pd.to_numeric(
        master["population"].astype(str).str.replace(" ", "").str.replace(",", "."),
        errors="coerce"
    )
    base = master["population_num"] / 100_000
    master["pros_pour_100k"]       = master["nb_pros"]       / base
    master["med_gen_pour_100k"]    = master["nb_med_gen"]     / base
    master["hopitaux_pour_100k"]   = master["nb_hopitaux"]    / base
    master["cardio_pour_100k"]     = master["nb_cardio"]      / base
    master["ophtalmo_pour_100k"]   = master["nb_ophtalmo"]    / base
    master["psychiatre_pour_100k"] = master["nb_psychiatre"]  / base
    master["gyneco_pour_100k"]     = master["nb_gyneco"]      / base
    master["pediatre_pour_100k"]   = master["nb_pediatre"]    / base

    # ── Scoring santé global (normalisation min-max 0-100) ───────────────────
    def norm_inv(series):
        mn, mx = series.min(), series.max()
        return 100 - (series - mn) / (mx - mn + 1e-9) * 100

    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9) * 100

    master["score_acces"]  = norm_inv(master["temps_acces_moyen"].fillna(master["temps_acces_moyen"].median()))
    master["score_pros"]   = norm(master["pros_pour_100k"].fillna(master["pros_pour_100k"].median()))
    master["score_etabs"]  = norm(master["hopitaux_pour_100k"].fillna(master["hopitaux_pour_100k"].median()))
    master["score_env"]    = norm(master["enviro_score"].fillna(master["enviro_score"].median()))
    master["score_global"] = (
        master["score_acces"]  * 0.30 +
        master["score_pros"]   * 0.30 +
        master["score_etabs"]  * 0.25 +
        master["score_env"]    * 0.15
    )

    # Indice de Désertification Médicale (IDM) : 0 = bien couvert, 100 = désert
    master["indice_desertification"] = (
        norm(master["temps_acces_moyen"].fillna(master["temps_acces_moyen"].median()))     * 0.40 +
        norm_inv(master["pros_pour_100k"].fillna(master["pros_pour_100k"].median()))       * 0.40 +
        norm_inv(master["med_gen_pour_100k"].fillna(master["med_gen_pour_100k"].median())) * 0.20
    )

    # Indice de Vulnérabilité Sénior (IVS) : 0 = peu vulnérable, 100 = très vulnérable
    master["indice_vulnerabilite_senior"] = (
        norm(master["pct_plus_65"].fillna(master["pct_plus_65"].median()))                    * 0.50 +
        norm_inv(master["hopitaux_pour_100k"].fillna(master["hopitaux_pour_100k"].median())) * 0.30 +
        norm(master["temps_acces_moyen"].fillna(master["temps_acces_moyen"].median()))        * 0.20
    )

    master["zone"] = master["score_global"].apply(
        lambda s: "🔴 Zone critique" if s < 33 else ("🟡 Zone intermédiaire" if s < 66 else "🟢 Zone favorable")
    )
    master["zone_short"] = master["score_global"].apply(
        lambda s: "Critique" if s < 33 else ("Intermédiaire" if s < 66 else "Favorable")
    )

    return master, medic, immo_type_dept, etabs, temps, env, patho_dept, patho_senior


master, medic, immo_type_dept, etabs, temps, env, patho_dept, patho_senior = load_all_data()
patho_available = not patho_dept.empty


# ─── GEOJSON DÉPARTEMENTS ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement de la carte…")
def load_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    try:
        return requests.get(url, timeout=15).json()
    except Exception:
        return None

geojson = load_geojson()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Filtres")
    st.markdown("---")

    regions_list = ["Toutes les régions"] + sorted(master["Nom de la région"].dropna().unique().tolist())
    selected_region = st.selectbox("Région", regions_list)

    depts_list = (
        master[master["Nom de la région"] == selected_region]["Nom du département"].dropna().tolist()
        if selected_region != "Toutes les régions"
        else master["Nom du département"].dropna().tolist()
    )
    selected_depts = st.multiselect("Départements", options=sorted(depts_list), default=[])

    st.markdown("---")
    st.markdown("### 🎚 Pondération du score global")
    w_acces  = st.slider("Poids Accès aux soins", 0, 100, 30, 5)
    w_pros   = st.slider("Poids Professionnels",  0, 100, 30, 5)
    w_etabs  = st.slider("Poids Établissements",  0, 100, 25, 5)
    w_env    = st.slider("Poids Environnement",   0, 100, 15, 5)
    total_w  = w_acces + w_pros + w_etabs + w_env
    if total_w > 0:
        master["score_global"] = (
            master["score_acces"]  * (w_acces / total_w) +
            master["score_pros"]   * (w_pros  / total_w) +
            master["score_etabs"]  * (w_etabs / total_w) +
            master["score_env"]    * (w_env   / total_w)
        )
    master["zone"] = master["score_global"].apply(
        lambda s: "🔴 Zone critique" if s < 33 else ("🟡 Zone intermédiaire" if s < 66 else "🟢 Zone favorable")
    )
    master["zone_short"] = master["score_global"].apply(
        lambda s: "Critique" if s < 33 else ("Intermédiaire" if s < 66 else "Favorable")
    )

    st.markdown("---")
    st.markdown("### 📊 Sources des données")
    st.caption(
        "• Population 2021 – INSEE\n"
        "• Pros santé – RPPS\n"
        "• Établissements – FINESS\n"
        "• Immo 2025 – DVF\n"
        "• Médicaments – ANSM\n"
        "• Enviro – Score régional\n"
        "• Pathologies – AMELI/SNDS"
    )


# ─── DONNÉES FILTRÉES ─────────────────────────────────────────────────────────
df = master.copy()
if selected_region != "Toutes les régions":
    df = df[df["Nom de la région"] == selected_region]
if selected_depts:
    df = df[df["Nom du département"].isin(selected_depts)]


# ─── EN-TÊTE ──────────────────────────────────────────────────────────────────
scope_label = selected_region if selected_region != "Toutes les régions" else "France entière"
st.markdown(f"""
<div class="main-header">
  <h1>🏥 Dashboard Santé & Territoires — Aide à la Décision</h1>
  <p>Analyse croisée · {scope_label} · {len(df)} département(s) · Données 2021–2025</p>
</div>
""", unsafe_allow_html=True)


# ─── KPI GLOBAUX ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

n_crit  = int((df["zone_short"] == "Critique").sum())
n_inter = int((df["zone_short"] == "Intermédiaire").sum())
n_fav   = int((df["zone_short"] == "Favorable").sum())

pop_total = df["population_num"].sum()
# Moyenne pondérée par population (représente le temps d'accès réel du territoire)
avg_acces = (
    (df["temps_acces_moyen"] * df["population_num"]).sum() / pop_total
    if pop_total > 0 else np.nan
)
# Ratio agrégé sur l'ensemble de la population filtrée (pas moyenne de ratios)
avg_pros = (
    df["nb_pros"].sum() / (pop_total / 100_000)
    if pop_total > 0 else np.nan
)

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
        <div class="kpi-label">Temps accès moy. (pond.)</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{avg_pros:.0f}</div>
        <div class="kpi-label">Pros santé / 100k hab.</div></div>""", unsafe_allow_html=True)

st.markdown("")

COLOR_ZONES = {"Critique": "#e74c3c", "Intermédiaire": "#f39c12", "Favorable": "#27ae60"}

# ─── ONGLETS ──────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🗺️ Carte Territoriale",
    "📊 Analyse Comparative",
    "🔬 Croisement Données",
    "🏗️ Désertification & Implantation",
    "🧬 Pathologies & Prévention",
    "💊 Médicaments",
    "🏠 Immobilier & Santé",
    "🎯 Aide à la Décision",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CARTE TERRITORIALE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">🗺️ Carte des Départements</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        map_metric = st.selectbox("Indicateur cartographié", [
            "Score global santé",
            "Indice de Désertification (IDM)",
            "Indice Vulnérabilité Sénior (IVS)",
            "Temps d'accès moyen (min)",
            "Professionnels pour 100k hab.",
            "Médecins généralistes pour 100k",
            "Hôpitaux pour 100k hab.",
            "Prix immobilier moyen (€/m²)",
            "Part des +65 ans (%)",
        ])
    with c2:
        color_scale = st.selectbox("Palette", ["RdYlGn", "Blues", "Reds", "Plasma"])

    metric_map = {
        "Score global santé":                ("score_global",                True),
        "Indice de Désertification (IDM)":   ("indice_desertification",      False),
        "Indice Vulnérabilité Sénior (IVS)": ("indice_vulnerabilite_senior", False),
        "Temps d'accès moyen (min)":         ("temps_acces_moyen",           False),
        "Professionnels pour 100k hab.":     ("pros_pour_100k",              True),
        "Médecins généralistes pour 100k":   ("med_gen_pour_100k",           True),
        "Hôpitaux pour 100k hab.":           ("hopitaux_pour_100k",          True),
        "Prix immobilier moyen (€/m²)":      ("prix_m2_moyen",               False),
        "Part des +65 ans (%)":              ("pct_plus_65",                 False),
    }
    col_key, higher_better = metric_map[map_metric]

    if geojson:
        map_df = master.copy() if selected_region == "Toutes les régions" else df.copy()
        map_df["dept_code"] = map_df["dept"].astype(str)
        fig_map = px.choropleth(
            map_df,
            geojson=geojson,
            locations="dept_code",
            color=col_key,
            featureidkey="properties.code",
            hover_name="Nom du département",
            hover_data={
                "dept_code": False,
                col_key: ":.1f",
                "zone": True,
                "pros_pour_100k": ":.0f",
                "hopitaux_pour_100k": ":.2f",
                "temps_acces_moyen": ":.1f",
            },
            color_continuous_scale=color_scale if higher_better else color_scale + "_r",
            labels={
                col_key: map_metric,
                "zone": "Zone",
                "pros_pour_100k": "Pros/100k",
                "hopitaux_pour_100k": "Hôp./100k",
                "temps_acces_moyen": "Accès (min)",
            },
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            height=560, margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title=map_metric, thickness=12, len=0.7)
        )
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.warning("Carte non disponible — vérifiez votre connexion internet.")

    l1, l2, l3 = st.columns(3)
    crit_dept = df[df["zone_short"] == "Critique"].sort_values("score_global").head(5)
    inter_dept = df[df["zone_short"] == "Intermédiaire"].sort_values("score_global").head(5)
    fav_dept  = df[df["zone_short"] == "Favorable"].sort_values("score_global", ascending=False).head(5)
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

    top_n  = st.slider("Nombre de départements", 10, 50, 20)
    sort_by = st.radio("Trier par", ["Score global", "Temps d'accès", "Pros / 100k", "IDM"], horizontal=True)

    sort_map = {
        "Score global":   ("score_global",            False),
        "Temps d'accès":  ("temps_acces_moyen",        True),
        "Pros / 100k":    ("pros_pour_100k",           False),
        "IDM":            ("indice_desertification",   True),
    }
    sort_col, asc = sort_map[sort_by]
    plot_df = df.sort_values(sort_col, ascending=asc).head(top_n)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig_score = px.bar(
            plot_df.sort_values("score_global"),
            x="score_global", y="Nom du département",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            orientation="h", title="Score global santé (0–100)",
            labels={"score_global": "Score", "zone_short": "Zone"},
        )
        fig_score.add_vline(x=33, line_dash="dash", line_color="red",    opacity=0.5)
        fig_score.add_vline(x=66, line_dash="dash", line_color="orange", opacity=0.5)
        fig_score.update_layout(height=500, showlegend=True, legend_title="Zone")
        st.plotly_chart(fig_score, width="stretch")

    with r1c2:
        fig_acces = px.bar(
            plot_df.sort_values("temps_acces_moyen"),
            x="temps_acces_moyen", y="Nom du département",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            orientation="h", title="Temps d'accès moyen aux soins (min)",
            labels={"temps_acces_moyen": "Minutes", "zone_short": "Zone"},
        )
        st.plotly_chart(fig_acces, width="stretch")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_pros = px.scatter(
            df,
            x="temps_acces_moyen", y="pros_pour_100k",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            size="population_num", size_max=30,
            hover_name="Nom du département", text="dept",
            title="Accès aux soins vs Densité médicale",
            labels={
                "temps_acces_moyen": "Temps accès (min)",
                "pros_pour_100k": "Pros / 100k hab.",
                "zone_short": "Zone"
            },
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
            color = "#e74c3c" if row["zone_short"] == "Critique" else "#27ae60"
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=categories + [categories[0]],
                fill="toself", name=row["zone_short"],
                line_color=color, fillcolor=color, opacity=0.35
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title="Profil moyen : Zones Critiques vs Favorables",
            height=420,
        )
        st.plotly_chart(fig_radar, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CROISEMENT DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🔬 Croisement Multi-dimensionnel</div>', unsafe_allow_html=True)

    axis_options = [
        "temps_acces_moyen", "pros_pour_100k", "hopitaux_pour_100k",
        "med_gen_pour_100k", "prix_m2_moyen", "pct_plus_65", "densite",
        "enviro_score", "indice_desertification", "indice_vulnerabilite_senior",
        "cardio_pour_100k", "psychiatre_pour_100k", "ophtalmo_pour_100k",
    ]
    axis_labels = {
        "temps_acces_moyen":           "Temps d'accès (min)",
        "pros_pour_100k":              "Pros santé / 100k",
        "hopitaux_pour_100k":          "Hôpitaux / 100k",
        "med_gen_pour_100k":           "Med. gén. / 100k",
        "prix_m2_moyen":               "Prix immo (€/m²)",
        "pct_plus_65":                 "Part +65 ans (%)",
        "densite":                     "Densité (hab/km²)",
        "enviro_score":                "Score enviro. (rég./20)",
        "indice_desertification":      "IDM (0=couvert, 100=désert)",
        "indice_vulnerabilite_senior": "IVS (0=ok, 100=vulnérable)",
        "cardio_pour_100k":            "Cardiologues / 100k",
        "psychiatre_pour_100k":        "Psychiatres / 100k",
        "ophtalmo_pour_100k":          "Ophtalmologues / 100k",
        "population_num":              "Population",
        "nb_etabs":                    "Nb établissements",
        "nb_transactions":             "Nb transactions immo",
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("Axe X", axis_options, index=0)
    with c2:
        y_axis = st.selectbox("Axe Y", axis_options, index=1)
    with c3:
        size_axis = st.selectbox("Taille bulles", ["population_num", "nb_etabs", "nb_transactions"], index=0)

    fig_cross = px.scatter(
        df.dropna(subset=[x_axis, y_axis, size_axis]),
        x=x_axis, y=y_axis,
        size=size_axis, size_max=45,
        color="zone_short", color_discrete_map=COLOR_ZONES,
        hover_name="Nom du département",
        hover_data={"dept": True, "zone_short": True, x_axis: ":.1f", y_axis: ":.1f"},
        labels={
            x_axis: axis_labels.get(x_axis, x_axis),
            y_axis: axis_labels.get(y_axis, y_axis),
            "zone_short": "Zone",
        },
        title=f"Croisement : {axis_labels.get(x_axis)} vs {axis_labels.get(y_axis)}",
    )
    fig_cross.update_layout(height=500)
    st.plotly_chart(fig_cross, width="stretch")

    st.markdown('<div class="section-title">🗂️ Tableau consolidé</div>', unsafe_allow_html=True)
    display_cols = [
        "dept", "Nom du département", "Nom de la région", "zone",
        "score_global", "indice_desertification", "temps_acces_moyen",
        "pros_pour_100k", "med_gen_pour_100k", "hopitaux_pour_100k",
        "prix_m2_moyen", "pct_plus_65", "enviro_score",
    ]
    display_df = df[display_cols].copy().rename(columns={
        "dept": "Code", "Nom du département": "Département", "Nom de la région": "Région",
        "zone": "Zone", "score_global": "Score /100",
        "indice_desertification": "IDM",
        "temps_acces_moyen": "Accès (min)", "pros_pour_100k": "Pros/100k",
        "med_gen_pour_100k": "MedGen/100k", "hopitaux_pour_100k": "Hôp./100k",
        "prix_m2_moyen": "Prix m² (€)", "pct_plus_65": "+65 ans (%)",
        "enviro_score": "Enviro. (rég./20)",
    })
    for col in ["Score /100", "IDM", "Accès (min)", "Pros/100k", "MedGen/100k",
                "Hôp./100k", "Prix m² (€)", "+65 ans (%)", "Enviro. (rég./20)"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(1)
    st.dataframe(display_df.sort_values("Score /100"), width="stretch", height=420)
    st.download_button(
        "⬇️ Télécharger (CSV)",
        display_df.to_csv(index=False).encode("utf-8"),
        "dashboard_sante_territoires.csv", "text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – DÉSERTIFICATION & IMPLANTATION (User 1 + User 3)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    # ── User 1 ────────────────────────────────────────────────────────────────
    st.markdown("""<div class="user-story">
    👤 <strong>User 1 — L'Investisseur Public (Président de Département)</strong><br>
    <em>"Je veux identifier les zones où le besoin médical est fort et le foncier accessible,
    afin d'ouvrir une Maison de Santé au meilleur coût."</em>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏗️ Indice de Désertification Médicale (IDM)</div>', unsafe_allow_html=True)

    # KPIs désertification
    d1, d2, d3, d4 = st.columns(4)
    n_desert   = int((df["indice_desertification"] > 66).sum())
    idm_max    = df["indice_desertification"].max()
    prix_desert = df[df["indice_desertification"] > 66]["prix_m2_moyen"].median()
    with d1:
        st.markdown(f"""<div class="kpi-card danger">
            <div class="kpi-value">{n_desert}</div>
            <div class="kpi-label">Déserts médicaux (IDM > 66)</div></div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="kpi-card danger">
            <div class="kpi-value">{idm_max:.0f}/100</div>
            <div class="kpi-label">IDM max (pire désert)</div></div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""<div class="kpi-card warning">
            <div class="kpi-value">{prix_desert:.0f} €</div>
            <div class="kpi-label">Prix m² médian des déserts</div></div>""", unsafe_allow_html=True)
    with d4:
        opp_depts = df[(df["indice_desertification"] > 50) & (df["prix_m2_moyen"] < df["prix_m2_moyen"].median())].shape[0]
        st.markdown(f"""<div class="kpi-card success">
            <div class="kpi-value">{opp_depts}</div>
            <div class="kpi-label">Zones opportunité (IDM élevé + prix bas)</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    u1c1, u1c2 = st.columns(2)
    with u1c1:
        # Scatter IDM vs Prix m² — quadrant d'opportunité
        scatter_df = df.dropna(subset=["indice_desertification", "prix_m2_moyen"])
        idm_med  = scatter_df["indice_desertification"].median()
        prix_med = scatter_df["prix_m2_moyen"].median()

        fig_idm_prix = px.scatter(
            scatter_df,
            x="prix_m2_moyen", y="indice_desertification",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            size="population_num", size_max=35,
            hover_name="Nom du département",
            hover_data={"dept": True, "prix_m2_moyen": ":.0f", "indice_desertification": ":.1f"},
            title="Opportunités d'implantation : Désertification vs Prix foncier",
            labels={
                "prix_m2_moyen": "Prix m² moyen (€)",
                "indice_desertification": "IDM (0=couvert, 100=désert)",
                "zone_short": "Zone",
            },
        )
        fig_idm_prix.add_vline(x=prix_med, line_dash="dot", line_color="gray", opacity=0.6,
                               annotation_text="Prix médian", annotation_position="top right")
        fig_idm_prix.add_hline(y=idm_med, line_dash="dot", line_color="gray", opacity=0.6,
                               annotation_text="IDM médian", annotation_position="bottom right")
        fig_idm_prix.add_annotation(
            x=prix_med * 0.55, y=idm_med * 1.3,
            text="🎯 Zone d'opportunité<br>(désert + foncier abordable)",
            showarrow=False, font=dict(color="#e74c3c", size=11),
            bgcolor="rgba(253,236,234,0.8)", bordercolor="#e74c3c", borderwidth=1
        )
        fig_idm_prix.update_layout(height=450)
        st.plotly_chart(fig_idm_prix, width="stretch")

    with u1c2:
        # Top 10 déserts médicaux
        top_desert = df.nlargest(10, "indice_desertification")[
            ["Nom du département", "indice_desertification", "prix_m2_moyen",
             "temps_acces_moyen", "med_gen_pour_100k", "zone_short"]
        ].copy()
        top_desert["indice_desertification"] = top_desert["indice_desertification"].round(1)
        fig_top_desert = px.bar(
            top_desert.sort_values("indice_desertification"),
            x="indice_desertification", y="Nom du département",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            orientation="h",
            title="Top 10 — Déserts médicaux (IDM le + élevé)",
            labels={"indice_desertification": "IDM", "zone_short": "Zone"},
            hover_data={"prix_m2_moyen": ":.0f", "temps_acces_moyen": ":.1f", "med_gen_pour_100k": ":.1f"},
        )
        fig_top_desert.update_layout(height=450)
        st.plotly_chart(fig_top_desert, width="stretch")

    # Top communes les plus éloignées des soins
    st.markdown('<div class="section-title">📍 Communes les plus éloignées des soins</div>', unsafe_allow_html=True)
    dept_codes_filter = df["dept"].tolist()
    temps_filtered = temps[temps["dept"].isin(dept_codes_filter)].copy()
    top_communes = temps_filtered.nlargest(15, "temps_acces")[
        ["commune", "dept", "temps_acces"]
    ].copy()
    top_communes = top_communes.merge(
        df[["dept", "Nom du département", "zone_short", "prix_m2_moyen"]],
        on="dept", how="left"
    )
    fig_communes = px.bar(
        top_communes.sort_values("temps_acces", ascending=False),
        x="temps_acces", y="commune",
        color="zone_short", color_discrete_map=COLOR_ZONES,
        orientation="h",
        title="Top 15 communes avec le temps d'accès le plus long",
        hover_data={"Nom du département": True, "prix_m2_moyen": ":.0f"},
        labels={"temps_acces": "Temps d'accès (min)", "commune": "Commune", "zone_short": "Zone"},
    )
    fig_communes.update_layout(height=480)
    st.plotly_chart(fig_communes, width="stretch")

    st.markdown("---")

    # ── User 3 ────────────────────────────────────────────────────────────────
    st.markdown("""<div class="user-story">
    👤 <strong>User 3 — Le Responsable Régional</strong><br>
    <em>"Je veux identifier les zones où la forte densité de seniors (65+) coïncide avec un manque
    de proximité des établissements, afin de planifier le déploiement de dispositifs mobiles
    ou de téléconsultation."</em>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">👴 Vulnérabilité Sénior (IVS)</div>', unsafe_allow_html=True)

    u3c1, u3c2 = st.columns(2)
    with u3c1:
        # TOP 7 depts vulnérables
        top_ivs = df.nlargest(7, "indice_vulnerabilite_senior")[
            ["Nom du département", "indice_vulnerabilite_senior", "pct_plus_65",
             "hopitaux_pour_100k", "temps_acces_moyen", "zone_short"]
        ].copy()
        top_ivs["Rang"] = range(1, 8)
        fig_ivs_bar = px.bar(
            top_ivs.sort_values("indice_vulnerabilite_senior"),
            x="indice_vulnerabilite_senior", y="Nom du département",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            orientation="h",
            title="Top 7 — Départements les + vulnérables (IVS)",
            labels={"indice_vulnerabilite_senior": "IVS", "zone_short": "Zone"},
            hover_data={"pct_plus_65": ":.1f", "hopitaux_pour_100k": ":.2f", "temps_acces_moyen": ":.1f"},
        )
        fig_ivs_bar.update_layout(height=350)
        st.plotly_chart(fig_ivs_bar, width="stretch")

    with u3c2:
        # Scatter pct_65 vs hopitaux_pour_100k
        scatter_s = df.dropna(subset=["pct_plus_65", "hopitaux_pour_100k"])
        p65_med  = scatter_s["pct_plus_65"].median()
        hop_med  = scatter_s["hopitaux_pour_100k"].median()

        fig_senior = px.scatter(
            scatter_s,
            x="pct_plus_65", y="hopitaux_pour_100k",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            size="population_num", size_max=30,
            hover_name="Nom du département",
            text="dept",
            title="Part des seniors vs Couverture hospitalière",
            labels={
                "pct_plus_65": "Part des +65 ans (%)",
                "hopitaux_pour_100k": "Hôpitaux / 100k hab.",
                "zone_short": "Zone",
            },
        )
        fig_senior.add_vline(x=p65_med,  line_dash="dot", line_color="gray", opacity=0.6)
        fig_senior.add_hline(y=hop_med,  line_dash="dot", line_color="gray", opacity=0.6)
        fig_senior.add_annotation(
            x=p65_med * 1.15, y=hop_med * 0.3,
            text="⚠️ Beaucoup de seniors<br>peu d'hôpitaux",
            showarrow=False, font=dict(color="#e74c3c", size=11),
            bgcolor="rgba(253,236,234,0.8)", bordercolor="#e74c3c", borderwidth=1
        )
        fig_senior.update_traces(textposition="top center", textfont_size=7)
        fig_senior.update_layout(height=350)
        st.plotly_chart(fig_senior, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – PATHOLOGIES & PRÉVENTION (User 2 + User 3 seniors)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""<div class="user-story">
    👤 <strong>User 2 — L'Expert Santé (Directeur ARS)</strong><br>
    <em>"Je veux corréler le manque de spécialistes avec les pathologies locales, afin de lancer
    des campagnes de dépistage là où elles manquent le plus."</em>
    </div>""", unsafe_allow_html=True)

    if not patho_available:
        st.info(
            "📂 **Données pathologies non chargées.**\n\n"
            "Pour activer cet onglet, renseignez `PATHO_DRIVE_ID` dans le code "
            "avec l'identifiant Google Drive de votre fichier AMELI/SNDS "
            "(colonnes : `annee; patho_niv1; dept; Ntop; Npop; prev; groupe_age…`)."
        )

    # ── Radars spécialistes (disponibles même sans données pathologies) ───────
    st.markdown('<div class="section-title">🔬 Profil spécialistes par département</div>', unsafe_allow_html=True)

    dept_for_radar = st.selectbox(
        "Département à analyser",
        options=df.sort_values("score_global")["Nom du département"].dropna().tolist(),
        key="dept_radar_patho"
    )
    row_radar = df[df["Nom du département"] == dept_for_radar].iloc[0]

    spec_cols = ["cardio_pour_100k", "ophtalmo_pour_100k", "psychiatre_pour_100k",
                 "gyneco_pour_100k", "pediatre_pour_100k"]
    spec_labels = ["Cardiologues", "Ophtalmologues", "Psychiatres", "Gynécologues", "Pédiatres"]

    # Mapping pathologie → spécialiste (pour corrélation User 2)
    PATHO_SPEC_MAP = {
        "Maladies cardiovasculaires": "cardio_pour_100k",
        "Troubles psychiatriques":    "psychiatre_pour_100k",
        "Maladies ophtalmologiques":  "ophtalmo_pour_100k",
        "Gynécologie-obstétrique":    "gyneco_pour_100k",
        "Pédiatrie":                  "pediatre_pour_100k",
    }

    pc1, pc2 = st.columns(2)
    with pc1:
        # Radar spécialistes du dept vs moyenne nationale
        dept_vals = [float(row_radar.get(c, 0) or 0) for c in spec_cols]
        nat_vals  = [float(master[c].mean()) for c in spec_cols]

        fig_spec_radar = go.Figure()
        fig_spec_radar.add_trace(go.Scatterpolar(
            r=dept_vals + dept_vals[:1], theta=spec_labels + [spec_labels[0]],
            fill="toself", name=dept_for_radar,
            line_color="#2980b9", fillcolor="rgba(41,128,185,0.25)"
        ))
        fig_spec_radar.add_trace(go.Scatterpolar(
            r=nat_vals + nat_vals[:1], theta=spec_labels + [spec_labels[0]],
            fill="toself", name="Moy. nationale",
            line_color="#7f8c8d", fillcolor="rgba(127,140,141,0.15)", line_dash="dash"
        ))
        fig_spec_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=f"Spécialistes / 100k hab. — {dept_for_radar} vs France",
            height=380,
        )
        st.plotly_chart(fig_spec_radar, width="stretch")

    with pc2:
        # Top 5 spécialités les MOINS représentées (par rapport à la moy. nationale)
        gaps = {
            lbl: max(0, nat - float(row_radar.get(col, 0) or 0))
            for lbl, col, nat in zip(spec_labels, spec_cols, nat_vals)
        }
        gap_df = pd.DataFrame({"Spécialité": list(gaps.keys()), "Manque vs moy. nat.": list(gaps.values())})
        gap_df = gap_df.sort_values("Manque vs moy. nat.", ascending=False)
        fig_gap = px.bar(
            gap_df,
            x="Manque vs moy. nat.", y="Spécialité",
            orientation="h",
            color="Manque vs moy. nat.",
            color_continuous_scale="Reds",
            title=f"Déficit de spécialistes — {dept_for_radar} (vs moy. nationale /100k)",
        )
        fig_gap.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_gap, width="stretch")

    # ── Section pathologies (si données disponibles) ──────────────────────────
    if patho_available:
        st.markdown('<div class="section-title">🧬 Pathologies — Prévalence par département</div>', unsafe_allow_html=True)

        # Filtrage patho sur les depts sélectionnés
        dept_codes_sel = df["dept"].tolist()
        patho_filt = patho_dept[patho_dept["dept"].isin(dept_codes_sel)].copy()
        patho_filt = patho_filt.merge(
            df[["dept", "Nom du département", "zone_short"]], on="dept", how="left"
        )

        all_pathos = sorted(patho_filt["patho_niv1"].dropna().unique().tolist())
        selected_patho = st.selectbox("Pathologie à analyser", all_pathos)

        pp1, pp2 = st.columns(2)
        with pp1:
            # Radar prévalence top 5 pathologies pour le dept sélectionné
            dept_patho = patho_dept[patho_dept["dept"] == row_radar["dept"]].copy()
            if not dept_patho.empty:
                top5_patho = dept_patho.nlargest(5, "prevalence_pct")
                cats_patho  = top5_patho["patho_niv1"].tolist()
                vals_patho  = top5_patho["prevalence_pct"].tolist()
                fig_patho_radar = go.Figure()
                fig_patho_radar.add_trace(go.Scatterpolar(
                    r=vals_patho + vals_patho[:1],
                    theta=cats_patho + [cats_patho[0]],
                    fill="toself", name=dept_for_radar,
                    line_color="#e74c3c", fillcolor="rgba(231,76,60,0.2)"
                ))
                fig_patho_radar.update_layout(
                    title=f"Top 5 pathologies — {dept_for_radar} (prévalence %)",
                    height=380,
                )
                st.plotly_chart(fig_patho_radar, width="stretch")
            else:
                st.info("Données pathologies non disponibles pour ce département.")

        with pp2:
            # Scatter prévalence vs spécialiste correspondant pour tous les depts
            spec_for_patho = PATHO_SPEC_MAP.get(selected_patho, "cardio_pour_100k")
            spec_lbl = next((l for l, c in zip(spec_labels, spec_cols) if c == spec_for_patho), spec_for_patho)

            patho_sel = patho_filt[patho_filt["patho_niv1"] == selected_patho].copy()
            patho_sel = patho_sel.merge(
                df[["dept", spec_for_patho, "zone_short", "Nom du département", "population_num"]],
                on="dept", how="left"
            )
            if not patho_sel.empty and spec_for_patho in patho_sel.columns:
                fig_corr_patho = px.scatter(
                    patho_sel.dropna(subset=["prevalence_pct", spec_for_patho]),
                    x="prevalence_pct", y=spec_for_patho,
                    color="zone_short", color_discrete_map=COLOR_ZONES,
                    size="population_num", size_max=30,
                    hover_name="Nom du département",
                    title=f"Prévalence {selected_patho} vs {spec_lbl} / 100k",
                    labels={
                        "prevalence_pct": f"Prévalence {selected_patho} (%)",
                        spec_for_patho: f"{spec_lbl} / 100k hab.",
                        "zone_short": "Zone",
                    },
                )
                fig_corr_patho.update_layout(height=380)
                st.plotly_chart(fig_corr_patho, width="stretch")
            else:
                st.info(f"Pas de mapping spécialiste défini pour '{selected_patho}'.")

        # ── User 3 – Pathologies seniors ──────────────────────────────────────
        if not patho_senior.empty:
            st.markdown("---")
            st.markdown("""<div class="user-story">
            👤 <strong>User 3 — Pathologies des seniors (65+)</strong><br>
            <em>Pathologies les plus prévalentes chez les 65+ dans les zones à forte vulnérabilité.</em>
            </div>""", unsafe_allow_html=True)

            top7_ivs_depts = df.nlargest(7, "indice_vulnerabilite_senior")["dept"].tolist()
            senior_filt = patho_senior[patho_senior["dept"].isin(top7_ivs_depts)].copy()
            senior_filt = senior_filt.merge(
                df[["dept", "Nom du département"]], on="dept", how="left"
            )
            if not senior_filt.empty:
                top5_senior_pathos = (
                    senior_filt.groupby("patho_niv1")["nb_patients_senior"].sum()
                    .nlargest(5).index.tolist()
                )
                senior_plot = senior_filt[senior_filt["patho_niv1"].isin(top5_senior_pathos)]
                fig_senior_patho = px.bar(
                    senior_plot,
                    x="Nom du département", y="prevalence_pct",
                    color="patho_niv1", barmode="group",
                    title="Top 5 pathologies seniors — 7 depts les + vulnérables",
                    labels={
                        "prevalence_pct": "Prévalence (%)",
                        "Nom du département": "Département",
                        "patho_niv1": "Pathologie",
                    },
                )
                fig_senior_patho.update_xaxes(tickangle=30)
                fig_senior_patho.update_layout(height=420)
                st.plotly_chart(fig_senior_patho, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – MÉDICAMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">💊 Disponibilité des Médicaments</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    nb_ruptures = int((medic["Statut"] == "Rupture de stock").sum())
    nb_tensions = int((medic["Statut"] == "Tension d'approvisionnement").sum())
    nb_arrets   = int((medic["Statut"] == "Arrêt de commercialisation").sum())
    nb_remise   = int((medic["Statut"] == "Remise à disposition").sum())
    with m1:
        st.markdown(f"""<div class="kpi-card danger"><div class="kpi-value">{nb_ruptures}</div>
            <div class="kpi-label">Ruptures de stock</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="kpi-card warning"><div class="kpi-value">{nb_tensions}</div>
            <div class="kpi-label">Tensions approvisionnement</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-card danger"><div class="kpi-value">{nb_arrets}</div>
            <div class="kpi-label">Arrêts commercialisation</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="kpi-card success"><div class="kpi-value">{nb_remise}</div>
            <div class="kpi-label">Remises à disposition</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    mc1, mc2 = st.columns(2)
    with mc1:
        fig_med_stat = px.pie(
            medic, names="Statut",
            color="Statut",
            color_discrete_map={
                "Rupture de stock":                "#e74c3c",
                "Tension d'approvisionnement":     "#f39c12",
                "Arrêt de commercialisation":      "#8e44ad",
                "Remise à disposition":            "#27ae60",
            },
            title="Répartition par statut", hole=0.4,
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
                "Rupture de stock":            "#e74c3c",
                "Tension d'approvisionnement": "#f39c12",
                "Arrêt de commercialisation":  "#8e44ad",
                "Remise à disposition":        "#27ae60",
            },
            orientation="h",
            title="Top 10 domaines médicaux touchés",
            labels={"count": "Nb médicaments", "Domaine(s) médical(aux)": ""},
        )
        st.plotly_chart(fig_dom, width="stretch")

    st.markdown('<div class="section-title">📋 Détail des médicaments en tension / rupture</div>', unsafe_allow_html=True)
    filter_statut = st.multiselect(
        "Filtrer par statut",
        options=medic["Statut"].unique().tolist(),
        default=["Rupture de stock", "Tension d'approvisionnement"]
    )
    st.dataframe(
        medic[medic["Statut"].isin(filter_statut)][[
            "Nom", "Statut", "Domaine(s) médical(aux)", "Produit(s) de santé",
            "Date de début d'incident", "Date de fin d'incident"
        ]].sort_values("Statut"),
        width="stretch", height=350
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – IMMOBILIER & SANTÉ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">🏠 Immobilier & Attractivité Territoriale</div>', unsafe_allow_html=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        fig_immo_zone = px.box(
            df.dropna(subset=["prix_m2_moyen", "zone_short"]),
            x="zone_short", y="prix_m2_moyen",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            title="Distribution des prix par zone santé",
            labels={"prix_m2_moyen": "Prix moyen (€/m²)", "zone_short": "Zone santé"},
        )
        st.plotly_chart(fig_immo_zone, width="stretch")
    with ic2:
        fig_immo_acces = px.scatter(
            df.dropna(subset=["prix_m2_moyen", "temps_acces_moyen"]),
            x="temps_acces_moyen", y="prix_m2_moyen",
            color="zone_short", color_discrete_map=COLOR_ZONES,
            hover_name="Nom du département",
            title="Prix immobilier vs Temps d'accès aux soins",
            labels={"prix_m2_moyen": "Prix (€/m²)", "temps_acces_moyen": "Temps accès (min)", "zone_short": "Zone"},
        )
        st.plotly_chart(fig_immo_acces, width="stretch")

    st.markdown('<div class="section-title">🏙️ Prix par type de bien — Top 20 départements</div>', unsafe_allow_html=True)
    immo_merged = immo_type_dept.merge(df[["dept", "zone_short"]], on="dept", how="inner")
    top_immo = immo_merged.groupby("dept")["prix_m2"].mean().nlargest(20).index
    plot_immo = immo_merged[immo_merged["dept"].isin(top_immo)]
    fig_immo_type = px.bar(
        plot_immo, x="nom_departement", y="prix_m2", color="type_local",
        barmode="group",
        title="Top 20 départements — Prix m² par type de bien",
        labels={"prix_m2": "Prix moyen (€/m²)", "nom_departement": "Département", "type_local": "Type"},
        color_discrete_sequence=["#2980b9", "#27ae60"],
    )
    fig_immo_type.update_xaxes(tickangle=45)
    st.plotly_chart(fig_immo_type, width="stretch")

    st.markdown('<div class="section-title">📈 Matrice de corrélations — Santé & Marché immobilier</div>', unsafe_allow_html=True)
    corr_cols = [
        "score_global", "prix_m2_moyen", "temps_acces_moyen", "pros_pour_100k",
        "hopitaux_pour_100k", "pct_plus_65", "enviro_score", "indice_desertification",
    ]
    corr_labels = {
        "score_global": "Score santé",
        "prix_m2_moyen": "Prix m²",
        "temps_acces_moyen": "Accès (min)",
        "pros_pour_100k": "Pros/100k",
        "hopitaux_pour_100k": "Hôp./100k",
        "pct_plus_65": "+65 ans (%)",
        "enviro_score": "Enviro.",
        "indice_desertification": "IDM",
    }
    corr_df = df[corr_cols].dropna()
    corr_matrix = corr_df.rename(columns=corr_labels).corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale="RdBu", zmin=-1, zmax=1,
        title="Matrice de corrélations (Pearson)",
        labels=dict(color="r"),
    )
    fig_corr.update_layout(height=440)
    st.plotly_chart(fig_corr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 – AIDE À LA DÉCISION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-title">🎯 Aide à la Décision — Analyse par Département</div>', unsafe_allow_html=True)
    st.markdown("""
    Synthèse complète d'un département : score santé, profil multidimensionnel,
    recommandations stratégiques et score d'opportunité d'investissement.
    """)

    dept_choice = st.selectbox(
        "Département à analyser",
        options=df.sort_values("score_global")["Nom du département"].dropna().tolist(),
        key="dept_aide_decision"
    )
    row = df[df["Nom du département"] == dept_choice].iloc[0]
    score = row["score_global"]

    if score < 33:
        badge_class, verdict, alert_class = "badge-red", "🔴 Zone Critique — Intervention prioritaire recommandée", "alert-critical"
    elif score < 66:
        badge_class, verdict, alert_class = "badge-orange", "🟡 Zone Intermédiaire — Surveillance et amélioration ciblée", "alert-warning"
    else:
        badge_class, verdict, alert_class = "badge-green", "🟢 Zone Favorable — Maintien des acquis recommandé", "alert-ok"

    st.markdown(f"""
    <div class="alert-box {alert_class}">
      <strong>{dept_choice} ({row['dept']}) — {row.get('Nom de la région','')}</strong><br>
      {verdict}<br>
      Score global : <span class="score-badge {badge_class}">{score:.1f}/100</span>
      &nbsp;|&nbsp; IDM : <strong>{row.get('indice_desertification', 0):.1f}/100</strong>
      &nbsp;|&nbsp; IVS : <strong>{row.get('indice_vulnerabilite_senior', 0):.1f}/100</strong>
    </div>
    """, unsafe_allow_html=True)

    # KPIs détaillés
    st.markdown("#### 📊 Indicateurs clés")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        ta = float(row.get("temps_acces_moyen", 0) or 0)
        lvl = "danger" if ta > 12 else ("warning" if ta > 7 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{ta:.1f} min</div>
            <div class="kpi-label">Temps accès moy.</div></div>""", unsafe_allow_html=True)
    with d2:
        pp = float(row.get("pros_pour_100k", 0) or 0)
        lvl = "danger" if pp < 200 else ("warning" if pp < 400 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{pp:.0f}</div>
            <div class="kpi-label">Pros / 100k hab.</div></div>""", unsafe_allow_html=True)
    with d3:
        hp = float(row.get("hopitaux_pour_100k", 0) or 0)
        lvl = "danger" if hp < 0.5 else ("warning" if hp < 1.5 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{hp:.2f}</div>
            <div class="kpi-label">Hôpitaux / 100k</div></div>""", unsafe_allow_html=True)
    with d4:
        pm = float(row.get("prix_m2_moyen", 0) or 0)
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{pm:.0f} €</div>
            <div class="kpi-label">Prix immo /m²</div></div>""", unsafe_allow_html=True)
    with d5:
        es = float(str(row.get("enviro_score", 10) or 10).replace(",", "."))
        lvl = "danger" if es < 7 else ("warning" if es < 12 else "success")
        st.markdown(f"""<div class="kpi-card {lvl}">
            <div class="kpi-value">{es:.1f}/20</div>
            <div class="kpi-label">Enviro. (régional)</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    ra1, ra2 = st.columns([1, 1])
    with ra1:
        categories = ["Accès aux soins", "Professionnels", "Établissements", "Environnement"]
        vals_dept = [
            float(row.get("score_acces", 0) or 0),
            float(row.get("score_pros",  0) or 0),
            float(row.get("score_etabs", 0) or 0),
            float(row.get("score_env",   0) or 0),
        ]
        vals_nat = [
            float(master["score_acces"].mean()),
            float(master["score_pros"].mean()),
            float(master["score_etabs"].mean()),
            float(master["score_env"].mean()),
        ]
        fig_radar_dept = go.Figure()
        fig_radar_dept.add_trace(go.Scatterpolar(
            r=vals_dept + vals_dept[:1], theta=categories + [categories[0]],
            fill="toself", name=dept_choice,
            line_color="#2980b9", fillcolor="rgba(41,128,185,0.25)"
        ))
        fig_radar_dept.add_trace(go.Scatterpolar(
            r=vals_nat + vals_nat[:1], theta=categories + [categories[0]],
            fill="toself", name="Moy. nationale",
            line_color="#7f8c8d", fillcolor="rgba(127,140,141,0.15)", line_dash="dash"
        ))
        fig_radar_dept.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title=f"Profil santé — {dept_choice} vs France",
            height=380,
        )
        st.plotly_chart(fig_radar_dept, width="stretch")

    with ra2:
        # Recommandations
        st.markdown("#### 💡 Recommandations stratégiques")
        reco_list = []
        if ta > 12:
            reco_list.append(("🔴", "Accès aux soins critique", "Renforcer les maisons de santé pluridisciplinaires et les dispositifs de télémédecine."))
        elif ta > 7:
            reco_list.append(("🟡", "Accès aux soins à surveiller", "Envisager des consultations avancées et des transports sanitaires renforcés."))
        else:
            reco_list.append(("🟢", "Bon accès aux soins", "Maintenir les structures existantes."))

        if pp < 200:
            reco_list.append(("🔴", "Désert médical", "Incitations fiscales et bourses pour l'installation de médecins généralistes."))
        elif pp < 400:
            reco_list.append(("🟡", "Densité médicale insuffisante", "Renforcer les partenariats avec les facultés de médecine locales."))
        else:
            reco_list.append(("🟢", "Bonne densité médicale", "Encourager la spécialisation et la formation continue."))

        if es < 7:
            reco_list.append(("🔴", "Risque environnemental élevé", "Audit sanitaire recommandé (air, eau, sols). Score régional /20."))
        elif es < 12:
            reco_list.append(("🟡", "Environnement à surveiller", "Suivi des indicateurs de pollution et prévention ciblée. Score régional /20."))
        else:
            reco_list.append(("🟢", "Bon environnement santé", "Valoriser cet atout dans la communication territoriale. Score régional /20."))

        ivs = float(row.get("indice_vulnerabilite_senior", 0) or 0)
        if ivs > 66:
            reco_list.append(("🔴", "Forte vulnérabilité sénior", "Prioriser le déploiement de dispositifs mobiles et de téléconsultation pour les 65+."))
        elif ivs > 33:
            reco_list.append(("🟡", "Vigilance séniors", "Renforcer les services de proximité pour personnes âgées."))

        for icon, title, text in reco_list:
            cls = "alert-critical" if icon == "🔴" else ("alert-warning" if icon == "🟡" else "alert-ok")
            st.markdown(f"""<div class="alert-box {cls}"><strong>{icon} {title}</strong><br>{text}</div>""", unsafe_allow_html=True)

        # Score d'opportunité d'investissement — formule scientifique basée sur IDM + IVS + foncier
        st.markdown("#### 🏗️ Score d'opportunité d'investissement")
        idm_score = float(row.get("indice_desertification", 50) or 50)
        ivs_score = float(row.get("indice_vulnerabilite_senior", 20) or 20)
        prix_val  = float(row.get("prix_m2_moyen", 3000) or 3000)
        # Normalise prix : 500€/m² = très accessible (100), 8000€/m² = inaccessible (0)
        prix_norm = max(0.0, min(100.0, 100 - (prix_val - 500) / (8000 - 500) * 100))
        opp_score = min(100, max(0,
            idm_score  * 0.50 +   # besoin médical
            ivs_score  * 0.30 +   # vulnérabilité de la population
            prix_norm  * 0.20     # accessibilité foncière
        ))
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
                    {"range": [0,  35], "color": "#eafaf1"},
                    {"range": [35, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fdecea"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": opp_score},
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

    st.markdown("---")
    st.markdown("#### 🏆 Top 10 priorités d'intervention")
    priority_df = df.sort_values("score_global").head(10)[[
        "dept", "Nom du département", "Nom de la région", "zone",
        "score_global", "indice_desertification", "indice_vulnerabilite_senior",
        "temps_acces_moyen", "pros_pour_100k", "enviro_score"
    ]].copy()
    priority_df["Rang"] = range(1, len(priority_df) + 1)
    priority_df = priority_df.rename(columns={
        "dept": "Code", "Nom du département": "Département", "Nom de la région": "Région",
        "zone": "Zone", "score_global": "Score /100",
        "indice_desertification": "IDM", "indice_vulnerabilite_senior": "IVS",
        "temps_acces_moyen": "Accès (min)", "pros_pour_100k": "Pros/100k",
        "enviro_score": "Enviro. (rég./20)"
    })
    for col in ["Score /100", "IDM", "IVS", "Accès (min)", "Pros/100k", "Enviro. (rég./20)"]:
        if col in priority_df.columns:
            priority_df[col] = pd.to_numeric(priority_df[col], errors="coerce").round(1)
    st.dataframe(priority_df.set_index("Rang"), width="stretch")


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#95a5a6; font-size:0.8rem;'>"
    "Dashboard Santé & Territoires · INSEE · RPPS · FINESS · DVF · ANSM · AMELI/SNDS · "
    "Aide à la décision territoriale · 3 profils utilisateurs"
    "</div>",
    unsafe_allow_html=True
)
