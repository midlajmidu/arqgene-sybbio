"""
app.py — SynB Metabolic Engineering Platform · Step 1 (API-connected frontend)
===============================================================================
Streamlit UI that delegates ALL scientific computation to the FastAPI backend
running at localhost:8001.  No COBRApy imports in this file.

Run order:
    1. Terminal A:  uvicorn backend.main:app --port 8001 --reload
    2. Terminal B:  streamlit run app.py
"""

from __future__ import annotations

import datetime
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------------
# Backend base URL — editable via sidebar
# ------------------------------------------------------------------
_DEFAULT_BACKEND = "http://localhost:8001/api/v1"

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="SynB · Metabolic Engineering Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Shared CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    :root {
        --primary: #4f8ef7;
        --primary-glow: rgba(79,142,247,0.25);
        --accent: #00d2c1;
        --warn: #f5a623;
        --error: #e74c3c;
        --success: #27ae60;
        --bg-card: rgba(22,27,46,0.85);
        --border: rgba(79,142,247,0.18);
        --text-muted: #8892a4;
    }
    .stApp { background: linear-gradient(135deg,#0a0e1a 0%,#111827 50%,#0d1529 100%); background-attachment: fixed; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0f1423 0%,#131d35 100%); border-right:1px solid var(--border); }
    div[data-testid="stMetric"] {
        background: var(--bg-card); border:1px solid var(--border); border-radius:12px;
        padding:16px 20px; backdrop-filter:blur(12px);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover { border-color:var(--primary); box-shadow:0 0 20px var(--primary-glow); }
    .stDataFrame { border:1px solid var(--border); border-radius:10px; overflow:hidden; }
    .stTabs [data-baseweb="tab-list"] {
        gap:6px; background:rgba(22,27,46,0.6); border-radius:12px;
        padding:6px; border:1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] { border-radius:8px; font-weight:500; color:var(--text-muted); }
    .stTabs [aria-selected="true"] { background:var(--primary)!important; color:white!important; }
    .stAlert { border-radius:10px; border-left-width:4px; }
    .stButton > button {
        background: linear-gradient(135deg,var(--primary),#3b6ed4); color:white;
        border:none; border-radius:8px; font-weight:600; letter-spacing:0.02em;
        transition:all 0.2s; box-shadow:0 4px 15px var(--primary-glow);
    }
    .stButton > button:hover { transform:translateY(-1px); box-shadow:0 8px 25px var(--primary-glow); }
    .hero-header {
        background: linear-gradient(135deg,rgba(79,142,247,0.15) 0%,rgba(0,210,193,0.08) 100%);
        border:1px solid var(--border); border-radius:16px; padding:28px 32px;
        margin-bottom:28px; backdrop-filter:blur(10px);
    }
    .hero-title {
        font-size:2rem; font-weight:700;
        background:linear-gradient(90deg,#4f8ef7,#00d2c1);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; margin:0;
    }
    .hero-subtitle { color:var(--text-muted); font-size:0.95rem; margin-top:6px; }
    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; }
    .badge-ok   { background:rgba(39,174,96,0.2);  color:#2ecc71; border:1px solid rgba(39,174,96,0.4); }
    .badge-warn { background:rgba(245,166,35,0.2); color:#f5a623; border:1px solid rgba(245,166,35,0.4); }
    .badge-err  { background:rgba(231,76,60,0.2);  color:#e74c3c; border:1px solid rgba(231,76,60,0.4); }
    .section-heading {
        font-size:1.1rem; font-weight:600; color:#c8d5f0;
        border-left:3px solid var(--primary); padding-left:12px; margin:20px 0 12px;
    }
    code { font-family:'JetBrains Mono',monospace; }
    [data-testid="stFileUploader"] { border:2px dashed var(--border); border-radius:14px; padding:12px; }
    [data-testid="stFileUploader"]:hover { border-color:var(--primary); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# Session state
# ==================================================================

def _init_session() -> None:
    defaults: Dict[str, Any] = {
        "model_id": None,       # UUID returned by backend
        "model_summary": None,  # dict from /upload-model
        "fba_result": None,
        "pfba_result": None,
        "validation_result": None,
        "backend_url": _DEFAULT_BACKEND,
        "solver": "glpk",
        "feasibility_tol": 1e-7,
        "optimality_tol": 1e-7,
        "backend_ok": None,
        "rxn_page": 1,
        "rxn_page_size": 25,
        "rxn_search": "",
        "rxn_subsystem": "All",
        # FVA
        "fva_result": None,
        "fva_error": None,
        "fva_confirmed": False,
        # Production optimization
        "objective_info": None,
        "envelope_result": None,
        "envelope_error": None,
        "prod_reactions": None,   # cached list of reaction IDs for dropdowns
        # Medium configuration
        "medium_data": None,       # raw MediumResponse dict from GET /medium
        "medium_success": None,    # last success banner message
        # OptKnock strain design
        "optknock_result": None,   # last OptKnockResponse dict
        "optknock_error": None,    # last error message string
        # Genome reconstruction
        "genome_job_id": None,     # current genome reconstruction job ID
        "genome_report": None,     # last reconstruction report dict
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ==================================================================
# HTTP helpers
# ==================================================================

def _base_url() -> str:
    return st.session_state.backend_url.rstrip("/")


def _get(path: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """HTTP GET with timeout and error handling."""
    try:
        r = requests.get(f"{_base_url()}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend. Is it running on the configured port?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out. The computation may be taking too long.")
        return None
    except Exception as exc:
        st.error(f"❌ API error: {exc}")
        return None


def _post_json(path: str, payload: Dict) -> Optional[Dict]:
    """HTTP POST with JSON body."""
    try:
        r = requests.post(f"{_base_url()}{path}", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend. Is it running on the configured port?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out (model may be very large).")
        return None
    except Exception as exc:
        st.error(f"❌ API error: {exc}")
        return None


def _post_json_raw(path: str, payload: Dict):
    """
    HTTP POST returning (status_code, json_body) for routes that use non-2xx
    codes to signal structured domain errors (e.g. MODEL_TOO_LARGE → 400).
    Returns (None, None) on network failure.
    """
    try:
        r = requests.post(f"{_base_url()}{path}", json=payload, timeout=660)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend.")
        return None, None
    except requests.exceptions.Timeout:
        return 408, {"error": "CLIENT_TIMEOUT", "message": "Request timed out on the client side."}
    except Exception as exc:
        st.error(f"❌ API error: {exc}")
        return None, None


def _post_file(path: str, file_bytes: bytes, filename: str, extra_data: Dict) -> Optional[Dict]:
    """Multipart file upload."""
    try:
        files = {"file": (filename, file_bytes, "application/xml")}
        r = requests.post(
            f"{_base_url()}{path}",
            files=files,
            data=extra_data,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend. Is it running?")
        return None
    except Exception as exc:
        st.error(f"❌ Upload error: {exc}")
        return None


def _check_backend_health() -> bool:
    """Ping /health and return True if backend is reachable."""
    try:
        r = requests.get(f"{_base_url()}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ==================================================================
# Sidebar
# ==================================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding:16px 0 8px;'>
                <span style='font-size:2.5rem;'>🧬</span><br>
                <span style='font-weight:700; font-size:1.1rem; color:#4f8ef7;'>SynB</span><br>
                <span style='font-size:0.75rem; color:#8892a4;'>Metabolic Engineering Platform</span><br>
                <span style='font-size:0.68rem; color:#4f5a72; background:rgba(79,142,247,0.1);
                       padding:2px 8px; border-radius:10px; display:inline-block; margin-top:4px;'>
                    Step 1 · API-Connected
                </span>
            </div>
            <hr style='border-color:rgba(79,142,247,0.15); margin:12px 0;'>
            """,
            unsafe_allow_html=True,
        )

        # ---- Backend connection ----
        st.markdown("#### 🔌 Backend Connection")
        backend_url = st.text_input(
            "API Base URL",
            value=st.session_state.backend_url,
            placeholder="http://localhost:8000/api/v1",
        )
        if backend_url != st.session_state.backend_url:
            st.session_state.backend_url = backend_url
            st.session_state.backend_ok = None

        if st.button("🔗 Test Connection", use_container_width=True):
            ok = _check_backend_health()
            st.session_state.backend_ok = ok

        if st.session_state.backend_ok is True:
            st.markdown("<span class='badge badge-ok'>● CONNECTED</span>", unsafe_allow_html=True)
        elif st.session_state.backend_ok is False:
            st.markdown("<span class='badge badge-err'>● UNREACHABLE</span>", unsafe_allow_html=True)
            st.caption("Start the backend: `uvicorn backend.main:app --port 8001 --reload`")

        # ---- Solver config ----
        st.markdown("#### ⚙️ Solver Configuration")
        st.session_state.solver = st.selectbox(
            "LP Solver",
            options=["glpk", "cplex", "gurobi", "highs"],
            index=["glpk", "cplex", "gurobi", "highs"].index(
                st.session_state.solver if st.session_state.solver in ["glpk","cplex","gurobi","highs"] else "glpk"
            ),
            help="GLPK is always available. Others require separate installation.",
        )

        st.markdown("#### 🔢 Numerical Tolerances")
        st.session_state.feasibility_tol = st.select_slider(
            "Feasibility Tolerance",
            options=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            value=st.session_state.feasibility_tol,
            format_func=lambda x: f"{x:.0e}",
        )
        st.session_state.optimality_tol = st.select_slider(
            "Optimality Tolerance",
            options=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            value=st.session_state.optimality_tol,
            format_func=lambda x: f"{x:.0e}",
        )

        # ---- Actions ----
        if st.session_state.model_id:
            st.markdown("<hr style='border-color:rgba(79,142,247,0.15);'>", unsafe_allow_html=True)
            st.markdown("#### 🚀 Actions")

            if st.button("▶ Run FBA", use_container_width=True):
                _action_run_fba()

            if st.button("⚡ Run pFBA", use_container_width=True):
                _action_run_pfba()

            if st.button("🔍 Run Validation", use_container_width=True):
                _action_run_validation()

            if st.button("📊 Run FVA", use_container_width=True):
                st.session_state.fva_result = None
                st.session_state.fva_error = None
                st.session_state.fva_confirmed = False
                st.session_state["_fva_trigger"] = True

            if st.button("🏭 Production", use_container_width=True):
                st.session_state["_prod_trigger"] = True

            if st.button("🧬 Medium", use_container_width=True):
                st.session_state.medium_data = None   # force refresh
                st.session_state["_medium_trigger"] = True

            st.markdown("")
            if st.button("🧬 Strain Design", use_container_width=True):
                st.session_state["_optknock_trigger"] = True

            st.markdown("")
            if st.button("🗑 Clear Model", use_container_width=True):
                _action_clear_model()

        # ---- Reproducibility footer ----
        st.markdown("<hr style='border-color:rgba(79,142,247,0.15);'>", unsafe_allow_html=True)
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st.markdown(
            f"""
            <div style='font-size:0.72rem; color:#4f5a72; line-height:1.8;'>
                🕐 <b>Timestamp:</b> {ts}<br>
                🐍 <b>Python:</b> {py_ver}<br>
                ⚡ <b>Solver:</b> {st.session_state.solver}<br>
                🆔 <b>Model UUID:</b><br>
                <code style='font-size:0.62rem; word-break:break-all;'>
                    {st.session_state.model_id or 'none'}
                </code>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================================
# Sidebar action helpers
# ==================================================================

def _action_run_fba() -> None:
    with st.spinner("🔬 Running FBA via backend…"):
        result = _post_json("/run-fba", {
            "model_id": st.session_state.model_id,
            "solver": st.session_state.solver,
            "feasibility_tol": st.session_state.feasibility_tol,
            "optimality_tol": st.session_state.optimality_tol,
        })
    if result:
        st.session_state.fba_result = result
        st.session_state.analysis_latest_source = "FBA"
        if result.get("success"):
            st.success(f"FBA complete — growth rate: {result.get('growth_rate', 0):.6f}", icon="✅")
        else:
            st.warning(result.get("message", "FBA returned non-optimal."), icon="⚠️")


def _action_run_pfba() -> None:
    with st.spinner("⚡ Running pFBA via backend…"):
        result = _post_json("/run-pfba", {
            "model_id": st.session_state.model_id,
            "solver": st.session_state.solver,
            "feasibility_tol": st.session_state.feasibility_tol,
            "optimality_tol": st.session_state.optimality_tol,
        })
    if result:
        st.session_state.pfba_result = result
        st.session_state.analysis_latest_source = "pFBA"
        if result.get("success"):
            st.success(f"pFBA complete — total flux: {result.get('total_absolute_flux', 0):.4f}", icon="✅")
        else:
            st.warning(result.get("message", "pFBA returned non-optimal."), icon="⚠️")


def _action_run_validation() -> None:
    with st.spinner("🔍 Running validation suite via backend…"):
        result = _post_json("/validate-model", {
            "model_id": st.session_state.model_id,
            "run_fva": True,
        })
    if result:
        st.session_state.validation_result = result
        if result.get("success"):
            st.success("Validation complete.", icon="✅")
        else:
            st.warning(result.get("message", "Validation issues found."), icon="⚠️")


def _action_clear_model() -> None:
    if st.session_state.model_id:
        _get(f"/models/{st.session_state.model_id}")  # inform backend (fire-and-forget)
    for k in ["model_id", "model_summary", "fba_result", "pfba_result", "validation_result", "analysis_latest_source"]:
        st.session_state[k] = None
    st.rerun()


# ==================================================================
# Upload section
# ==================================================================

def render_upload_section() -> None:
    st.markdown(
        """
        <div class='hero-header'>
            <h1 class='hero-title'>🧬 SynB · Metabolic Model Diagnostics</h1>
            <p class='hero-subtitle'>
                Step 1 — Upload your SBML model, inspect network topology,
                run baseline FBA and validate model integrity.
                <span style='color:#4f8ef7;'>All computation runs on the FastAPI backend.</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Backend health check on first load
    if st.session_state.backend_ok is None:
        st.session_state.backend_ok = _check_backend_health()

    if not st.session_state.backend_ok:
        st.error(
            "⚠️ **Backend unreachable.** Start it in a separate terminal:\n\n"
            "```bash\nuvicorn backend.main:app --port 8001 --reload\n```",
            icon="🚨",
        )

    if st.session_state.model_id is None:
        upload_mode = st.radio(
            "Input type",
            ["📄 SBML Model (.xml)", "🧬 Genome / Proteome (.fna / .faa)"],
            horizontal=True, key="upload_mode",
            label_visibility="collapsed",
        )

        if upload_mode.startswith("📄"):
            uploaded = st.file_uploader(
                "📂 Upload SBML Model (.xml)",
                type=["xml"],
                help="Genome-scale metabolic model in SBML format.",
            )
            if uploaded is not None:
                _handle_upload(uploaded)
            else:
                st.info(
                    "👆 Upload an SBML file to begin. "
                    "Test models: **E. coli** (*iJO1366.xml*), **S. cerevisiae** (*iMM904.xml*).",
                    icon="🧪",
                )
        else:
            uploaded_genome = st.file_uploader(
                "🧬 Upload Genome / Proteome",
                type=["fna", "faa", "fa", "fasta", "pep", "ffn"],
                help="FASTA file: .fna (nucleotide) or .faa (protein sequences).",
            )
            if uploaded_genome is not None:
                _handle_genome_upload(uploaded_genome)
            elif st.session_state.genome_job_id:
                _poll_genome_job()
            else:
                st.info(
                    "🧬 Upload a **FASTA proteome** (.faa) or **genome** (.fna) file.\n\n"
                    "The system will:\n"
                    "1. Annotate genes using **KEGG** enzyme databases\n"
                    "2. Map EC numbers to metabolic reactions\n"
                    "3. Build a **COBRApy** stoichiometric model\n"
                    "4. Gap-fill and add biomass reaction\n"
                    "5. Export as **SBML** and register for simulation",
                    icon="🔬",
                )
    else:
        summary = st.session_state.model_summary or {}
        st.success(
            f"✅ Model **{summary.get('internal_id', 'loaded')}** is active — "
            f"{summary.get('num_reactions', 0):,} reactions · "
            f"{summary.get('num_metabolites', 0):,} metabolites · "
            f"{summary.get('num_genes', 0):,} genes",
            icon="🧬",
        )


def _handle_upload(uploaded_file) -> None:
    with st.spinner("📡 Sending model to backend for parsing…"):
        result = _post_file(
            "/upload-model",
            file_bytes=uploaded_file.read(),
            filename=uploaded_file.name,
            extra_data={
                "solver": st.session_state.solver,
                "feasibility_tol": str(st.session_state.feasibility_tol),
                "optimality_tol": str(st.session_state.optimality_tol),
            },
        )
    if result:
        if result.get("success"):
            st.session_state.model_id = result["model_id"]
            st.session_state.model_summary = result
            st.success(
                f"✅ Model **{result.get('internal_id')}** uploaded and registered. "
                f"UUID: `{result['model_id']}`",
                icon="✅",
            )
            st.rerun()
        else:
            err = result.get("error", {})
            st.error(
                f"❌ **Upload failed** [{err.get('code', 'ERROR')}]\n\n{err.get('message', result.get('message', ''))}",
                icon="🚨",
            )


# ==================================================================
# Tab: Summary
# ==================================================================

def render_summary_tab() -> None:
    summary = st.session_state.model_summary or {}

    st.markdown("<div class='section-heading'>📊 Model Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔗 Reactions",   f"{summary.get('num_reactions', 0):,}")
    c2.metric("⚗️ Metabolites", f"{summary.get('num_metabolites', 0):,}")
    c3.metric("🧬 Genes",       f"{summary.get('num_genes', 0):,}")
    c4.metric("🏢 Compartments",summary.get('num_compartments', 0))

    st.markdown("")
    c5, c6, c7 = st.columns(3)
    c5.metric("📥 Exchange Rxns", summary.get('exchange_reactions', 0))
    c6.metric("💧 Demand Rxns",   summary.get('demand_reactions', 0))
    c7.metric("🚰 Sink Rxns",     summary.get('sink_reactions', 0))

    # ---- Objective ----
    st.markdown("<div class='section-heading'>🎯 Objective Function</div>", unsafe_allow_html=True)
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown(
            f"<div style='background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);"
            f"border-radius:10px;padding:16px;'>"
            f"<div style='font-size:0.8rem;color:#8892a4;margin-bottom:4px;'>REACTION</div>"
            f"<div style='font-weight:600;color:#c8d5f0;font-family:JetBrains Mono;'>"
            f"{summary.get('objective_reaction', '—')}</div></div>",
            unsafe_allow_html=True,
        )
    with oc2:
        d = summary.get('objective_direction', 'max')
        col = "#27ae60" if d == "max" else "#e74c3c"
        st.markdown(
            f"<div style='background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);"
            f"border-radius:10px;padding:16px;'>"
            f"<div style='font-size:0.8rem;color:#8892a4;margin-bottom:4px;'>DIRECTION</div>"
            f"<div style='font-weight:700;color:{col};font-size:1.1rem;'>{str(d).upper()}</div></div>",
            unsafe_allow_html=True,
        )
    with oc3:
        st.markdown(
            f"<div style='background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);"
            f"border-radius:10px;padding:16px;'>"
            f"<div style='font-size:0.8rem;color:#8892a4;margin-bottom:4px;'>MODEL ID</div>"
            f"<div style='font-weight:600;color:#c8d5f0;font-family:JetBrains Mono;'>"
            f"{summary.get('internal_id', '—')}</div></div>",
            unsafe_allow_html=True,
        )

    # ---- Compartments ----
    compartments = summary.get("compartments", [])
    if compartments:
        st.markdown("<div class='section-heading'>🏗️ Compartments</div>", unsafe_allow_html=True)
        df_comp = pd.DataFrame([{"ID": c["compartment_id"], "Description": c["description"]} for c in compartments])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # ---- Paginated reaction browser (via backend API) ----
    _render_reaction_browser_api()


def _render_reaction_browser_api() -> None:
    """Render the paginated reaction browser using backend /reactions/{id} endpoint."""
    st.markdown("<div class='section-heading'>⚡ All Reactions — Paginated Browser</div>", unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1])
    with ctrl1:
        search_q = st.text_input(
            "Search", value=st.session_state.rxn_search,
            placeholder="Filter by ID, name, subsystem, GPR…",
            key="rxn_search_input", label_visibility="collapsed",
        )
        if search_q != st.session_state.rxn_search:
            st.session_state.rxn_search = search_q
            st.session_state.rxn_page = 1

    with ctrl2:
        subsystem_input = st.text_input(
            "Subsystem filter", value=("" if st.session_state.rxn_subsystem == "All" else st.session_state.rxn_subsystem),
            placeholder="Subsystem exact match…",
            key="rxn_sub_input", label_visibility="collapsed",
        )
        new_sub = subsystem_input.strip() or "All"
        if new_sub != st.session_state.rxn_subsystem:
            st.session_state.rxn_subsystem = new_sub
            st.session_state.rxn_page = 1

    with ctrl3:
        page_size = st.selectbox(
            "Per page", options=[10, 25, 50, 100],
            index=[10, 25, 50, 100].index(
                st.session_state.rxn_page_size if st.session_state.rxn_page_size in [10, 25, 50, 100] else 25
            ),
            key="rxn_ps", label_visibility="collapsed",
        )
        if page_size != st.session_state.rxn_page_size:
            st.session_state.rxn_page_size = page_size
            st.session_state.rxn_page = 1

    # Fetch from backend
    with st.spinner("Loading reactions…"):
        params: Dict[str, Any] = {
            "page": st.session_state.rxn_page,
            "page_size": st.session_state.rxn_page_size,
        }
        if st.session_state.rxn_search.strip():
            params["search"] = st.session_state.rxn_search.strip()
        if st.session_state.rxn_subsystem != "All":
            params["subsystem"] = st.session_state.rxn_subsystem

        data = _get(f"/reactions/{st.session_state.model_id}", params=params)

    if not data or not data.get("success"):
        st.warning("Could not load reactions. Is the backend running?", icon="⚠️")
        return

    total = data.get("total", 0)
    total_pages = data.get("total_pages", 1)
    current_page = data.get("page", 1)
    page_size_actual = data.get("page_size", 25)

    # ---- Info bar ----
    st.markdown(
        f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    background:rgba(22,27,46,0.6);border:1px solid rgba(79,142,247,0.15);
                    border-radius:8px;padding:8px 16px;margin:8px 0;'>
            <span style='color:#8892a4;font-size:0.83rem;'>
                Showing <b style='color:#c8d5f0;'>{total:,}</b> reaction(s)
            </span>
            <span style='color:#4f5a72;font-size:0.8rem;'>
                Page <b style='color:#c8d5f0;'>{current_page}</b> of
                <b style='color:#c8d5f0;'>{total_pages}</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Table ----
    reactions = data.get("reactions", [])
    if reactions:
        start_row = (current_page - 1) * page_size_actual + 1
        df = pd.DataFrame(reactions)
        df.index = range(start_row, start_row + len(df))
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("Reaction ID", width="medium"),
                "name": st.column_config.TextColumn("Name", width="large"),
                "subsystem": st.column_config.TextColumn("Subsystem"),
                "lower_bound": st.column_config.NumberColumn("LB", format="%.2f"),
                "upper_bound": st.column_config.NumberColumn("UB", format="%.2f"),
                "gpr": st.column_config.TextColumn("GPR Rule", width="large"),
                "formula": st.column_config.TextColumn("Formula", width="large"),
                "num_metabolites": st.column_config.NumberColumn("Mets", width="small"),
                "num_genes": st.column_config.NumberColumn("Genes", width="small"),
                "is_boundary": st.column_config.CheckboxColumn("Boundary?", width="small"),
            },
        )
    else:
        st.info("No reactions match the current filters.", icon="🔎")

    # ---- Pagination controls ----
    st.markdown("")
    pl, pml, pm, pmr, pr = st.columns([1, 1, 2, 1, 1])
    with pl:
        if st.button("⏮ First", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.rxn_page = 1; st.rerun()
    with pml:
        if st.button("◀ Prev", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.rxn_page = current_page - 1; st.rerun()
    with pm:
        jump = st.number_input("Page", min_value=1, max_value=total_pages,
                               value=current_page, step=1,
                               key="rxn_jump", label_visibility="collapsed")
        if int(jump) != current_page:
            st.session_state.rxn_page = int(jump); st.rerun()
    with pmr:
        if st.button("Next ▶", use_container_width=True, disabled=(current_page >= total_pages)):
            st.session_state.rxn_page = current_page + 1; st.rerun()
    with pr:
        if st.button("Last ⏭", use_container_width=True, disabled=(current_page >= total_pages)):
            st.session_state.rxn_page = total_pages; st.rerun()

    start_row2 = (current_page - 1) * page_size_actual + 1
    end_row = min(current_page * page_size_actual, total)
    st.markdown(
        f"<div style='text-align:center;color:#4f5a72;font-size:0.75rem;margin-top:4px;'>"
        f"Rows {start_row2}–{end_row} of {total:,}</div>",
        unsafe_allow_html=True,
    )


# ==================================================================
# Tab: Diagnostics
# ==================================================================

def render_diagnostics_tab() -> None:
    fba  = st.session_state.fba_result
    pfba = st.session_state.pfba_result
    summary = st.session_state.model_summary or {}
    latest_src = st.session_state.get("analysis_latest_source")

    if fba is None and pfba is None:
        st.info("Click **▶ Run FBA** or **⚡ Run pFBA** in the sidebar.", icon="▶️")
        return

    st.markdown("<div class='section-heading'>📈 Flux Balance Analysis Results</div>", unsafe_allow_html=True)

    # ─── 1️⃣ Solution Selection (Rule 1) ──────────────────────────────────────
    # Priority logic: pFBA > FBA if both available and pFBA run last
    active_solution = None
    source_name = "None"
    
    if latest_src == "pFBA" and pfba and pfba.get("success"):
        active_solution = pfba
        source_name = "pFBA"
    elif fba and fba.get("success"):
        active_solution = fba
        source_name = "FBA"
    elif pfba and pfba.get("success"):
        active_solution = pfba
        source_name = "pFBA"
    else:
        # Fallback to failed solution for error display
        active_solution = fba or pfba
        source_name = "FBA (Failed)" if fba else "pFBA (Failed)"

    # ─── 2️⃣ Optimization Summary Panel ────────────────────────────────────────
    obj_rxn_id = summary.get("objective_reaction", "")
    
    def _classify_objective(rxn_id: str) -> str:
        if "biomass" in rxn_id.lower(): return "Growth"
        if rxn_id.startswith("EX_"): return "Product Formation"
        return "Custom Objective"

    obj_class = _classify_objective(obj_rxn_id)
    status_raw = active_solution.get("solver_status", "unknown")
    status_color = "#27ae60" if status_raw == "optimal" else "#e74c3c"
    
    st.markdown(f"""
    <div style='background:rgba(22,27,46,0.85); border:1px solid rgba(79,142,247,0.18);
                border-radius:14px; padding:22px 26px; margin-bottom:18px; line-height:2.0;'>
        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
            <div>
                <div style='display:flex; gap:12px; align-items:center; margin-bottom:6px;'>
                    <span style='color:#8892a4; width:180px; font-size:0.9rem;'>Objective Reaction:</span>
                    <span style='color:#c8d5f0; font-family:JetBrains Mono; font-weight:600;'>{obj_rxn_id or "—"}</span>
                </div>
                <div style='display:flex; gap:12px; align-items:center; margin-bottom:6px;'>
                    <span style='color:#8892a4; width:180px; font-size:0.9rem;'>Objective Type:</span>
                    <span style='color:#4f8ef7; font-weight:700;'>{obj_class}</span>
                </div>
                <div style='display:flex; gap:12px; align-items:center; margin-bottom:6px;'>
                    <span style='color:#8892a4; width:180px; font-size:0.9rem;'>Solver:</span>
                    <span style='color:#c8d5f0;'>{active_solution.get("solver_name", "—").upper()} (via Optlang)</span>
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:0.75rem; color:#8892a4; margin-bottom:4px;'>EXCHANGE FLUX SOURCE</div>
                <div style='font-weight:700; color:#AEC6FF;'>{source_name}</div>
                <div style='margin-top:8px; font-weight:700; color:{status_color}; font-family:JetBrains Mono;'>{status_raw.upper()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── 3️⃣ Objective Result Panel (Rule 2 + 8) ────────────────────────────────
    if active_solution.get("success"):
        obj_val = active_solution.get("objective_value", 0.0)
        
        if obj_class == "Growth":
            label = "Growth Rate"
            unit = "hr⁻¹"
        elif obj_class == "Product Formation":
            prod_name = obj_rxn_id.replace("EX_", "").replace("_e", "").replace("_", " ").title()
            label = f"{prod_name} Production Rate"
            unit = "mmol · gDW⁻¹ · h⁻¹"
        else:
            label = "Objective Value"
            unit = "mmol · gDW⁻¹ · h⁻¹"

        st.markdown(f"""
        <div style='background:rgba(79,142,247,0.07); border:1px solid rgba(79,142,247,0.18);
                    border-radius:14px; padding:22px 26px; margin-bottom:18px; text-align:center;'>
            <div style='font-size:0.9rem; color:#8892a4; margin-bottom:6px;'>📈 {label}</div>
            <div style='font-size:2.4rem; font-weight:800; color:#4f8ef7; font-family:JetBrains Mono;'>{obj_val:.6f}</div>
            <div style='font-size:0.82rem; color:#4f5a72; margin-top:4px;'>{unit}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── 4️⃣ Exchange Flux Summary (Rule 2–7, 9) ───────────────────────────────
    st.markdown("<div class='section-heading'>📥 Exchange Flux Summary</div>", unsafe_allow_html=True)
    
    ex_fluxes = active_solution.get("exchange_fluxes", [])
    if ex_fluxes:
        df_ex = pd.DataFrame(ex_fluxes)
        df_ex.columns = ["Metabolite", "Reaction ID", "Flux", "Direction"]
        
        # Formatting (Rule 6)
        df_ex["Flux"] = df_ex["Flux"].map(lambda x: f"{x:.4f}")
        
        # Tooltip (Rule 9)
        st.info("💡 **Scientific Convention:** Negative flux indicates substrate **Uptake**. Positive flux indicates **Secretion** into the environment.", icon="💡")

        def _style_exchange(row):
            is_uptake = row["Direction"] == "Uptake"
            color = "rgba(231,76,60,0.1)" if is_uptake else "rgba(39,174,96,0.1)"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_ex.style.apply(_style_exchange, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No active exchange fluxes under current medium conditions (Rule 7).", icon="⚠️")

    # ─── 5️⃣ Top Flux-Carrying Reactions (Rule 4) ──────────────────────────────
    st.markdown("<div class='section-heading'>🏆 Top Flux-Carrying Reactions</div>", unsafe_allow_html=True)
    top_rxns = active_solution.get("top_reactions", [])
    if top_rxns:
        df_top = pd.DataFrame(top_rxns)
        # Re-mapping to requested columns (Reaction ID, Equation, Flux Value, Subsystem)
        df_to_show = df_top.rename(columns={
            "reaction_id": "Reaction ID",
            "equation":    "Equation",
            "flux":        "Flux Value",
            "subsystem":   "Subsystem"
        })[["Reaction ID", "Equation", "Flux Value", "Subsystem"]]
        
        st.dataframe(df_to_show, use_container_width=True, hide_index=True)
        _render_flux_chart(df_top, f"{source_name} — Top Fluxes")

    # ─── 6️⃣ Active vs Blocked Summary (Rule 5) ───────────────────────────────
    st.markdown("<div class='section-heading'>📊 Solution Sparsity</div>", unsafe_allow_html=True)
    if active_solution.get("success"):
        n_total = summary.get("num_reactions", 0)
        # Sparsity usually refers to the whole model, but we only have top reactions in the response
        # To be accurate, we just show the count for what were return or if pFBA provided more.
        # However, for a proper "Active" count we'd need the full flux vector.
        # Since we don't return the full vector for performance, we'll label this "Returned Reactions"
        t1, t2, t3 = st.columns(3)
        t1.metric("Total Model Reactions", f"{n_total:,}")
        t2.metric("Active Exchange Reactions", len(ex_fluxes))
        t3.metric("Top Reactions Analyzed", len(top_rxns))

    # ─── 7️⃣ LP Formulation (Rule 6 - Collapsible) ───────────────────────────
    with st.expander("📐 Linear Programming Formulation", expanded=False):
        st.markdown(fr"""
        ### 📐 Linear Programming Formulation
        **Objective Function:**
        $Maximize: c^T v$

        **Constraints:**
        *   **Steady-State:** $S \cdot v = 0$
        *   **Capacity Bounds:** $LB \leq v \leq UB$

        *Where **S** is the stoichiometric matrix and **v** is the flux vector.*
        """)

    # ─── 8️⃣ Numerical Tolerances (Rule 7) ──────────────────────────────────
    st.markdown("<div class='section-heading'>🔢 Numerical Tolerances</div>", unsafe_allow_html=True)
    feas_tol = st.session_state.get("feasibility_tol", 1e-7)
    opt_tol  = st.session_state.get("optimality_tol", 1e-7)
    st.markdown(f"""
    <div style='font-family:JetBrains Mono; color:#8892a4; font-size:0.85rem; background:rgba(22,27,46,0.6); padding:12px; border-radius:8px;'>
    Feasibility Tolerance: {feas_tol:.0e}<br>
    Optimality Tolerance: &nbsp;{opt_tol:.0e}
    </div>
    """, unsafe_allow_html=True)


def _render_result_card(title: str, data: Dict) -> None:
    """Compact header card — intentionally minimal; detail is in the structured sections above."""
    ok     = data.get("success", False)
    status = data.get("solver_status", "unknown")
    solver = data.get("solver_name", "—")
    err_msg = (data.get("error") or {}).get("message", "") if not ok else ""
    badge = "<span class='badge badge-ok'>OPTIMAL</span>" if ok else "<span class='badge badge-err'>NON-OPTIMAL</span>"
    err_html = (
        f"<div style='margin-top:8px;padding:8px 12px;background:rgba(231,76,60,0.1);"
        f"border-radius:8px;color:#e74c3c;font-size:0.8rem;'>⚠ {err_msg}</div>"
        if err_msg else ""
    )
    st.markdown(f"""
    <div style='background:rgba(22,27,46,0.9);border:1px solid rgba(79,142,247,0.2);
                border-radius:12px;padding:14px 18px;margin-bottom:4px;'>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-weight:600;font-size:0.95rem;color:#c8d5f0;'>{title}</span>
            {badge}
        </div>
        <div style='font-size:0.75rem;color:#4f5a72;margin-top:4px;'>Solver: {solver}</div>
        {err_html}
    </div>""", unsafe_allow_html=True)


def _render_flux_chart(df: pd.DataFrame, title: str) -> None:
    # Use standardized column names from our new schema
    x_col = "reaction_id"
    y_col = "flux"
    
    colors = ["#e74c3c" if v < 0 else "#4f8ef7" for v in df[y_col]]
    
    fig = go.Figure(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.4f}" for v in df[y_col]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Flux: %{y:.4f} mmol·gDW⁻¹·h⁻¹<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#c8d5f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,46,0.7)",
        font=dict(family="Inter", color="#8892a4"),
        xaxis=dict(tickangle=-35, showgrid=False, tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(title="Flux (mmol·gDW⁻¹·h⁻¹)", gridcolor="rgba(79,142,247,0.1)"),
        margin=dict(t=50, b=80, l=60, r=60),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# Tab: Validation
# ==================================================================

def render_validation_tab() -> None:
    data = st.session_state.validation_result

    if data is None:
        st.info("Click **🔍 Run Validation** in the sidebar.", icon="🔍")
        return

    st.markdown("<div class='section-heading'>📋 Validation Summary</div>", unsafe_allow_html=True)

    # ---- Feasibility banner ----
    if data.get("is_feasible"):
        st.markdown(
            f"<div style='background:rgba(39,174,96,0.1);border:1px solid rgba(39,174,96,0.3);"
            f"border-radius:12px;padding:16px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px;'>"
            f"<span style='font-size:1.8rem;'>✅</span>"
            f"<div><div style='font-weight:600;color:#2ecc71;'>Objective Feasible</div>"
            f"<div style='color:#8892a4;font-size:0.85rem;'>{data.get('feasibility_message','')}</div></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.3);"
            f"border-radius:12px;padding:16px 20px;margin-bottom:16px;display:flex;align-items:center;gap:12px;'>"
            f"<span style='font-size:1.8rem;'>🚨</span>"
            f"<div><div style='font-weight:600;color:#e74c3c;'>Objective Infeasible</div>"
            f"<div style='color:#8892a4;font-size:0.85rem;'>{data.get('feasibility_message','')}</div></div></div>",
            unsafe_allow_html=True,
        )

    for err in data.get("errors", []):
        st.error(f"🚨 {err}")
    for warn in data.get("warnings", []):
        st.warning(f"⚠️ {warn}")
    if not data.get("errors") and not data.get("warnings"):
        st.success("✅ All checks passed — model appears structurally sound.", icon="✅")

    # ---- Check sections ----
    _render_val_section(
        "🔒 Blocked Reactions",
        data.get("blocked_count", 0),
        data.get("blocked_reactions", []),
        ["reaction_id", "min_flux", "max_flux"],
        "FVA-detected zero-flux reactions under all conditions.",
        "No blocked reactions detected.",
    )
    _render_val_section(
        "⚖️ Inconsistent Bounds",
        data.get("inconsistent_bounds_count", 0),
        data.get("inconsistent_bounds", []),
        ["reaction_id", "name", "lower_bound", "upper_bound"],
        "Reactions where lb > ub (LP-infeasible by construction).",
        "All reaction bounds are consistent.",
    )
    _render_val_section(
        "🧬 Gene-Orphan Reactions",
        data.get("gene_orphan_count", 0),
        data.get("gene_orphan_reactions", []),
        ["reaction_id", "name", "subsystem"],
        "Internal reactions with no gene-protein-reaction (GPR) rule.",
        "All non-boundary reactions have gene associations.",
    )

    # Mass balance (special — shows both balanced/unbalanced counts)
    unbal = data.get("unbalanced_count", 0)
    bal   = data.get("balanced_count", 0)
    with st.expander(
        f"⚗️ Mass Balance — {'⚠️ ' + str(unbal) + ' unbalanced' if unbal else '✅ OK'}",
        expanded=unbal > 0,
    ):
        st.markdown(
            f"<div style='color:#8892a4;font-size:0.87rem;margin-bottom:10px;'>"
            f"Stoichiometric conservation check. {bal} balanced · {unbal} unbalanced.</div>",
            unsafe_allow_html=True,
        )
        imbalances = data.get("mass_imbalances", [])
        if unbal == 0:
            st.success(f"All {bal} annotated reactions are mass-balanced.", icon="✅")
        elif imbalances:
            st.dataframe(pd.DataFrame(imbalances), use_container_width=True, hide_index=True)


def _render_val_section(
    title: str,
    count: int,
    rows: List[Dict],
    columns: List[str],
    description: str,
    ok_msg: str,
) -> None:
    label = f"⚠️ {count} issue(s)" if count else "✅ OK"
    with st.expander(f"{title} — {label}", expanded=count > 0):
        st.markdown(
            f"<div style='color:#8892a4;font-size:0.87rem;margin-bottom:10px;'>{description}</div>",
            unsafe_allow_html=True,
        )
        if count == 0:
            st.success(ok_msg, icon="✅")
        else:
            st.warning(f"{count} issue(s) found.", icon="⚠️")
            if rows:
                df = pd.DataFrame(rows)
                display_cols = [c for c in columns if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ==================================================================
# FVA Tab
# ==================================================================

def _action_run_fva(
    fraction: float,
    reaction_ids: Optional[List[str]],
    confirm: bool,
) -> None:
    """
    Call POST /models/{id}/fva and store result in session_state.
    Handles the MODEL_TOO_LARGE confirmation flow transparently.
    """
    payload: Dict[str, Any] = {
        "solver": st.session_state.solver,
        "fraction_of_optimum": fraction,
        "confirm_full_model": confirm,
    }
    if reaction_ids:
        payload["reaction_ids"] = reaction_ids

    status, body = _post_json_raw(
        f"/models/{st.session_state.model_id}/fva", payload
    )

    if status is None:
        return  # network error already shown by _post_json_raw

    if status == 200 and body and body.get("success"):
        st.session_state.fva_result = body
        st.session_state.fva_error = None
        return

    # Store error state for the tab to render the confirmation flow
    st.session_state.fva_result = None
    st.session_state.fva_error = {"status": status, "body": body}


def render_fva_tab() -> None:
    """Render the Flux Variability Analysis tab."""
    summary = st.session_state.model_summary or {}
    n_reactions = summary.get("num_reactions", 0)

    # ---- Header ----
    st.markdown(
        """
        <div class='section-heading'>📊 Flux Variability Analysis</div>
        <p style='color:#8892a4;font-size:0.9rem;margin-bottom:1.2rem;'>
        FVA computes the <b>minimum and maximum flux</b> for each reaction
        compatible with a given fraction of the optimal objective value.
        It requires 2 × |R| LP solves — slow on large models.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---- Large model warning ----
    if n_reactions > 2_000:
        st.warning(
            f"⚠️ Large model detected ({n_reactions:,} reactions). "
            "Full FVA may take **several minutes**. "
            "Consider using a reaction subset or reducing `fraction_of_optimum`.",
            icon="⏱",
        )

    # ---- Controls ----
    col1, col2 = st.columns([1, 1])
    with col1:
        fraction = st.slider(
            "Fraction of Optimum",
            min_value=0.0, max_value=1.0, value=1.0, step=0.05,
            help="1.0 = only flux distributions at the FBA optimum. 0.0 = all thermodynamically feasible fluxes.",
        )

    with col2:
        st.markdown("**Optional Reaction Subset**")
        all_rxn_ids = [
            r["id"] for r in
            (_post_json(f"/reactions/{st.session_state.model_id}",
                        {}) or {}).get("reactions", [])
        ] if False else []  # avoid GET-on-render; use text_area instead

        subset_text = st.text_area(
            "Reaction IDs (one per line, blank = full model)",
            height=100,
            placeholder="e.g.\nPFK\nPYK\nCS",
            help="Leave blank to run on the entire model.",
        )
        reaction_ids: Optional[List[str]] = (
            [r.strip() for r in subset_text.splitlines() if r.strip()]
            if subset_text.strip() else None
        )

    # ---- Confirmation checkbox (shown only if server returned 400 / MODEL_TOO_LARGE) ----
    err = st.session_state.fva_error
    confirm_checked = False
    if err and err.get("status") == 400:
        body = err.get("body", {})
        if body.get("error") == "MODEL_TOO_LARGE":
            st.warning(
                f"⚠️ {body.get('message', 'Large model — confirmation required.')}",
                icon="⚠️",
            )
            confirm_checked = st.checkbox(
                "I understand this may take several minutes. Proceed with full FVA.",
                key="fva_confirm_checkbox",
            )
        else:
            st.error(f"❌ {body.get('message', 'Unknown error.')}", icon="🚫")
    elif err and err.get("status") == 408:
        st.error(
            "⏱ FVA timed out. Try a smaller model, a reaction subset, "
            "or a lower fraction_of_optimum.",
            icon="⏱",
        )
    elif err and err.get("status") == 422:
        body = err.get("body", {})
        st.error(
            f"🧬 Model infeasible under current constraints: "
            f"{body.get('message', '')}",
            icon="🧬",
        )
    elif err:
        body = err.get("body", {})
        st.error(f"❌ {body.get('message', 'FVA failed.')}", icon="🚫")

    # ---- Run button ----
    # Also auto-trigger if sidebar "Run FVA" was clicked
    auto_trigger = st.session_state.pop("_fva_trigger", False)
    run_clicked = st.button("▶ Run FVA", type="primary", use_container_width=False)

    if run_clicked or auto_trigger:
        confirm = confirm_checked or (reaction_ids is not None)  # subset bypasses confirmation
        with st.spinner("Running FVA… this may take a while for large models."):
            _action_run_fva(fraction, reaction_ids, confirm)
        st.rerun()

    # ---- Results display ----
    data = st.session_state.fva_result
    if data is None:
        st.info(
            "Configure options above and click **▶ Run FVA** "
            "(or use the sidebar button).",
            icon="📊",
        )
        return

    # ---- Summary metrics ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objective Value", f"{data.get('objective_value', 0):.4f}")
    m2.metric("Total Reactions", f"{data.get('total_reactions', 0):,}")
    m3.metric("Analysed", f"{data.get('analyzed_reactions', 0):,}")
    m4.metric("Blocked", f"{data.get('blocked_count', 0):,}")

    st.caption(
        f"Solver: **{data.get('solver_name', '—')}** · "
        f"Fraction of optimum: **{data.get('fraction_of_optimum', 1.0):.2f}** · "
        f"Status: **{data.get('solver_status', '—')}**"
    )
    st.markdown("---")

    results = data.get("results", [])
    if not results:
        st.info("No reactions in result.")
        return

    df = pd.DataFrame(results)

    # ---- Filter controls ----
    filter_col, _ = st.columns([1, 2])
    with filter_col:
        show_blocked_only = st.checkbox("Show blocked reactions only", value=False)

    df_display = df[df["is_blocked"]] if show_blocked_only else df

    # Rename for display
    df_display = df_display.rename(columns={
        "reaction_id": "Reaction ID",
        "minimum": "Min Flux",
        "maximum": "Max Flux",
        "range": "Range",
        "is_blocked": "Blocked",
    })

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Min Flux":  st.column_config.NumberColumn(format="%.6f"),
            "Max Flux":  st.column_config.NumberColumn(format="%.6f"),
            "Range":     st.column_config.NumberColumn(format="%.6f"),
            "Blocked":   st.column_config.CheckboxColumn(),
        },
    )

    # ---- Histogram: flux range distribution ----
    st.markdown("<div class='section-heading'>📈 Flux Range Distribution</div>", unsafe_allow_html=True)
    st.caption(
        "Histogram of (max − min) per reaction. "
        "Reactions with range = 0 are blocked. "
        "Large ranges indicate highly flexible reactions."
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["range"],
            nbinsx=60,
            marker_color="#4f8ef7",
            opacity=0.85,
            name="Flux range",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(11,14,26,0)",
        paper_bgcolor="rgba(11,14,26,0)",
        font=dict(family="Inter, sans-serif", color="#c8d5f0"),
        xaxis=dict(
            title="Flux Range (max − min) [mmol gDW⁻¹ h⁻¹]",
            gridcolor="rgba(79,142,247,0.1)",
            zerolinecolor="rgba(79,142,247,0.25)",
        ),
        yaxis=dict(
            title="Reaction Count",
            gridcolor="rgba(79,142,247,0.1)",
        ),
        bargap=0.05,
        margin=dict(l=0, r=0, t=20, b=0),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# Unified: Environment-to-Outcome Simulation
# ==================================================================


def _fetch_reaction_ids(limit: int = 200) -> List[str]:
    if st.session_state.prod_reactions is not None:
        return st.session_state.prod_reactions
    data = _get(
        f"/reactions/{st.session_state.model_id}",
        params={"page": 1, "page_size": limit},
    )
    ids: List[str] = []
    if data and data.get("success"):
        ids = [r["id"] for r in data.get("reactions", [])]
    st.session_state.prod_reactions = ids
    return ids


def _action_set_objective(reaction_id: str, direction: str) -> None:
    payload = {"reaction_id": reaction_id, "direction": direction}
    status, body = _post_json_raw(f"/models/{st.session_state.model_id}/objective", payload)
    if status == 200 and body and body.get("success"):
        st.session_state.objective_info = body
    else:
        st.session_state.objective_info = None
        err = (body or {}).get("message", "Unknown error")
        if status == 404:
            st.error(f"❌ Reaction not found: {err}")
        elif status == 422:
            st.error(f"🧬 Infeasible: {err}")
        elif status == 408:
            st.error("⏱ Objective validation timed out.")
        else:
            st.error(f"❌ {err}")


def _action_run_envelope(biomass_rxn: str, product_rxn: str, steps: int, solver: str) -> None:
    payload = {
        "biomass_reaction": biomass_rxn,
        "product_reaction": product_rxn,
        "steps": steps,
        "solver": solver,
    }
    status, body = _post_json_raw(
        f"/models/{st.session_state.model_id}/production-envelope", payload
    )
    if status == 200 and body and body.get("success"):
        st.session_state.envelope_result = body
        st.session_state.envelope_error = None
    else:
        st.session_state.envelope_result = None
        st.session_state.envelope_error = {"status": status, "body": body or {}}


def _action_apply_preset(preset: str) -> None:
    status, body = _post_json_raw(
        f"/models/{st.session_state.model_id}/medium/preset",
        {"preset": preset},
    )
    if status == 200 and body and body.get("success"):
        st.session_state.medium_data = body
        st.session_state.medium_success = f"Preset '{preset}' applied."
        _invalidate_analysis_results()
    else:
        st.error(f"❌ {(body or {}).get('message', 'Preset failed.')}")


def _action_reset_medium() -> None:
    status, body = _post_json_raw(
        f"/models/{st.session_state.model_id}/medium/reset", {}
    )
    if status == 200 and body and body.get("success"):
        st.session_state.medium_data = body
        st.session_state.medium_success = "Medium reset to original SBML bounds."
        _invalidate_analysis_results()
    else:
        st.error(f"❌ {(body or {}).get('message', 'Reset failed.')}")


def _action_update_medium(updates: Dict[str, Any]) -> None:
    status, body = _post_json_raw(
        f"/models/{st.session_state.model_id}/medium",
        {"updates": updates},
    )
    if status == 200 and body and body.get("success"):
        st.session_state.medium_data = body
        st.session_state.medium_success = f"{len(updates)} exchange bound(s) updated."
        _invalidate_analysis_results()
    else:
        st.error(f"❌ {(body or {}).get('message', 'Update failed.')}")


def _invalidate_analysis_results() -> None:
    for key in ("fba_result", "pfba_result", "validation_result",
                "fva_result", "envelope_result", "objective_info"):
        if key in st.session_state:
            st.session_state[key] = None


def _fetch_medium() -> Optional[Dict]:
    if st.session_state.medium_data is not None:
        return st.session_state.medium_data
    data = _get(f"/models/{st.session_state.model_id}/medium")
    if data and data.get("success"):
        st.session_state.medium_data = data
    return st.session_state.medium_data


def _run_quick_fba_for_simulation() -> Optional[Dict]:
    """Run FBA silently and return the result dict."""
    return _post_json("/run-fba", {
        "model_id": st.session_state.model_id,
        "solver": st.session_state.solver,
        "feasibility_tol": st.session_state.feasibility_tol,
        "optimality_tol": st.session_state.optimality_tol,
    })


def render_env_simulation_tab() -> None:
    """Unified Environment-to-Outcome Simulation tab."""
    auto_trigger = st.session_state.pop("_medium_trigger", False) or st.session_state.pop("_prod_trigger", False)
    if auto_trigger:
        st.session_state.medium_data = None

    summary = st.session_state.model_summary or {}
    n_reactions = summary.get("num_reactions", 0)
    default_objective = summary.get("objective_reaction", "")

    # ── Hero Header ────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(79,142,247,0.12) 0%,
                rgba(0,210,193,0.06) 100%); border:1px solid rgba(79,142,247,0.2);
                border-radius:16px; padding:24px 28px; margin-bottom:24px;'>
        <div style='font-size:1.55rem; font-weight:700;
                    background:linear-gradient(90deg,#4f8ef7,#00d2c1);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-bottom:6px;'>
            🌐 Environment-to-Outcome Simulation
        </div>
        <div style='color:#8892a4; font-size:0.9rem; line-height:1.6;'>
            A unified cause–effect pipeline: <b style='color:#4f8ef7;'>define what the cell consumes</b>
            → <b style='color:#00d2c1;'>choose what to optimize</b>
            → <b style='color:#a78bfa;'>observe growth &amp; production outcomes</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Phase Indicators ──────────────────────────────────
    medium = _fetch_medium()
    obj_info = st.session_state.objective_info
    env_data = st.session_state.envelope_result

    ph1_done = medium is not None
    ph2_done = obj_info is not None and obj_info.get("success")
    ph3_done = env_data is not None

    def _phase_dot(done: bool) -> str:
        return ("<span style='color:#00d2c1;font-size:1.1rem;'>●</span>"
                if done else
                "<span style='color:#3a4a6b;font-size:1.1rem;'>○</span>")

    st.markdown(f"""
    <div style='display:flex; gap:0; margin-bottom:28px; border-radius:12px;
                overflow:hidden; border:1px solid rgba(79,142,247,0.15);'>
        <div style='flex:1; padding:14px 18px;
                    background:{'rgba(0,210,193,0.08)' if ph1_done else 'rgba(22,27,46,0.6)'};
                    border-right:1px solid rgba(79,142,247,0.12);'>
            <div style='font-size:0.72rem; color:#8892a4; letter-spacing:.06em;
                        text-transform:uppercase; margin-bottom:4px;'>
                {_phase_dot(ph1_done)} Phase 1
            </div>
            <div style='font-weight:600; color:#c8d5f0; font-size:0.9rem;'>🧪 Define Environment</div>
            <div style='color:#4f5a72; font-size:0.78rem; margin-top:2px;'>What goes in</div>
        </div>
        <div style='flex:1; padding:14px 18px;
                    background:{'rgba(0,210,193,0.08)' if ph2_done else 'rgba(22,27,46,0.6)'};
                    border-right:1px solid rgba(79,142,247,0.12);'>
            <div style='font-size:0.72rem; color:#8892a4; letter-spacing:.06em;
                        text-transform:uppercase; margin-bottom:4px;'>
                {_phase_dot(ph2_done)} Phase 2
            </div>
            <div style='font-weight:600; color:#c8d5f0; font-size:0.9rem;'>🎯 Set Objective</div>
            <div style='color:#4f5a72; font-size:0.78rem; margin-top:2px;'>What is being optimized</div>
        </div>
        <div style='flex:1; padding:14px 18px;
                    background:{'rgba(0,210,193,0.08)' if ph3_done else 'rgba(22,27,46,0.6)'};'>
            <div style='font-size:0.72rem; color:#8892a4; letter-spacing:.06em;
                        text-transform:uppercase; margin-bottom:4px;'>
                {_phase_dot(ph3_done)} Phase 3
            </div>
            <div style='font-weight:600; color:#c8d5f0; font-size:0.9rem;'>📈 Run &amp; Visualize</div>
            <div style='color:#4f5a72; font-size:0.78rem; margin-top:2px;'>What comes out</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================
    # PHASE 1 — Define Environment (Medium)
    # ==============================================================
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
        <div style='width:28px; height:28px; border-radius:50%;
                    background:linear-gradient(135deg,#4f8ef7,#00d2c1);
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.8rem; font-weight:700; color:white; flex-shrink:0;'>1</div>
        <div style='font-size:1.05rem; font-weight:700; color:#c8d5f0;'>Define the Environment</div>
        <div style='font-size:0.8rem; color:#4f5a72; margin-left:4px;'>
            — Configure what the cell is allowed to consume (exchange reaction bounds)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Success banner
    success_msg = st.session_state.get("medium_success")
    if success_msg:
        del st.session_state["medium_success"]
        st.success(f"✅ {success_msg} — run simulation below to see updated outcomes.", icon="✅")

    # Load medium
    with st.spinner("Loading exchange reactions…"):
        medium = _fetch_medium()

    if medium is None:
        st.error("❌ Could not load medium from backend. Is it running?", icon="🚫")
        return

    # ── Status bar
        has_c = medium.get("has_carbon_uptake")
        has_o2 = medium.get("has_oxygen_uptake")
        uptake_on = medium.get("uptake_enabled_count", 0)
        total_ex = medium.get("total_exchanges", 0)
        preset_lbl = medium.get("active_preset", "")

        cond_aerobic = "🌬️ Aerobic" if has_o2 else "💤 Anaerobic"
        cond_carbon = "🍬 Carbon available" if has_c else "🚫 No carbon"
        cond_color_o2 = "#4f8ef7" if has_o2 else "#f5a623"
        cond_color_c = "#27ae60" if has_c else "#e74c3c"

        st.markdown(f"""
        <div style='display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;'>
            <div style='background:rgba(22,27,46,0.85); border:1px solid rgba(79,142,247,0.15);
                        border-radius:10px; padding:10px 16px; flex:1; min-width:120px;'>
                <div style='font-size:0.72rem; color:#4f5a72; text-transform:uppercase;
                            letter-spacing:.05em;'>O₂ Condition</div>
                <div style='font-weight:700; color:{cond_color_o2}; margin-top:3px;
                            font-size:0.95rem;'>{cond_aerobic}</div>
            </div>
            <div style='background:rgba(22,27,46,0.85); border:1px solid rgba(79,142,247,0.15);
                        border-radius:10px; padding:10px 16px; flex:1; min-width:120px;'>
                <div style='font-size:0.72rem; color:#4f5a72; text-transform:uppercase;
                            letter-spacing:.05em;'>Carbon</div>
                <div style='font-weight:700; color:{cond_color_c}; margin-top:3px;
                            font-size:0.95rem;'>{cond_carbon}</div>
            </div>
            <div style='background:rgba(22,27,46,0.85); border:1px solid rgba(79,142,247,0.15);
                        border-radius:10px; padding:10px 16px; flex:1; min-width:120px;'>
                <div style='font-size:0.72rem; color:#4f5a72; text-transform:uppercase;
                            letter-spacing:.05em;'>Uptake Active</div>
                <div style='font-weight:700; color:#c8d5f0; margin-top:3px;
                            font-size:0.95rem;'>{uptake_on} / {total_ex}</div>
            </div>
            <div style='background:rgba(22,27,46,0.85); border:1px solid rgba(79,142,247,0.15);
                        border-radius:10px; padding:10px 16px; flex:1; min-width:120px;'>
                <div style='font-size:0.72rem; color:#4f5a72; text-transform:uppercase;
                            letter-spacing:.05em;'>Preset</div>
                <div style='font-weight:700; color:#a78bfa; margin-top:3px;
                            font-size:0.95rem;'>{str(medium.get('active_preset') or "Custom").replace('_', ' ')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not has_c:
            st.error("🚫 No carbon source — growth will be zero. Enable a carbon exchange below.", icon="❌")
        if not has_o2:
            st.warning("💤 Anaerobic — O₂ exchange closed. Ensure fermentation pathways are active.", icon="⚠️")

        # ── Quick Presets
        st.markdown("<div style='font-size:0.85rem; font-weight:600; color:#8892a4; "
                    "text-transform:uppercase; letter-spacing:.05em; margin-bottom:8px;'>"
                    "⚡ Quick Presets — impact is immediately reflected in simulation</div>",
                    unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            if st.button("⭐ Aerobic Glucose", use_container_width=True, key="preset_aerobic"):
                with st.spinner("Applying…"):
                    _action_apply_preset("aerobic_glucose")
                st.rerun()
        with p2:
            if st.button("🌿 Anaerobic Glucose", use_container_width=True, key="preset_anaerobic"):
                with st.spinner("Applying…"):
                    _action_apply_preset("anaerobic_glucose")
                st.rerun()
        with p3:
            if st.button("🔒 Close All", use_container_width=True, key="preset_closed"):
                with st.spinner("Applying…"):
                    _action_apply_preset("minimal_closed")
                st.rerun()
        with p4:
            if st.button("🔄 Reset to SBML", use_container_width=True, key="preset_reset"):
                with st.spinner("Restoring…"):
                    _action_reset_medium()
                st.rerun()
        # ── Quick Search & Add Nutrient
        st.markdown("<div style='margin:16px 0 8px 0; font-size:0.85rem; font-weight:600; color:#8892a4;'>"
                    "🔍 Search & Add Nutrient</div>", unsafe_allow_html=True)
        
        exchanges = medium.get("exchanges", [])
        if exchanges:
            # Prepare search options: "Reaction Name (ID)"
            search_opts = {f"{e.get('name', 'Unnamed')} ({e['reaction_id']})": e['reaction_id'] for e in exchanges}
            
            sc1, sc2, sc3 = st.columns([3, 1, 1])
            with sc1:
                selected_lbl = st.selectbox("Select Nutrient / Exchange", options=list(search_opts.keys()), 
                                            label_visibility="collapsed", key="quick_search_rxn")
                selected_id = search_opts[selected_lbl]
                # Find current value for this rxn
                curr_e = next((e for e in exchanges if e['reaction_id'] == selected_id), None)
                curr_lb = curr_e['effective_lower_bound'] if curr_e else 0.0
            
            with sc2:
                new_lb = st.number_input("Lower Bound", value=float(curr_lb), step=1.0, 
                                          label_visibility="collapsed", key="quick_search_lb",
                                          help="Negative for uptake (e.g. -10.0), 0.0 for closed.")
            
            with sc3:
                if st.button("➕ Add / Update", use_container_width=True, type="secondary"):
                    with st.spinner("Updating environment…"):
                        _action_update_medium({selected_id: {"lower_bound": float(new_lb)}})
                    st.rerun()
            
            if curr_e and curr_e.get('is_modified'):
                st.caption(f"ℹ️ Current setting for **{selected_id}** is **{curr_lb:.2f}** (modified).")
            elif curr_e:
                st.caption(f"ℹ️ Current setting for **{selected_id}** is **{curr_lb:.2f}** (SBML default).")

        # ── Manual Editor (collapsible)
        with st.expander("✏️ Manual Exchange Bound Editor", expanded=False):
            st.caption(
                "Edit lower bounds (negative = uptake enabled, 0 = closed). "
                "Submit with **Apply Changes** — changes invalidate cached simulation results."
            )
            exchanges = medium.get("exchanges", [])
            if exchanges:
                df_all = pd.DataFrame([
                    {
                        "reaction_id": e["reaction_id"],
                        "name": e.get("name", "")[:40],
                        "lb": e["effective_lower_bound"],
                        "ub": e["effective_upper_bound"],
                        "uptake": e["uptake_enabled"],
                        "modified": e["is_modified"],
                    }
                    for e in exchanges
                ])
                PAGE_SIZE = 50
                total_pages = max(1, -(-len(df_all) // PAGE_SIZE))
                col_page, _ = st.columns([1, 4])
                with col_page:
                    page = st.number_input(
                        "Page", min_value=1, max_value=total_pages, value=1, step=1, key="medium_page"
                    )
                df_page = df_all.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE].copy()
                st.caption(f"Showing {len(df_page)} of {len(df_all)} exchanges.")

                edited = st.data_editor(
                    df_page, use_container_width=True, hide_index=True, num_rows="fixed",
                    key=f"medium_editor_{page}",
                    column_config={
                        "reaction_id": st.column_config.TextColumn("Reaction ID", disabled=True, width="medium"),
                        "name": st.column_config.TextColumn("Name", disabled=True, width="large"),
                        "lb": st.column_config.NumberColumn("Lower Bound", min_value=-1000.0,
                                                             max_value=1000.0, step=1.0, format="%.2f",
                                                             help="Negative = uptake; 0 = closed"),
                        "ub": st.column_config.NumberColumn("Upper Bound", disabled=True, format="%.2f"),
                        "uptake": st.column_config.CheckboxColumn("Uptake ✔", disabled=True),
                        "modified": st.column_config.CheckboxColumn("Custom ✏", disabled=True),
                    },
                )
                changed: Dict[str, Any] = {}
                for orig_row, edited_row in zip(df_page.itertuples(), edited.itertuples()):
                    if abs(orig_row.lb - edited_row.lb) > 1e-9:
                        changed[orig_row.reaction_id] = {"lower_bound": float(edited_row.lb)}

                st.caption(f"{'🟡 ' + str(len(changed)) + ' bound(s) edited.' if changed else '🟢 No changes pending.'}")
                if st.button("✔ Apply Changes", type="primary", key="medium_apply_btn",
                             disabled=(len(changed) == 0)):
                    with st.spinner(f"Applying {len(changed)} update(s)…"):
                        _action_update_medium(changed)
                    st.rerun()

    st.markdown("<div style='height:6px; background:linear-gradient(90deg,rgba(79,142,247,0.3),"
                "rgba(0,210,193,0.1)); border-radius:3px; margin:24px 0;'></div>",
                unsafe_allow_html=True)

    # ==============================================================
    # PHASE 2 — Set Objective
    # ==============================================================
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
        <div style='width:28px; height:28px; border-radius:50%;
                    background:linear-gradient(135deg,#a78bfa,#4f8ef7);
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.8rem; font-weight:700; color:white; flex-shrink:0;'>2</div>
        <div style='font-size:1.05rem; font-weight:700; color:#c8d5f0;'>Set the Objective</div>
        <div style='font-size:0.8rem; color:#4f5a72; margin-left:4px;'>
            — What is being optimized by the LP solver
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.82rem; color:#4f5a72; margin-bottom:12px; 
                padding:10px 14px; background:rgba(22,27,46,0.6); 
                border-radius:8px; border-left:3px solid rgba(79,142,247,0.4);'>
        🔬 <b style='color:#8892a4;'>Scientific note:</b>
        Switching the objective re-directs the LP from maximizing <i>growth</i>
        to maximizing a <i>product flux</i>. The medium constraints from Phase 1 remain active.
    </div>
    """, unsafe_allow_html=True)

    rxn_ids = _fetch_reaction_ids(limit=200)
    if not rxn_ids:
        st.warning("⚠️ Could not load reaction list.", icon="⚠️")
        rxn_ids = []
    if n_reactions > 200:
        st.caption(f"ℹ️ Showing first 200 of {n_reactions:,} reactions.")

    obj_c1, obj_c2, obj_c3 = st.columns([3, 1, 1])
    with obj_c1:
        obj_rxn = st.selectbox(
            "Objective reaction (maximize / minimize)",
            options=rxn_ids if rxn_ids else [""],
            index=(
                rxn_ids.index(default_objective)
                if default_objective in rxn_ids else 0
            ),
            key="prod_obj_rxn",
            help="Choose the reaction the LP optimizer will maximize or minimize.",
        )
    with obj_c2:
        direction = st.radio(
            "Direction", options=["max", "min"], index=0,
            key="prod_direction", label_visibility="collapsed"
        )
    with obj_c3:
        if st.button("🎯 Set Objective", type="primary", key="prod_set_btn"):
            with st.spinner(f"Validating '{obj_rxn}' ({direction})…"):
                _action_set_objective(obj_rxn, direction)

    obj_info = st.session_state.objective_info
    if obj_info and obj_info.get("success"):
        oi1, oi2, oi3 = st.columns(3)
        oi1.metric("Objective Reaction", obj_info.get("objective_reaction", "—"))
        oi2.metric("Direction", obj_info.get("direction", "—").upper())
        oi3.metric("Baseline Flux", f"{obj_info.get('baseline_objective_value', 0.0):.4f}")
        st.caption(
            f"Solver: **{obj_info.get('solver_name', '—')}** · "
            f"Status: **{obj_info.get('solver_status', '—')}** — "
            "objective validated under current medium constraints."
        )

    st.markdown("<div style='height:6px; background:linear-gradient(90deg,rgba(167,139,250,0.3),"
                "rgba(79,142,247,0.1)); border-radius:3px; margin:24px 0;'></div>",
                unsafe_allow_html=True)

    # ==============================================================
    # PHASE 3 — Run & Visualize Outcomes
    # ==============================================================
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
        <div style='width:28px; height:28px; border-radius:50%;
                    background:linear-gradient(135deg,#00d2c1,#27ae60);
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.8rem; font-weight:700; color:white; flex-shrink:0;'>3</div>
        <div style='font-size:1.05rem; font-weight:700; color:#c8d5f0;'>Run &amp; Visualize Outcomes</div>
        <div style='font-size:0.8rem; color:#4f5a72; margin-left:4px;'>
            — Observe growth and production under the configured environment
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick FBA outcome ──────────────────────────────────────────
    fba_result = st.session_state.fba_result

    run_c1, run_c2 = st.columns([2, 1])
    with run_c1:
        if st.button("▶ Run Quick FBA" + (" (re-run after medium change)" if medium is not None else ""),
                     type="primary", key="sim_fba_btn", use_container_width=True):
            with st.spinner("Running FBA under current medium…"):
                result = _run_quick_fba_for_simulation()
            if result:
                st.session_state.fba_result = result
                st.session_state.analysis_latest_source = "FBA"
            st.rerun()
    with run_c2:
        st.markdown("""
        <div style='font-size:0.78rem; color:#4f5a72; padding:8px 12px;
                    background:rgba(22,27,46,0.5); border-radius:8px; margin-top:4px;'>
            Uses current medium bounds &amp; objective.
            LP algorithm: FBA (steady-state, constraint-based).
        </div>
        """, unsafe_allow_html=True)

    if fba_result and fba_result.get("success"):
        growth = fba_result.get("growth_rate", fba_result.get("objective_value", 0.0))
        ex_fluxes = fba_result.get("exchange_fluxes", [])
        obj_val = fba_result.get("objective_value", 0.0)

        st.markdown("<div class='section-heading' style='margin-top:16px;'>📤 Simulation Outcomes</div>",
                    unsafe_allow_html=True)

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("🌱 Growth Rate (h⁻¹)", f"{growth:.6f}")
        oc2.metric("🎯 Objective Value", f"{obj_val:.6f}")
        oc3.metric("💱 Active Exchanges", len([e for e in ex_fluxes if abs(e.get("flux", 0.0)) > 1e-9]) if ex_fluxes else 0)

        if ex_fluxes:
            uptake_ex = [e for e in ex_fluxes if e.get("direction") == "Uptake"]
            secr_ex = [e for e in ex_fluxes if e.get("direction") == "Secretion"]

            col_exch_l, col_exch_r = st.columns(2)
            with col_exch_l:
                st.markdown("<div style='font-size:0.8rem; color:#e74c3c; font-weight:600;"
                            " margin-bottom:6px;'>⬇️ Uptake (what goes in)</div>",
                            unsafe_allow_html=True)
                if uptake_ex:
                    df_up = pd.DataFrame(uptake_ex).rename(columns={
                        "metabolite_name": "Metabolite",
                        "reaction_id": "Reaction",
                        "flux": "Flux"
                    })
                    df_up["Flux"] = df_up["Flux"].map(lambda x: f"{float(x):.4f}")
                    st.dataframe(df_up[["Metabolite", "Reaction", "Flux"]], use_container_width=True,
                                 hide_index=True)
                else:
                    st.caption("No active uptake.")
            with col_exch_r:
                st.markdown("<div style='font-size:0.8rem; color:#27ae60; font-weight:600;"
                            " margin-bottom:6px;'>⬆️ Secretion (what comes out)</div>",
                            unsafe_allow_html=True)
                if secr_ex:
                    df_sec = pd.DataFrame(secr_ex).rename(columns={
                        "metabolite_name": "Metabolite",
                        "reaction_id": "Reaction",
                        "flux": "Flux"
                    })
                    df_sec["Flux"] = df_sec["Flux"].map(lambda x: f"{float(x):.4f}")
                    st.dataframe(df_sec[["Metabolite", "Reaction", "Flux"]], use_container_width=True,
                                 hide_index=True)
                else:
                    st.caption("No active secretion.")

            # ─── New Synthesis Card ─────────────────────────────────────
            carbon_src = ", ".join([e.get("metabolite_name", e["reaction_id"]) for e in uptake_ex[:2]])
            prod_info = (f"Producing **{len(secr_ex)}** compounds" if secr_ex else "No metabolic byproducts")
            status_color = "#27ae60" if growth > 1e-4 else "#e74c3c"
            
            st.markdown(f"""
            <div style='background:rgba(79,142,247,0.05); border-left:4px solid {status_color};
                        padding:16px; border-radius:8px; margin-top:20px;'>
                <div style='font-size:0.85rem; font-weight:700; color:#c8d5f0; margin-bottom:8px;'>
                    🧬 Simulation Synthesis
                </div>
                <div style='font-size:0.88rem; color:#8892a4; line-height:1.5;'>
                    Under the current <b>{str(medium.get('active_preset') or 'Custom').replace('_', ' ')}</b> environment:
                    <ul>
                        <li>The system achieves a steady-state growth rate of <b style='color:{status_color};'>{growth:.4f} h⁻¹</b>.</li>
                        <li>It is primarily consuming <b>{carbon_src}{'...' if len(uptake_ex) > 2 else ''}</b>.</li>
                        <li>{prod_info}.</li>
                    </ul>
                    <i style='font-size:0.8rem; color:#4f5a72;'>* Predicted values based on standard stoichiometry & steady-state assumptions.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
    elif fba_result and not fba_result.get("success"):
        st.warning(fba_result.get("message", "FBA returned non-optimal."), icon="⚠️")
    else:
        st.info("👆 Click **▶ Run Quick FBA** to see what the cell produces under the current environment.",
                icon="▶️")

    st.markdown("<hr style='border-color:rgba(79,142,247,0.1); margin:28px 0;'>",
                unsafe_allow_html=True)

    # ── Growth–Production Envelope ─────────────────────────────────
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:10px;'>
        <div style='font-size:1.0rem; font-weight:700; color:#c8d5f0;'>📈 Growth–Production Envelope</div>
        <div style='font-size:0.78rem; color:#4f5a72; background:rgba(79,142,247,0.08);
                    padding:3px 10px; border-radius:12px; border:1px solid rgba(79,142,247,0.18);'>
            Depends entirely on the medium configured in Phase 1
        </div>
    </div>
    <div style='font-size:0.82rem; color:#4f5a72; margin-bottom:14px;'>
        Scans growth rate from 0 → max, maximizing product at each point.
        The resulting Pareto curve reveals the <b style='color:#8892a4;'>trade-off between
        growth and product formation</b> under the current medium constraints.
    </div>
    """, unsafe_allow_html=True)

    if n_reactions > 5_000:
        st.warning(f"⚠️ Large model ({n_reactions:,} reactions). Keep Steps ≤ 30.", icon="⏱")

    env_c1, env_c2, env_c3 = st.columns([2, 2, 1])
    with env_c1:
        biomass_default = (obj_info or {}).get("objective_reaction", default_objective)
        biomass_rxn = st.selectbox(
            "🌱 Biomass / Growth reaction",
            options=rxn_ids if rxn_ids else [""],
            index=(rxn_ids.index(biomass_default) if biomass_default in rxn_ids else 0),
            key="env_biomass",
            help="The reaction representing cell growth (scanned from 0 → max).",
        )
    with env_c2:
        product_rxn = st.selectbox(
            "🧪 Product reaction",
            options=rxn_ids if rxn_ids else [""],
            index=min(1, len(rxn_ids) - 1) if len(rxn_ids) > 1 else 0,
            key="env_product",
            help="Exchange or pathway reaction for the compound of interest.",
        )
    with env_c3:
        env_steps = st.slider(
            "Steps", min_value=5, max_value=50, value=20, step=5, key="env_steps",
            help="More steps = smoother Pareto curve, longer compute time.",
        )

    env_err = st.session_state.envelope_error
    if env_err:
        status = env_err.get("status")
        msg = env_err.get("body", {}).get("message", "Unknown error.")
        if status == 408:
            st.error("⏱ Envelope timed out. Reduce steps or use a smaller model.", icon="⏱")
        elif status == 422:
            st.error(f"🧬 Infeasible: {msg}", icon="🧬")
        elif status == 400:
            st.warning(f"⚠️ {msg}", icon="⚠️")
        else:
            st.error(f"❌ {msg}")

    if st.button("▶ Run Envelope", type="primary", key="env_run_btn"):
        with st.spinner(f"Computing envelope ({env_steps} points) under current medium…"):
            _action_run_envelope(biomass_rxn, product_rxn, env_steps, st.session_state.solver)
        st.rerun()

    env_data = st.session_state.envelope_result
    if env_data is None:
        st.info("Select reactions above and click **▶ Run Envelope** to compute the Pareto frontier.",
                icon="📈")
    else:
        mx_g = env_data.get("max_growth", 0.0)
        mx_p = env_data.get("max_product", 0.0)
        yield_ratio = (mx_p / mx_g) if mx_g > 1e-12 else 0.0

        em1, em2, em3, em4 = st.columns(4)
        em1.metric("Max Growth", f"{mx_g:.4f}", help="Max biomass flux [mmol gDW⁻¹ h⁻¹]")
        em2.metric("Max Product", f"{mx_p:.4f}", help="Max product flux in scan")
        em3.metric("Theoretical Yield", f"{yield_ratio:.4f}",
                   help="max_product / max_growth")
        em4.metric("Points", str(env_data.get("steps", 0)))

        st.caption(
            f"Environment: **{str(medium.get('active_preset') or 'Custom').replace('_', ' ')}** · "
            f"Biomass: **{env_data.get('biomass_reaction')}** · "
            f"Product: **{env_data.get('product_reaction')}** · "
            f"Solver: **{env_data.get('solver_name', '—')}**"
        )

        growth_vals = env_data.get("growth_values", [])
        product_vals = env_data.get("product_values", [])

        # Medium condition annotation
        medium_label = medium.get("active_preset", "Custom medium") if medium else "Unknown medium"
        aerobic_str = "Aerobic" if (medium and medium.get("has_oxygen_uptake")) else "Anaerobic"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=growth_vals + growth_vals[::-1],
            y=product_vals + [0.0] * len(product_vals),
            fill="toself", fillcolor="rgba(79,142,247,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=growth_vals, y=product_vals,
            mode="lines+markers",
            name=f"Pareto frontier ({aerobic_str})",
            line=dict(color="#4f8ef7", width=2.5),
            marker=dict(size=7, color="#00d2c1", line=dict(color="#4f8ef7", width=1.5)),
            hovertemplate="Growth: <b>%{x:.4f}</b><br>Product: <b>%{y:.4f}</b><extra></extra>",
        ))
        fig.update_layout(
            plot_bgcolor="rgba(11,14,26,0)",
            paper_bgcolor="rgba(11,14,26,0)",
            font=dict(family="Inter, sans-serif", color="#c8d5f0"),
            title=dict(
                text=f"Growth–Production Envelope  ·  {medium_label}  ·  {aerobic_str}",
                font=dict(size=13, color="#8892a4"),
            ),
            annotations=[
                dict(
                    text="← curve shape depends on medium constraints |"
                         " change Phase 1 medium to see a different curve →",
                    xref="paper", yref="paper", x=0.5, y=-0.13,
                    showarrow=False, font=dict(size=10, color="#4f5a72"),
                )
            ],
            xaxis=dict(
                title=f"Biomass flux — {env_data.get('biomass_reaction', '')} [mmol gDW⁻¹ h⁻¹]",
                gridcolor="rgba(79,142,247,0.1)", zerolinecolor="rgba(79,142,247,0.25)",
            ),
            yaxis=dict(
                title=f"Product flux — {env_data.get('product_reaction', '')} [mmol gDW⁻¹ h⁻¹]",
                gridcolor="rgba(79,142,247,0.1)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=60),
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ─── New Synthesis Card (Envelope) ──────────────────────────
        has_coupling = mx_p > 1e-5 and growth_vals[0] < 1e-8
        st.markdown(f"""
        <div style='background:rgba(79,142,247,0.05); border:1px solid rgba(79,142,247,0.1);
                    padding:16px; border-radius:12px; margin-top:12px;'>
            <div style='font-size:0.85rem; font-weight:700; color:#c8d5f0; margin-bottom:8px;'>
                🔬 Biological Interpretation
            </div>
            <div style='font-size:0.88rem; color:#8892a4; line-height:1.5;'>
                This envelope scans biomass flux up to <b>{mx_g:.4f} h⁻¹</b>.
                <ul>
                    <li><b>Max Product:</b> At zero growth, the theoretical max for <i>{env_data.get('product_reaction')}</i> is <b>{max(product_vals):.4f}</b>.</li>
                    <li><b>Yield:</b> Every 1 mmol of biomass produced costs approximately <b style='color:#a78bfa;'>{1.0/yield_ratio if yield_ratio > 1e-8 else "inf"}</b> mmol of product deviation.</li>
                    <li><b>Coupling:</b> {'✅ Growth-coupled production detected' if has_coupling else '❌ No growth-coupling (obligate byproduct) detected in this range'}.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Scientific transparency footer
        st.markdown("""
        <div style='background:rgba(22,27,46,0.5); border:1px solid rgba(79,142,247,0.1);
                    border-radius:10px; padding:14px 18px; margin-top:8px;
                    font-size:0.8rem; color:#4f5a72; line-height:1.8;'>
            <b style='color:#8892a4;'>🔬 Scientific Transparency</b><br>
            • Medium changes modify <b style='color:#c8d5f0;'>exchange reaction lower bounds</b>
              (uptake constraints in the stoichiometric matrix <i>S·v=0</i>).<br>
            • Optimization uses <b style='color:#c8d5f0;'>Flux Balance Analysis (FBA)</b> —
              linear programming under steady-state assumption.<br>
            • The envelope is computed by <b style='color:#c8d5f0;'>parametric scanning</b>
              of the biomass constraint across its feasible range.<br>
            • Results depend entirely on the chosen constraints and objective.
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================
    # FINAL SESSION SUMMARY (Master Report)
    # ==============================================================
    st.markdown("<hr style='border-color:rgba(79,142,247,0.2); margin:40px 0;'>", 
                unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:1.25rem; font-weight:700; color:#c8d5f0; margin-bottom:20px;'>"
                "📊 Final Simulation Master Report</div>", unsafe_allow_html=True)
    
    rep_c1, rep_c2, rep_c3 = st.columns(3)
    
    with rep_c1:
        st.markdown(f"""
        <div style='background:rgba(22,27,46,0.6); padding:15px; border-radius:10px; border:1px solid rgba(79,142,247,0.1); height:160px;'>
            <div style='font-size:0.7rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.05em;'>Environment Status</div>
            <div style='font-size:1.1rem; font-weight:700; color:#4f8ef7; margin:8px 0;'>{str(medium.get('active_preset') or 'Custom').replace('_', ' ').title()}</div>
            <div style='font-size:0.82rem; color:#8892a4; line-height:1.6;'>
                • {medium.get('uptake_enabled_count', 0)} active uptakes<br>
                • {'Aerobic' if medium.get('has_oxygen_uptake') else 'Anaerobic'} conditions
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rep_c2:
        obj_name = obj_info.get("objective_reaction", "Not Set") if obj_info else "None Selected"
        st.markdown(f"""
        <div style='background:rgba(22,27,46,0.6); padding:15px; border-radius:10px; border:1px solid rgba(79,142,247,0.1); height:160px;'>
            <div style='font-size:0.7rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.05em;'>Current Objective</div>
            <div style='font-size:1.1rem; font-weight:700; color:#00d2c1; margin:8px 0;'>{obj_name}</div>
            <div style='font-size:0.82rem; color:#8892a4; line-height:1.6;'>
                • Goal: {str(obj_info.get('direction', 'N/A')).upper() if obj_info else 'N/A'}<br>
                • Baseline: <b style='color:#c8d5f0;'>{obj_info.get('baseline_objective_value', 0.0) if obj_info else 0.0:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rep_c3:
        growth_val = fba_result.get("growth_rate", 0.0) if fba_result else 0.0
        st.markdown(f"""
        <div style='background:rgba(22,27,46,0.6); padding:15px; border-radius:10px; border:1px solid rgba(79,142,247,0.1); height:160px;'>
            <div style='font-size:0.7rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.05em;'>Simulation Status</div>
            <div style='font-size:1.1rem; font-weight:700; color:{'#27ae60' if growth_val > 0 else '#e74c3c'}; margin:8px 0;'>{growth_val:.4f} h⁻¹</div>
            <div style='font-size:0.82rem; color:#8892a4; line-height:1.6;'>
                • Solve: {'✅ Success' if fba_result and fba_result.get('success') else '⚠️ Not Run'}<br>
                • Envelope: {'✅ Completed' if env_data else '⚠️ Pending'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if env_data:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg, rgba(79,142,247,0.08), rgba(0,210,193,0.08)); 
                    padding:20px; border-radius:12px; border:1px solid rgba(79,142,247,0.2); margin-top:24px; text-align:center;'>
            <div style='font-size:0.82rem; color:#8892a4; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em;'>Final Summary Findings</div>
            <div style='font-size:1.1rem; color:#c8d5f0; font-weight:500;'>
                Maximum possible <b style='color:#4f8ef7;'>{env_data.get('product_reaction')}</b> yield is 
                <b style='color:#a78bfa; font-size:1.2rem;'>{max(env_data.get('product_values', [0.0])):.4f}</b> 
                at a growth rate of <b>{env_data.get('growth_values', [0.0])[0]:.4f}</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 Complete Ph1 and Ph2, then run Ph3 to generate your full Master Simulation Report.", icon="💡")



# ==================================================================
# OptKnock Strain Design Tab
# ==================================================================

_OPTKNOCK_TIMEOUT_HINT = (
    "OptKnock can take 2–10 min on large models. "
    "Reduce max_knockouts or use a manual candidate subset to speed it up."
)


def _action_run_optknock(
    biomass_rxn: str,
    product_rxn: str,
    max_ko: int,
    growth_frac: float,
    candidates: Optional[List[str]],
    timeout: int,
) -> None:
    """Call OptKnock backend and store result in session state."""
    payload: Dict[str, Any] = {
        "biomass_reaction": biomass_rxn,
        "product_reaction": product_rxn,
        "max_knockouts": max_ko,
        "growth_fraction": growth_frac,
        "timeout": timeout,
    }
    if candidates:
        payload["candidate_reactions"] = candidates

    status, body = _post_json_raw(
        f"/models/{st.session_state.model_id}/optknock", payload
    )
    if status == 200 and body:
        st.session_state.optknock_result = body
        st.session_state.optknock_error = None
    elif status == 408:
        st.session_state.optknock_error = (
            "⏱ OptKnock timed out. Reduce max_knockouts, growth_fraction, "
            "or provide a smaller candidate_reactions list."
        )
        st.session_state.optknock_result = None
    elif status == 422:
        msg = (body or {}).get("message", "Model infeasible or invalid configuration.")
        st.session_state.optknock_error = f"❌ {msg}"
        st.session_state.optknock_result = None
    elif status == 400:
        msg = (body or {}).get("message", "Computation too expensive.")
        st.session_state.optknock_error = f"⚠️ {msg}"
        st.session_state.optknock_result = None
    elif body is None:
        st.session_state.optknock_error = "❌ Network error. Is the backend running?"
        st.session_state.optknock_result = None
    else:
        msg = (body or {}).get("message", f"Unexpected error (HTTP {status}).")
        st.session_state.optknock_error = f"❌ {msg}"
        st.session_state.optknock_result = None


def render_optknock_tab() -> None:
    """Render the Greedy Strain Design tab."""
    st.markdown(
        """
        <div class='section-heading'>🔬 Greedy Strain Design</div>
        <p style='color:#8892a4;font-size:0.9rem;margin-bottom:1.2rem;'>
        Identifies reaction <b>knockouts</b> that force co-production of your target compound
        as a metabolic by-product of growth.
        <b>Algorithm:</b> greedy sequential LP search (growth-constrained knockout heuristic).
        This approximates OptKnock but does <b>not</b> perform bilevel MILP optimisation.
        Final phenotype (growth + product flux) is validated from a single joint LP solve.
        All LP work is isolated in <code>with model:</code> context — registry model never mutated.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Load reaction list (reuse prod_reactions cache) ───────────────
    if not st.session_state.get("prod_reactions"):
        with st.spinner("Loading reactions…"):
            data = _get(f"/reactions/{st.session_state.model_id}")
            if data and data.get("reactions"):
                st.session_state.prod_reactions = [r["id"] for r in data["reactions"]]

    rxn_list: List[str] = st.session_state.get("prod_reactions") or []

    if not rxn_list:
        st.warning("⚠️ No reactions loaded. Ensure the model is uploaded and backend is running.")
        return

    n_rxns = len(rxn_list)

    # ── Controls ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        biomass_default = (
            st.session_state.get("objective_info", {}) or {}
        ).get("objective_reaction", rxn_list[0] if rxn_list else "")
        biomass_idx = rxn_list.index(biomass_default) if biomass_default in rxn_list else 0
        biomass_rxn = st.selectbox(
            "🌱 Biomass Reaction",
            rxn_list, index=biomass_idx, key="ok_biomass",
        )
    with c2:
        product_rxn = st.selectbox(
            "🧪 Product Reaction",
            rxn_list, index=min(1, len(rxn_list) - 1), key="ok_product",
        )

    sc1, sc2 = st.columns(2)
    with sc1:
        growth_frac = st.slider(
            "🎯 Growth Fraction",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="Min growth = growth_fraction × baseline_growth. Low values allow more product.",
            key="ok_gf",
        )
    with sc2:
        max_ko = st.slider(
            "🔥 Max Knockouts",
            min_value=1, max_value=3, value=1, step=1,
            help="Number of reaction knockouts to test (1–3). More knockouts = exponentially more compute.",
            key="ok_maxko",
        )

    candidates = st.multiselect(
        "🔬 Candidate Reactions (optional, max 300)",
        options=rxn_list,
        default=[],
        key="ok_candidates",
        help="Leave empty to auto-select non-essential internal reactions (recommended).",
    )

    timeout = st.number_input(
        "⏱ Timeout (s)",
        min_value=60, max_value=600, value=300, step=30,
        key="ok_timeout",
    )

    # ── Scientific safeguard warnings ─────────────────────────────────
    if n_rxns > 5_000:
        st.warning(
            f"⚠️ Large model ({n_rxns:,} reactions). OptKnock may be slow. "
            "Provide a candidate_reactions subset for faster results.",
            icon="⚠️",
        )
    if max_ko > 2:
        st.warning(
            "💡 max_knockouts = 3 tests all triple combinations greedily. "
            "This can take 10–30 minutes on large models. Consider max_knockouts = 1 or 2 first.",
            icon="⚠️",
        )
    if candidates and len(candidates) > 300:
        st.error("❌ Candidate list exceeds 300 reactions. Please select ≤ 300.", icon="❌")
        return

    # ── Run button ────────────────────────────────────────────────────
    if st.button("🚀 Run OptKnock", type="primary", key="ok_run_btn"):
        st.session_state.optknock_result = None
        st.session_state.optknock_error = None
        with st.spinner(
            f"Running OptKnock (k={max_ko}, gf={growth_frac:.2f}) — "
            "this may take several minutes…"
        ):
            _action_run_optknock(
                biomass_rxn, product_rxn,
                max_ko, growth_frac,
                candidates or None,
                timeout,
            )
        st.rerun()

    # ── Error display ─────────────────────────────────────────────────
    err = st.session_state.get("optknock_error")
    if err:
        st.error(err)

    # ── Results display ───────────────────────────────────────────────
    result = st.session_state.get("optknock_result")
    if not result:
        return

    st.markdown("<hr style='border-color:rgba(79,142,247,0.12);margin:1.2rem 0;'>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📊 OptKnock Results</div>",
                unsafe_allow_html=True)

    success = result.get("success", False)
    knocked = result.get("knocked_reactions", [])
    pred_growth = result.get("predicted_growth", 0.0)
    pred_product = result.get("predicted_product_flux", 0.0)
    base_growth = result.get("baseline_growth", 0.0)
    base_product = result.get("baseline_product_flux", 0.0)
    fold = result.get("fold_improvement", 1.0)
    cand_tested = result.get("candidates_tested", 0)
    ess_excl = result.get("essential_excluded", 0)
    search_log = result.get("search_log", [])

    if not success or not knocked:
        st.info(
            "🔍 " + result.get("message", "No beneficial knockouts found."),
            icon="ℹ️",
        )
        st.caption(
            f"Candidates tested: {cand_tested} | "
            f"Essential (excluded): {ess_excl}"
        )
        return

    # Knockout badges
    st.markdown("**🗑️ Proposed Knockouts:**")
    badges = " ".join(
        f"`Δ{rxn}`" for rxn in knocked
    )
    st.markdown(badges)

    # Metrics
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🌱 Pred. Growth (h⁻¹)",
              f"{pred_growth:.4f}", f"{pred_growth - base_growth:+.4f}")
    r2.metric("🧪 Pred. Product Flux",
              f"{pred_product:.4f}", f"{pred_product - base_product:+.4f}")
    r3.metric("✨ Fold Improvement", f"×{fold:.2f}")
    r4.metric("🧐 Candidates Tested",
              f"{cand_tested}", f"{ess_excl} essential excluded")

    # Comparison scatter plot
    try:
        import plotly.graph_objects as go
        fig = go.Figure()

        # Connecting line
        fig.add_trace(go.Scatter(
            x=[base_growth, pred_growth],
            y=[base_product, pred_product],
            mode="lines",
            line=dict(color="rgba(79,142,247,0.4)", dash="dash", width=2),
            showlegend=False,
        ))

        # Baseline point
        fig.add_trace(go.Scatter(
            x=[base_growth], y=[base_product],
            mode="markers+text",
            name="Baseline",
            text=["Baseline"],
            textposition="top center",
            marker=dict(size=14, color="#6c7a9c", line=dict(width=2, color="white")),
        ))

        # OptKnock point
        fig.add_trace(go.Scatter(
            x=[pred_growth], y=[pred_product],
            mode="markers+text",
            name="OptKnock " + ", ".join(f"Δ{r}" for r in knocked),
            text=[f"×{fold:.2f}"],
            textposition="top center",
            marker=dict(size=18, color="#4f8ef7",
                        symbol="star",
                        line=dict(width=2, color="white")),
        ))

        fig.update_layout(
            title=dict(
                text="Baseline vs OptKnock Phenotype",
                font=dict(color="#e2e8f0", size=15),
            ),
            xaxis=dict(
                title="Growth Rate (h⁻¹)",
                color="#8892a4", gridcolor="rgba(79,142,247,0.08)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Product Flux (mmol gDW⁻¹ h⁻¹)",
                color="#8892a4", gridcolor="rgba(79,142,247,0.08)",
                zeroline=False,
            ),
            plot_bgcolor="rgba(15,23,42,0.0)",
            paper_bgcolor="rgba(15,23,42,0.0)",
            legend=dict(font=dict(color="#8892a4"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=50, b=40, l=60, r=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as chart_err:
        st.caption(f"(Chart unavailable: {chart_err})")

    # Greedy search log
    if search_log:
        with st.expander("📝 Greedy Search Log", expanded=False):
            for entry in search_log:
                st.markdown(f"- {entry}")

    st.caption(
        f"Candidates tested: {cand_tested} | "
        f"Essential reactions excluded: {ess_excl} | "
        f"Solver: {result.get('solver_status', '')} | "
        f"Growth fraction: {result.get('growth_fraction', 0):.2f}"
    )


# ==================================================================
# Genome Upload Handlers
# ==================================================================

def _handle_genome_upload(uploaded_file) -> None:
    """Handle .fna/.faa upload — start async reconstruction."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    with st.spinner("📡 Sending genome to backend for reconstruction…"):
        resp = _post_file(
            "/upload-genome",
            file_bytes=file_bytes,
            filename=filename,
            extra_data={"solver": st.session_state.solver},
        )

    if resp and resp.get("success"):
        st.session_state.genome_job_id = resp["job_id"]
        st.info(f"🧬 Reconstruction job started. Job ID: `{resp['job_id']}`", icon="🧬")
        st.rerun()
    elif resp:
        st.error(f"❌ {resp.get('message', 'Upload failed.')}", icon="❌")
    else:
        st.error("❌ Backend unreachable.", icon="🚨")


def _poll_genome_job() -> None:
    """Poll the genome reconstruction job and display progress."""
    job_id = st.session_state.genome_job_id
    if not job_id:
        return

    data = _get(f"/genome-job/{job_id}")
    if not data:
        st.warning("Could not fetch job status.", icon="⚠️")
        return

    status = data.get("status", "unknown")
    progress = data.get("progress", 0.0)
    message = data.get("message", "")

    # Status indicator labels
    status_labels = {
        "pending": "⏳ Queued",
        "parsing": "📄 Parsing FASTA",
        "annotating": "🔬 Annotating Genes",
        "mapping": "🗺️ Mapping Pathways",
        "building": "🏗️ Building Model",
        "gap_filling": "🔧 Gap Filling",
        "exporting": "💾 Exporting SBML",
        "completed": "✅ Completed",
        "failed": "❌ Failed",
    }

    st.markdown(f"""
    <div style='background:rgba(22,27,46,0.7); border:1px solid rgba(79,142,247,0.2);
                border-radius:12px; padding:20px; margin:10px 0;'>
        <div style='font-size:1.1rem; font-weight:700; color:#c8d5f0; margin-bottom:8px;'>
            🧬 Genome Reconstruction Pipeline
        </div>
        <div style='font-size:0.9rem; color:#8892a4; margin-bottom:12px;'>
            File: <b>{data.get('filename', '')}</b> · 
            Status: <b style='color:#4f8ef7;'>{status_labels.get(status, status)}</b>
        </div>
        <div style='background:rgba(11,14,26,0.5); border-radius:8px; height:12px; overflow:hidden;'>
            <div style='background:linear-gradient(90deg,#4f8ef7,#00d2c1); height:100%;
                        width:{progress * 100:.0f}%; transition:width 0.3s ease;'></div>
        </div>
        <div style='font-size:0.8rem; color:#4f5a72; margin-top:6px;'>
            {progress * 100:.0f}% — {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if status == "completed":
        model_id = data.get("model_id")
        report = data.get("report", {})
        st.session_state.genome_report = report

        if model_id:
            st.success(f"✅ Model reconstructed and registered! Model ID: `{model_id}`", icon="✅")
            if st.button("🚀 Load Model & Start Simulation", type="primary", key="genome_load_btn"):
                st.session_state.model_id = model_id
                # Build a summary dict compatible with the existing platform
                st.session_state.model_summary = {
                    "success": True,
                    "model_id": model_id,
                    "internal_id": report.get("organism_name", "Reconstructed"),
                    "num_reactions": report.get("num_reactions", 0),
                    "num_metabolites": report.get("num_metabolites", 0),
                    "num_genes": report.get("num_genes", 0),
                    "num_compartments": report.get("num_compartments", 0),
                    "exchange_reactions": report.get("num_exchange_reactions", 0),
                    "objective_reaction": "BIOMASS_Reconstructed",
                    "objective_direction": "max",
                    "solver_name": "glpk",
                }
                st.session_state.genome_job_id = None
                st.rerun()

        _render_reconstruction_report(report)

    elif status == "failed":
        st.error(f"❌ Reconstruction failed: {data.get('error', 'Unknown error')}", icon="❌")
        if st.button("🔄 Try Again", key="genome_retry_btn"):
            st.session_state.genome_job_id = None
            st.rerun()
    else:
        # Auto-refresh every 3 seconds while in progress
        time.sleep(3)
        st.rerun()


def _render_reconstruction_report(report: Dict) -> None:
    """Display the reconstruction report in a structured format."""
    if not report:
        return

    st.markdown("<div class='section-heading'>📋 Reconstruction Report</div>",
                unsafe_allow_html=True)

    # Metrics row 1: Input
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("📄 Input Sequences", f"{report.get('total_sequences', 0):,}")
    r1c2.metric("📏 Avg Length", f"{report.get('avg_sequence_length', 0):.0f} aa")
    r1c3.metric("🔬 Annotated Genes", f"{report.get('annotated_genes', 0):,}")
    r1c4.metric("🧪 EC Numbers", f"{report.get('unique_ec_numbers', 0):,}")

    # Metrics row 2: Model
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("🔗 Reactions", f"{report.get('num_reactions', 0):,}")
    r2c2.metric("⚗️ Metabolites", f"{report.get('num_metabolites', 0):,}")
    r2c3.metric("🧬 Genes (GPR)", f"{report.get('gpr_associations', 0):,}")
    r2c4.metric("📥 Exchanges", f"{report.get('num_exchange_reactions', 0):,}")

    # Coverage & feasibility
    coverage = report.get('annotation_coverage', 0)
    feasible = report.get('biomass_feasible', False)
    growth = report.get('growth_rate', 0.0)
    gaps = report.get('gap_filled_reactions', 0)

    st.markdown(f"""
    <div style='display:flex; gap:16px; margin-top:16px;'>
        <div style='flex:1; background:rgba(22,27,46,0.6); padding:14px; border-radius:10px;
                    border:1px solid rgba(79,142,247,0.1);'>
            <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase;'>Annotation Coverage</div>
            <div style='font-size:1.3rem; font-weight:700; color:{'#27ae60' if coverage > 30 else '#e74c3c'}; margin-top:4px;'>
                {coverage:.1f}%
            </div>
        </div>
        <div style='flex:1; background:rgba(22,27,46,0.6); padding:14px; border-radius:10px;
                    border:1px solid rgba(79,142,247,0.1);'>
            <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase;'>Gap-Filled Reactions</div>
            <div style='font-size:1.3rem; font-weight:700; color:#a78bfa; margin-top:4px;'>{gaps}</div>
        </div>
        <div style='flex:1; background:rgba(22,27,46,0.6); padding:14px; border-radius:10px;
                    border:1px solid rgba(79,142,247,0.1);'>
            <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase;'>Biomass Feasibility</div>
            <div style='font-size:1.3rem; font-weight:700; color:{'#27ae60' if feasible else '#e74c3c'}; margin-top:4px;'>
                {'✅ Feasible' if feasible else '❌ Infeasible'}
            </div>
        </div>
        <div style='flex:1; background:rgba(22,27,46,0.6); padding:14px; border-radius:10px;
                    border:1px solid rgba(79,142,247,0.1);'>
            <div style='font-size:0.72rem; color:#8892a4; text-transform:uppercase;'>Max Growth Rate</div>
            <div style='font-size:1.3rem; font-weight:700; color:#c8d5f0; margin-top:4px;'>{growth:.4f} h⁻¹</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Transparency notice
    st.markdown(f"""
    <div style='background:rgba(79,142,247,0.05); border:1px solid rgba(79,142,247,0.15);
                border-radius:10px; padding:14px; margin-top:16px;
                font-size:0.85rem; color:#8892a4; line-height:1.6;'>
        <b style='color:#c8d5f0;'>⚠️ Reconstruction Notice</b><br>
        This model was <b>automatically reconstructed</b> from genomic data and may require manual curation.
        Annotation coverage of <b>{coverage:.1f}%</b> means {100 - coverage:.1f}% of input sequences
        did not match known enzymes. Gap-filling added <b>{gaps}</b> reaction(s) to achieve biomass feasibility.
        <br><br>
        <i style='color:#4f5a72;'>Pipeline: FASTA → KEGG Annotation → EC→Reaction Mapping → COBRApy Construction → Gap-Fill → SBML</i>
    </div>
    """, unsafe_allow_html=True)

    # Warnings
    warnings = report.get("warnings", [])
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} Warning(s)", expanded=False):
            for w in warnings:
                st.warning(w, icon="⚠️")

    errors = report.get("errors", [])
    if errors:
        with st.expander(f"❌ {len(errors)} Error(s)", expanded=True):
            for e in errors:
                st.error(e, icon="❌")

    st.caption(f"⏱️ Total reconstruction time: {report.get('total_time_seconds', 0):.1f}s")


def main() -> None:
    render_sidebar()
    render_upload_section()

    if st.session_state.model_id is None:
        return

    tab_summary, tab_diagnostics, tab_validation, tab_fva, tab_sim, tab_optknock = st.tabs(
        [
            "📊 Model Summary",
            "📈 FBA Diagnostics",
            "🔍 Validation",
            "📊 Flux Variability",
            "🌐 Environment → Outcome",
            "🔬 Strain Design",
        ]
    )
    with tab_summary:
        render_summary_tab()
    with tab_diagnostics:
        render_diagnostics_tab()
    with tab_validation:
        render_validation_tab()
    with tab_fva:
        render_fva_tab()
    with tab_sim:
        render_env_simulation_tab()
    with tab_optknock:
        render_optknock_tab()


if __name__ == "__main__":
    main()
