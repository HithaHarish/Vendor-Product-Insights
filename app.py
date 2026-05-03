from __future__ import annotations

import pandas as pd
import streamlit as st

from components.agents_work_view import render_agents_work_view
from components.fraud_review_view import render_fraud_review_view
from components.header import render_header
from components.product_insights_view import render_product_insights_view
from components.sections import render_intro_sections
from components.sidebar import render_sidebar
from components.styles import load_styles
from services.agentic.workflow import run_agentic_workflow

st.set_page_config(page_title="Review Fraud Detection Dashboard", layout="wide")


def _ensure_session_keys() -> None:
    defaults = {
        "processed": False,
        "preprocessed_samples": None,
        "model_saved": False,
        "training_df": None,
        "accuracy": 0.0,
        "roc_auc": 0.0,
        "agent_logs": [],
        "genuine_reviews_df": pd.DataFrame(),
        "fraud_reviews_df": pd.DataFrame(),
        "active_view": "Fraud Review Detection",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    load_styles()
    render_header()
    render_intro_sections()
    _, clean_data = render_sidebar()

    users_df = clean_data["customers"]
    products_df = clean_data["products"]
    orders_df = clean_data["orders"]
    reviews_df = clean_data["reviews"]
    platforms_df = clean_data["platforms"]
    if any(x is None for x in [users_df, products_df, orders_df, reviews_df, platforms_df]):
        st.warning("Please upload all required datasets.")
        return

    _ensure_session_keys()
    st.markdown("### Sections")
    b1, b2, b3 = st.columns(3)
    if b1.button("Fraud Review Detection", key="view_btn_fraud", use_container_width=True):
        st.session_state.active_view = "Fraud Review Detection"
    if b2.button("Agents Work", key="view_btn_agents", use_container_width=True):
        st.session_state.active_view = "Agents Work"
    if b3.button("Product Insights", key="view_btn_insights", use_container_width=True):
        st.session_state.active_view = "Product Insights"
    active = st.session_state.active_view

    if st.button("Run Agentic Workflow (6 Agents)", key="run_agentic_workflow_btn"):
        st.session_state.processed = True
        st.session_state.training_df = None
        st.session_state.active_view = "Agents Work"
        active = st.session_state.active_view

    if st.session_state.processed and st.session_state.training_df is None:
        with st.spinner("Running 6-agent workflow across all layers..."):
            run_result = run_agentic_workflow(
                users_df=users_df,
                products_df=products_df,
                orders_df=orders_df,
                reviews_df=reviews_df,
                platforms_df=platforms_df,
            )
        st.session_state.preprocessed_samples = run_result["preprocessed_samples"]
        st.session_state.training_df = run_result["training_df"]
        st.session_state.model_saved = run_result["saved_model"]
        st.session_state.reviews_df = reviews_df
        st.session_state.users_df = users_df
        st.session_state.products_df = products_df
        st.session_state.orders_df = orders_df
        st.session_state.platforms_df = platforms_df
        st.session_state.accuracy = run_result["accuracy"]
        st.session_state.roc_auc = run_result["roc_auc"]
        st.session_state.agent_logs = run_result["agent_logs"]
        st.session_state.genuine_reviews_df = run_result["genuine_reviews_df"]
        st.session_state.fraud_reviews_df = run_result["fraud_reviews_df"]
        st.session_state.active_view = "Agents Work"
        active = st.session_state.active_view

    if active == "Fraud Review Detection":
        render_fraud_review_view()
    elif active == "Agents Work":
        render_agents_work_view()
    else:
        render_product_insights_view()

if __name__ == "__main__":
    main()
