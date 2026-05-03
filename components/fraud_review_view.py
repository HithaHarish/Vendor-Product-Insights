from __future__ import annotations

import pandas as pd
import streamlit as st

from services.explainability.genai_reasoning import generate_llm_explanation
from services.explainability.shap_explainer import build_reasoning_summary, get_local_explanation
from services.formatting.format import format_llm_output
from services.models.predict import predict_review


def render_fraud_review_view() -> None:
    st.markdown("## Fraud Review Detection")
    training_df = st.session_state.get("training_df")
    saved = st.session_state.get("model_saved")
    if training_df is None:
        st.info("Run Agentic Workflow first.")
        return

    st.markdown(
        """
        <div class="section-heading">Fraud Review Datasets</div>
        <div class="section-underline"></div>
        """,
        unsafe_allow_html=True,
    )
    preprocessed_samples = st.session_state.get("preprocessed_samples")
    if preprocessed_samples:
        with st.expander("Preprocessed Feature Samples", expanded=False):
            for title, df in preprocessed_samples.items():
                st.markdown(f"<div class='dataset-heading'>{title}</div>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                st.divider()

    genuine_df = st.session_state.get("genuine_reviews_df", pd.DataFrame())
    fraud_df = st.session_state.get("fraud_reviews_df", pd.DataFrame())
    if not genuine_df.empty or not fraud_df.empty:
        gd, fd = st.columns(2)
        gd.download_button(
            label="Download Genuine Reviews CSV",
            data=genuine_df.to_csv(index=False).encode("utf-8"),
            file_name="genuine_reviews.csv",
            mime="text/csv",
            key="download_genuine_top",
        )
        fd.download_button(
            label="Download Fraud Reviews CSV",
            data=fraud_df.to_csv(index=False).encode("utf-8"),
            file_name="fraud_reviews.csv",
            mime="text/csv",
            key="download_fraud_top",
        )

    st.markdown(
        """
        <div class="section-heading">Model Details</div>
        <div class="section-underline"></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <ul class="section-text model-list">
        <li><b>Model Trained :</b> XGBoost Classifier</li>
        <li><b>Accuracy :</b> {st.session_state.get("accuracy", 0.0) * 100:.2f}%</li>
        <li><b>ROC-AUC Score :</b> {st.session_state.get("roc_auc", 0.0):.4f}</li>
        <li><b>Number of Features Used :</b> {training_df.shape[1]}</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    if not saved:
        st.info("Model not available yet.")
        return

    reviews_df = st.session_state.get("reviews_df")
    users_df = st.session_state.get("users_df")
    products_df = st.session_state.get("products_df")
    platforms_df = st.session_state.get("platforms_df")
    model = saved["model"]
    feature_list = saved["features"]

    st.markdown(
        """
        <div class="section-heading">Fraud Risk Inspection</div>
        <div class="section-underline"></div>
        """,
        unsafe_allow_html=True,
    )

    review_ids = training_df["review_id"].unique()
    selected_review_id = st.selectbox("Select Review", review_ids, key="review_selector")
    selected_row = training_df[training_df["review_id"] == selected_review_id]
    if selected_row.empty:
        return
    selected_row = selected_row.copy()
    customer_id = selected_row["customer_id"].values[0]
    product_id = selected_row["product_id"].values[0]

    def _col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _val(row, *candidates, default="-"):
        if row is None:
            return default
        for c in candidates:
            if c in row.index:
                v = row[c]
                return "-" if pd.isna(v) else str(v)
        return default

    rid_col = _col(reviews_df, "review_id", "Review_ID") or "review_id"
    product_col = _col(products_df, "product_id", "Product_ID") or "product_id"
    customer_col = _col(users_df, "customer_id", "Customer_ID") or "customer_id"
    review_row = reviews_df[reviews_df[rid_col] == selected_review_id]
    review_row = review_row.iloc[0] if not review_row.empty else None
    product_row = products_df[products_df[product_col] == product_id]
    product_row = product_row.iloc[0] if not product_row.empty else None
    user_row = users_df[users_df[customer_col] == customer_id]
    user_row = user_row.iloc[0] if not user_row.empty else None

    platform_name = ""
    if review_row is not None:
        pid_col = _col(reviews_df, "platform_id", "Platform_ID")
        if pid_col:
            pid = review_row[pid_col]
            plat_id_col = _col(platforms_df, "platform_id", "Platform_ID")
            plat_name_col = _col(platforms_df, "platform_name", "Platform_Name")
            if plat_id_col and plat_name_col:
                plat = platforms_df[platforms_df[plat_id_col].astype(str) == str(pid)]
                if not plat.empty:
                    platform_name = plat.iloc[0][plat_name_col]

    col3, col4 = st.columns([3, 1])
    with col3:
        original_review = _val(review_row, "review_text", "Review_Text", default="")
        st.text_area("Review Text", original_review, height=150)
    prediction, risk_score = predict_review(model, feature_list, selected_row)
    with col4:
        st.metric("Fraud Risk Score", f"{risk_score * 100:.2f}%")
        if risk_score < 0.30:
            label = "Genuine Review"
            reason = "0–30%: Genuine range"
        elif risk_score < 0.60:
            label = "Suspicious Review"
            reason = "30–60%: Suspicious range"
        else:
            label = "Fraudulent Review"
            reason = "60–100%: Fraud range"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"({reason})")

    st.divider()
    st.markdown("<div class='dataset-heading'>Review Details</div>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.text_input("Rating", value=_val(review_row, "rating", "Rating"), disabled=True, key="ri_rating")
        st.text_input("Verified Purchase", value=_val(review_row, "verified_purchase", "Verified_Purchase"), disabled=True, key="ri_verified")
    with r2:
        st.text_input("Platform", value=platform_name or _val(review_row, "platform_id", "Platform_ID"), disabled=True, key="ri_platform")
        st.text_input("Refunded", value=_val(review_row, "refunded_product", "Refunded_Product"), disabled=True, key="ri_refunded")
    if "rating_sentiment_mismatch" in selected_row.columns:
        try:
            mm = int(selected_row["rating_sentiment_mismatch"].values[0])
            st.text_input("Rating–Sentiment Mismatch", value="Yes" if mm == 1 else "No", disabled=True, key="ri_rating_mismatch")
        except Exception:
            pass

    st.markdown("<div class='dataset-heading'>Product Details</div>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.text_input("Name", value=_val(product_row, "name", "Name", "product_name"), disabled=True, key="pi_name")
        st.text_input("Brand", value=_val(product_row, "brand", "Brand"), disabled=True, key="pi_brand")
    with p2:
        st.text_input("Category", value=_val(product_row, "category", "Category"), disabled=True, key="pi_category")

    st.markdown("<div class='dataset-heading'>Customer Details</div>", unsafe_allow_html=True)
    st.text_input("Account Created", value=_val(user_row, "account_created", "Account_Created"), disabled=True, key="ui_account_created")
    st.divider()

    if risk_score <= 0.0:
        return

    explanation_data = get_local_explanation(
        model=model,
        selected_row=selected_row,
        feature_list=feature_list,
        risk_score=risk_score,
    )
    if explanation_data is None or not explanation_data.get("top_features"):
        return

    st.markdown("<div class='dataset-heading'>Top Contributing Parameters</div>", unsafe_allow_html=True)
    top_df = pd.DataFrame(
        [
            {
                "Feature": f["feature"].replace("_", " ").title(),
                "Value": f["feature_value"],
                "Impact": f["shap_impact"],
            }
            for f in explanation_data["top_features"]
        ]
    )
    st.dataframe(top_df, use_container_width=True)
    st.markdown("<div class='dataset-heading'>LLM-based Explanation</div>", unsafe_allow_html=True)
    evidence = {
        "rating": _val(review_row, "rating", "Rating"),
        "verified_purchase": _val(review_row, "verified_purchase", "Verified_Purchase"),
        "refunded_product": _val(review_row, "refunded_product", "Refunded_Product"),
        "sentiment_score": float(selected_row.get("sentiment_score", 0)),
        "sentiment_intensity": float(selected_row.get("sentiment_intensity", 0)),
        "product_relevance": float(selected_row.get("product_relevance_score", 0)),
        "rating_sentiment_mismatch": int(selected_row.get("rating_sentiment_mismatch", 0)),
        "punctuation_density": float(selected_row.get("punctuation_density", 0)),
        "review_length": int(selected_row.get("review_length", 0)),
    }
    summary = generate_llm_explanation(
        review_text=original_review,
        risk_score=risk_score,
        top_features=explanation_data["top_features"],
        prediction_label=label,
        evidence=evidence,
    )
    if summary:
        formatted_summary = format_llm_output(summary, original_review)
        st.markdown(formatted_summary)
    else:
        st.markdown(build_reasoning_summary(explanation_data))
