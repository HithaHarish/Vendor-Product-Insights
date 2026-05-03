from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from services.build_training_dataset import build_training_dataset
from services.models.predict import load_model, predict_review
from services.models.xgboost_model import train_xgboost_model
from services.preprocessing.behavioral_product import product_behavioral_features
from services.preprocessing.behavioral_user import customer_behavioral_features
from services.preprocessing.temporal import temporal_features
from services.preprocessing.textual import textual_training_dataset


@dataclass
class AgentLog:
    layer: str
    agent_name: str
    agent_role: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]


def _df_preview(df: pd.DataFrame, rows: int = 2) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    preview = df.head(rows).copy()
    preview.columns = [str(c) for c in preview.columns]
    return preview.to_dict("records")


def run_agentic_workflow(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    platforms_df: pd.DataFrame,
    genuine_threshold: float = 0.30,
) -> dict[str, Any]:
    logs: list[AgentLog] = []

    # Agent 1 - Data Understanding
    agent1_output = {
        "users_shape": list(users_df.shape),
        "products_shape": list(products_df.shape),
        "orders_shape": list(orders_df.shape),
        "reviews_shape": list(reviews_df.shape),
        "platforms_shape": list(platforms_df.shape),
        "users_columns": users_df.columns.tolist(),
        "products_columns": products_df.columns.tolist(),
        "orders_columns": orders_df.columns.tolist(),
        "reviews_columns": reviews_df.columns.tolist(),
        "platforms_columns": platforms_df.columns.tolist(),
    }
    logs.append(
        AgentLog(
            layer="Layer 1 - Data Understanding",
            agent_name="Agent 1",
            agent_role="Data Intake Agent",
            input_payload={"datasets": ["customers", "products", "orders", "reviews", "platforms"]},
            output_payload=agent1_output,
        )
    )

    # Agent 2 - Textual
    textual_df, vectorizer = textual_training_dataset(reviews_df, products_df)
    logs.append(
        AgentLog(
            layer="Layer 2 - Textual Intelligence",
            agent_name="Agent 2",
            agent_role="Textual Feature Agent",
            input_payload={
                "reviews_rows": len(reviews_df),
                "products_rows": len(products_df),
                "key_fields": ["review_text", "rating", "verified_purchase", "refunded_product"],
            },
            output_payload={
                "textual_rows": len(textual_df),
                "textual_columns": textual_df.columns.tolist(),
                "textual_preview": _df_preview(textual_df),
                "vectorizer_built": vectorizer is not None,
            },
        )
    )

    # Agent 3 - Behavioral
    product_behavior_df = product_behavioral_features(reviews_df)
    customer_behavior_df = customer_behavioral_features(users_df, reviews_df)
    logs.append(
        AgentLog(
            layer="Layer 3 - Behavioral Intelligence",
            agent_name="Agent 3",
            agent_role="Behavioral Pattern Agent",
            input_payload={
                "users_rows": len(users_df),
                "reviews_rows": len(reviews_df),
                "behavior_signals": ["account_age", "review_frequency", "refund_ratio", "verified_purchase_ratio"],
            },
            output_payload={
                "product_behavior_rows": len(product_behavior_df),
                "customer_behavior_rows": len(customer_behavior_df),
                "product_behavior_preview": _df_preview(product_behavior_df),
                "customer_behavior_preview": _df_preview(customer_behavior_df),
            },
        )
    )

    # Agent 4 - Temporal
    product_temporal_df, customer_temporal_df = temporal_features(reviews_df)
    logs.append(
        AgentLog(
            layer="Layer 4 - Temporal Intelligence",
            agent_name="Agent 4",
            agent_role="Temporal Burst Agent",
            input_payload={
                "reviews_rows": len(reviews_df),
                "temporal_fields": ["review_timestamp", "product_id", "customer_id"],
            },
            output_payload={
                "product_temporal_rows": len(product_temporal_df),
                "customer_temporal_rows": len(customer_temporal_df),
                "product_temporal_preview": _df_preview(product_temporal_df),
                "customer_temporal_preview": _df_preview(customer_temporal_df),
            },
        )
    )

    # Agent 5 - Fraud Scoring
    training_df = build_training_dataset(
        textual_df,
        reviews_df,
        customer_behavior_df,
        product_behavior_df,
        product_temporal_df,
        customer_temporal_df,
    )
    model, accuracy, roc_auc = train_xgboost_model(training_df)
    saved = load_model()
    feature_list = saved["features"]
    logs.append(
        AgentLog(
            layer="Layer 5 - Fraud Scoring",
            agent_name="Agent 5",
            agent_role="XGBoost Risk Agent",
            input_payload={
                "training_rows": len(training_df),
                "training_columns": training_df.columns.tolist(),
                "label_distribution": training_df["label"].value_counts(dropna=False).to_dict()
                if "label" in training_df.columns
                else {},
            },
            output_payload={
                "model": "XGBoost Classifier",
                "accuracy": float(accuracy),
                "roc_auc": float(roc_auc),
                "feature_count": len(feature_list),
                "model_saved_path": "services/models/xgb_fraud_model.pkl",
            },
        )
    )

    # Agent 6 - Explainability and clean export preparation
    genuine_review_ids: list[Any] = []
    risk_scores: list[float] = []
    for idx in training_df.index:
        row_df = training_df.loc[[idx]]
        _, score = predict_review(model, feature_list, row_df)
        row = training_df.loc[idx]
        risk_scores.append(float(score))
        if score < genuine_threshold:
            genuine_review_ids.append(row["review_id"])

    genuine_reviews_df = reviews_df[reviews_df["review_id"].isin(genuine_review_ids)].copy()
    fraud_reviews_df = reviews_df[~reviews_df["review_id"].isin(genuine_review_ids)].copy()
    logs.append(
        AgentLog(
            layer="Layer 6 - Explainability and Export",
            agent_name="Agent 6",
            agent_role="Decision and Export Agent",
            input_payload={
                "scored_reviews": len(training_df),
                "genuine_threshold": genuine_threshold,
                "explainability_inputs": ["top_features", "llm_reasoning", "risk_bands"],
            },
            output_payload={
                "genuine_reviews_count": len(genuine_reviews_df),
                "flagged_reviews_count": int(len(training_df) - len(genuine_reviews_df)),
                "risk_score_min": min(risk_scores) if risk_scores else 0.0,
                "risk_score_max": max(risk_scores) if risk_scores else 0.0,
                "genuine_preview": _df_preview(genuine_reviews_df),
            },
        )
    )

    preprocessed_samples = {
        "Textual Features": textual_df.head(2),
        "Product Behavioral Features": product_behavior_df.head(2),
        "Customer Behavioral Features": customer_behavior_df.head(2),
        "Product Temporal Features": product_temporal_df.head(2),
        "Customer Temporal Features": customer_temporal_df.head(2),
    }

    return {
        "agent_logs": [a.__dict__ for a in logs],
        "training_df": training_df,
        "model": model,
        "saved_model": saved,
        "preprocessed_samples": preprocessed_samples,
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "genuine_reviews_df": genuine_reviews_df,
        "fraud_reviews_df": fraud_reviews_df,
    }
