import xgboost as xgb
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os

def train_xgboost_model(training_df):
    print("XGBoost training function loaded")
    
    if training_df is None or training_df.empty:
        return None, 0.0, 0.0

    if "label" not in training_df.columns:
        # Provide a fallback just in case
        training_df["label"] = 0

    y = training_df["label"]
    
    drop_cols = ["review_id", "customer_id", "product_id", "label", "review_timestamp", "order_timestamp", "account_created", "review_text", "name", "category", "brand", "platform_name", "platform_id"]
    X = training_df.drop(columns=[col for col in drop_cols if col in training_df.columns])
    
    # Ensure numeric types
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    feature_list = X.columns.tolist()

    if len(training_df) > 1 and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else y_pred
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = 0.5 # fallback if only one class is present in y_test

    os.makedirs("services/models", exist_ok=True)
    saved = {"model": model, "features": feature_list}
    joblib.dump(saved, "services/models/xgb_fraud_model.pkl")

    return model, accuracy, roc_auc