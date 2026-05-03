import streamlit as st
import pandas as pd

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def ensure_columns(df, required_cols, default_values=None):
    default_values = default_values or {}

    for col in required_cols:
        if col not in df.columns:
            df[col] = default_values.get(col, 0)

    return df[required_cols].copy()


def load_file(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    return df


# --------------------------------------------------
# SIDEBAR UI
# --------------------------------------------------

def render_sidebar():
    st.sidebar.markdown("### Upload Dataset Folder")

    uploaded_files = st.sidebar.file_uploader(
        "Upload all CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    raw_data = {
        "customers": None,
        "products": None,
        "orders": None,
        "reviews": None,
        "platforms": None
    }

    clean_data = {
        "customers": None,
        "products": None,
        "orders": None,
        "reviews": None,
        "platforms": None
    }

    if uploaded_files:
        for file in uploaded_files:
            name = file.name.lower()

            try:
                df = load_file(file)

                # ✅ DISPLAY RAW DATA
                st.write(f"{file.name}")
                st.dataframe(df.head())

                if "customer" in name:
                    raw_data["customers"] = df
                    clean_data["customers"] = ensure_columns(
                        df,
                        ["customer_id", "account_created"]
                    )

                elif "product" in name:
                    raw_data["products"] = df
                    clean_data["products"] = ensure_columns(
                        df,
                        ["product_id", "name", "category", "brand"]
                    )

                elif "order" in name:
                    raw_data["orders"] = df
                    clean_data["orders"] = ensure_columns(
                        df,
                        ["order_id", "customer_id", "product_id", "order_timestamp"]
                    )

                elif "review" in name:
                    raw_data["reviews"] = df
                    clean_data["reviews"] = ensure_columns(
                        df,
                        [
                            "review_id",
                            "customer_id",
                            "product_id",
                            "rating",
                            "review_text",
                            "review_timestamp",
                            "verified_purchase",
                            "refunded_product",
                            "platform_id"
                        ],
                        default_values={
                            "review_text": "",
                            "verified_purchase": 0,
                            "refunded_product": 0,
                            "rating": 0
                        }
                    )

                elif "platform" in name:
                    raw_data["platforms"] = df
                    clean_data["platforms"] = ensure_columns(
                        df,
                        ["platform_id", "platform_name"]
                    )

            except Exception as e:
                st.sidebar.error(f"Error processing {file.name}: {e}")

    return raw_data, clean_data