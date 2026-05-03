from __future__ import annotations

import streamlit as st


NAV_ITEMS = {
    "fraud": "Fraud Review Detection",
    "agents": "Agents Work",
    "insights": "Product Insights",
}


def render_navbar() -> str:
    """Render fixed top navbar with mild professional colors."""
    current = st.query_params.get("page", "fraud")
    if current not in NAV_ITEMS:
        current = "fraud"

    def _btn(label: str, page_key: str) -> str:
        cls = "top-nav-link active" if current == page_key else "top-nav-link"
        return f'<a class="{cls}" href="?page={page_key}">{label}</a>'

    nav_html = (
        '<div class="top-nav-shell">'
        '<div class="top-nav-inner">'
        f'{_btn(NAV_ITEMS["fraud"], "fraud")}'
        f'{_btn(NAV_ITEMS["agents"], "agents")}'
        f'{_btn(NAV_ITEMS["insights"], "insights")}'
        "</div>"
        "</div>"
        '<div class="top-nav-spacer"></div>'
    )

    st.markdown(
        """
        <style>
        .top-nav-shell{
            position: fixed;
            top: 3.35rem; /* keep below Streamlit top bar */
            left: 400px; /* align with main content when sidebar is open */
            right: 0;
            z-index: 1000;
            background: #f4f7fb;
            border-bottom: 1px solid #d2dbe6;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
            padding: 10px 20px;
        }
        .top-nav-inner{
            width: 100%;
            margin: 0;
            display: flex;
            gap: 10px;
        }
        .top-nav-link{
            flex: 1;
            text-align: center;
            text-decoration: none;
            background: #e7edf4;
            color: #1f3348;
            border: 1px solid #c3d0df;
            border-radius: 10px;
            padding: 11px 14px;
            font-weight: 600;
            font-size: 14px;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.18s ease-in-out;
        }
        .top-nav-link:hover{
            background: #dce6f1;
            color: #112538;
            border-color: #b7c8da;
        }
        .top-nav-link.active{
            background: #3f5f80;
            color: #ffffff;
            border-color: #324f6b;
        }
        .top-nav-spacer{
            height: 106px;
        }

        @media (max-width: 1200px){
            .top-nav-shell{
                left: 0;
                padding-left: 14px;
                padding-right: 14px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(nav_html, unsafe_allow_html=True)
    return NAV_ITEMS[current]
