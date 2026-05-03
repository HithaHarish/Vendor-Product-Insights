from __future__ import annotations

import streamlit as st


def render_agents_work_view() -> None:
    st.markdown("## Agents Work")
    logs = st.session_state.get("agent_logs", [])


    st.caption("Each layer shows running agent with input and output payload.")
    for log in logs:
        title = f'{log["layer"]} · {log["agent_name"]} ({log["agent_role"]})'
        with st.expander(title, expanded=False):
            ci, co = st.columns(2)
            ci.markdown("**Agent Input**")
            ci.json(log["input_payload"])
            co.markdown("**Agent Output**")
            co.json(log["output_payload"])
