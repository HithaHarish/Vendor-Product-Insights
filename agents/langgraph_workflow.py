from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from agents.prompts import LAYER_PROMPTS


class AgentState(TypedDict, total=False):
    datasets: Dict[str, Any]
    layer_outputs: Dict[str, Any]
    run_log: list[Dict[str, Any]]


@dataclass
class LayerSpec:
    key: str
    prompt: str
    title: str


LAYER_SPECS = [
    LayerSpec("layer_1_data_understanding", LAYER_PROMPTS["layer_1_data_understanding"], "Layer 1 - Data Understanding"),
    LayerSpec("layer_2_textual_intelligence", LAYER_PROMPTS["layer_2_textual_intelligence"], "Layer 2 - Textual Intelligence"),
    LayerSpec("layer_3_behavioral_intelligence", LAYER_PROMPTS["layer_3_behavioral_intelligence"], "Layer 3 - Behavioral Intelligence"),
    LayerSpec("layer_4_temporal_intelligence", LAYER_PROMPTS["layer_4_temporal_intelligence"], "Layer 4 - Temporal Intelligence"),
    LayerSpec("layer_5_fraud_scoring", LAYER_PROMPTS["layer_5_fraud_scoring"], "Layer 5 - Fraud Scoring"),
    LayerSpec("layer_6_explainability_and_export", LAYER_PROMPTS["layer_6_explainability_and_export"], "Layer 6 - Explainability and Export"),
]


def _run_layer(layer: LayerSpec, state: AgentState) -> AgentState:
    layer_outputs = dict(state.get("layer_outputs", {}))
    run_log = list(state.get("run_log", []))

    input_payload = {
        "datasets_present": list((state.get("datasets") or {}).keys()),
        "previous_layers": list(layer_outputs.keys()),
    }
    output_payload = {
        "layer_title": layer.title,
        "agent_prompt": layer.prompt,
        "status": "completed",
    }

    layer_outputs[layer.key] = output_payload
    run_log.append(
        {
            "layer_key": layer.key,
            "layer_title": layer.title,
            "input_payload": input_payload,
            "output_payload": output_payload,
        }
    )

    state["layer_outputs"] = layer_outputs
    state["run_log"] = run_log
    return state


def build_langgraph_app():
    """
    Builds a LangGraph application for six-layer agent orchestration.
    """
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "LangGraph is not installed. Install it to build this workflow."
        ) from exc

    graph = StateGraph(AgentState)

    for layer in LAYER_SPECS:
        graph.add_node(layer.key, lambda state, _layer=layer: _run_layer(_layer, state))

    graph.set_entry_point(LAYER_SPECS[0].key)
    for i in range(len(LAYER_SPECS) - 1):
        graph.add_edge(LAYER_SPECS[i].key, LAYER_SPECS[i + 1].key)
    graph.add_edge(LAYER_SPECS[-1].key, END)

    return graph.compile()


def run_langgraph_workflow(datasets: Dict[str, Any]) -> AgentState:
    """
    Executes the six-layer LangGraph orchestration.
    """
    app = build_langgraph_app()
    initial_state: AgentState = {
        "datasets": datasets,
        "layer_outputs": {},
        "run_log": [],
    }
    return app.invoke(initial_state)

