from __future__ import annotations

from typing import Any, TypedDict

from agent import QAAgent


class AgentState(TypedDict):
    messages: list[dict[str, str]]
    session_id: str


def build_agent_graph(agent: QAAgent, checkpointer: Any | None = None):
    try:
        from langgraph.graph import END, START, MessagesState, StateGraph
    except ImportError as error:
        raise RuntimeError(
            "Integrarea LangGraph necesita pachetul langgraph. "
            "Instaleaza dependintele din requirements.txt."
        ) from error

    class GraphState(MessagesState):
        session_id: str

    def agent_node(state: AgentState) -> dict[str, list[dict[str, str]]]:
        messages = state.get("messages", [])

        if not messages:
            raise ValueError("AgentState trebuie sa contina cel putin un mesaj user.")

        user_message = messages[-1]
        role = _message_role(user_message)

        if role not in {"user", "human"}:
            raise ValueError("Ultimul mesaj din AgentState trebuie sa aiba role='user'.")

        session_id = state.get("session_id") or "default"
        reply = agent.answer(_message_content(user_message), session_id=session_id)

        return {"messages": [{"role": "assistant", "content": reply}]}

    builder = StateGraph(GraphState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    return builder.compile(checkpointer=checkpointer)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")

    return getattr(message, "type", None) or getattr(message, "role", None)


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return message["content"]

    return str(getattr(message, "content"))
