from dataclasses import dataclass
from tempfile import TemporaryDirectory

from agent import QAAgent
from intent_classifier import load_examples, load_model, predict_intent, train_intent_classifier
from memory import InMemoryConversationMemory
from prompt_cache import PromptCacheMetrics
from run_agent import LiteLLMClient
from tools import ToolWrapper
import tools.basic_tools  # important: registers tools


MONTHS_RO = {
    "ianuarie",
    "februarie",
    "martie",
    "aprilie",
    "mai",
    "iunie",
    "iulie",
    "august",
    "septembrie",
    "octombrie",
    "noiembrie",
    "decembrie",
}


@dataclass
class FakeResponse:
    content: str
    tool_calls: list[dict] | None = None


class DateQuestionLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> FakeResponse:
        system_prompt = messages[0]["content"]

        if tools is None:
            if "componenta de extracție" in system_prompt:
                assert "Nu menționa tool-uri" in system_prompt
                assert "Data curentă:" in system_prompt
                assert "Ora curentă:" in system_prompt
                return FakeResponse(content="- Data curentă este 26 mai 2026.\n- Ora curentă este 11:22.")

            if "Ești analistul" in system_prompt:
                assert "Nu menționa tool-uri" in system_prompt
                assert "Data curentă este 26 mai 2026" in system_prompt
                return FakeResponse(content="Utilizatorul întreabă data curentă.")

            if "componenta de sumarizare" in system_prompt:
                assert "Nu menționa niciodată tool-uri" in system_prompt
                assert "Data curentă este 26 mai 2026" in system_prompt
                return FakeResponse(content="Astăzi este 26 mai 2026, ora 11:22.")

            raise AssertionError("Unexpected prompt without tools.")

        if messages[-1]["role"] == "tool":
            assert "Data curentă:" in messages[-1]["content"]
            assert "Ora curentă:" in messages[-1]["content"]
            return FakeResponse(content="Am suficiente informații pentru răspuns.")

        assert tools, "Agent did not pass tools to the LLM."
        assert "Reguli pentru planificare" in system_prompt
        assert "Alege tool-urile potrivite" in system_prompt

        tool_names = {tool["name"] for tool in tools}
        assert "get_current_datetime" in tool_names
        assert "format_date" not in tool_names
        assert "format_datetime" not in tool_names

        self.calls += 1
        assert self.calls == 1
        return FakeResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_date",
                    "type": "function",
                    "function": {
                        "name": "get_current_datetime",
                        "arguments": "{}",
                    },
                }
            ],
        )


class InventedFormatToolLLM:
    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> FakeResponse:
        assert tools, "Agent did not pass tools to the LLM."
        return FakeResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "format_date",
                        "arguments": '{"result": "2026-05-26T11:26:41.421217+03:00"}',
                    },
                }
            ],
        )


class MemoryAwareLLM:
    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> FakeResponse:
        assert tools is not None

        user_messages = [
            message["content"] for message in messages if message["role"] == "user"
        ]
        assistant_messages = [
            message["content"] for message in messages if message["role"] == "assistant"
        ]

        if user_messages[-1] == "Mă numesc Andrei.":
            return FakeResponse(content="Salut, Andrei.")

        if user_messages[-1] == "Cum mă numesc?":
            if "Mă numesc Andrei." in user_messages and "Salut, Andrei." in assistant_messages:
                return FakeResponse(content="Te numești Andrei.")

            return FakeResponse(content="Nu am această informație.")

        raise AssertionError(f"Unexpected memory test messages: {messages}")


def main():
    datetime_result = ToolWrapper.call("get_current_datetime", {})
    assert datetime_result.startswith("Data curentă: ")
    assert "Ora curentă: " in datetime_result
    assert "Fus orar: " in datetime_result
    assert any(month in datetime_result for month in MONTHS_RO)
    assert "T" not in datetime_result
    assert "+03:00" not in datetime_result

    agent_answer = QAAgent(DateQuestionLLM()).answer("Salut! In ce data suntem astazi?")
    assert agent_answer.startswith("Astăzi este ")
    assert any(month in agent_answer for month in MONTHS_RO)
    assert "T" not in agent_answer
    assert "function" not in agent_answer
    assert "tool" not in agent_answer.lower()
    assert "get_current_datetime" not in agent_answer
    assert "folosit" not in agent_answer.lower()

    invented_tool_answer = QAAgent(InventedFormatToolLLM()).answer(
        "Salut! In ce data suntem astazi?"
    )
    assert invented_tool_answer == "Eroare: tool 'format_date' nu există."

    memory = InMemoryConversationMemory(window=4)
    memory_agent = QAAgent(MemoryAwareLLM(), memory=memory)
    assert memory_agent.answer("Mă numesc Andrei.", session_id="andrei") == "Salut, Andrei."
    assert memory_agent.answer("Cum mă numesc?", session_id="andrei") == "Te numești Andrei."
    assert (
        memory_agent.answer("Cum mă numesc?", session_id="alta-sesiune")
        == "Nu am această informație."
    )

    valid_message = LiteLLMClient._extract_message(
        {"choices": [{"message": {"content": "Răspuns test."}}]}
    )
    assert valid_message["content"] == "Răspuns test."

    try:
        LiteLLMClient._extract_message({"choices": []})
    except RuntimeError as error:
        assert "did not contain any choices" in str(error)
    else:
        raise AssertionError("Expected empty choices to raise RuntimeError.")

    try:
        LiteLLMClient._extract_message({"error": {"message": "provider unavailable"}})
    except RuntimeError as error:
        assert "provider unavailable" in str(error)
    else:
        raise AssertionError("Expected error response to raise RuntimeError.")

    cache_client = LiteLLMClient(prompt_caching=True)
    prepared = cache_client._prepare_messages(
        [
            {
                "role": "system",
                "content": "Prompt static.",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "Întrebare dinamică."},
        ]
    )
    assert prepared[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert prepared[0]["content"][0]["text"] == "Prompt static."
    assert "cache_control" not in prepared[0]
    assert prepared[1]["content"] == "Întrebare dinamică."

    uncached_prepared = LiteLLMClient(prompt_caching=False)._prepare_messages(
        [
            {
                "role": "system",
                "content": "Prompt static.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
    assert uncached_prepared == [{"role": "system", "content": "Prompt static."}]

    metrics = PromptCacheMetrics()
    metrics.record(
        usage={
            "cache_creation_input_tokens": 2500,
            "cache_read_input_tokens": 0,
            "input_tokens": 120,
        },
        latency_seconds=2.0,
    )
    metrics.record(
        usage={
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2500,
            "input_tokens": 80,
        },
        latency_seconds=0.8,
    )
    summary = metrics.summary()
    assert summary["estimated_saved_input_tokens"] == 2250
    assert summary["latency_reduction_percent"] == 60.0

    with TemporaryDirectory() as tmpdir:
        model_path = f"{tmpdir}/intent_classifier.pkl"
        report = train_intent_classifier(load_examples(), model_path=model_path)
        assert report["accuracy"] >= 0.80
        intent_model = load_model(model_path)
        assert predict_intent(intent_model, "Extrage suma din factura").label == "extract"
        assert predict_intent(intent_model, "Fa un rezumat scurt").label == "summarize"

    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
