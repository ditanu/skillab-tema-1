import argparse
from dataclasses import dataclass
import os

from agent import QAAgent
from memory import build_conversation_memory


@dataclass
class FakeResponse:
    content: str
    tool_calls: list[dict] | None = None


class MemoryProbeLLM:
    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> FakeResponse:
        user_messages = [
            message["content"]
            for message in messages
            if message.get("role") == "user"
        ]

        if user_messages[-1] == "Ma numesc Andrei.":
            return FakeResponse("Salut, Andrei. Am salvat contextul conversatiei.")

        if user_messages[-1] == "Cum ma numesc?":
            if "Ma numesc Andrei." in user_messages[:-1]:
                return FakeResponse("Te numesti Andrei.")

            return FakeResponse("Nu am aceasta informatie in memorie.")

        return FakeResponse("Mesaj primit.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testeaza Conversation Memory fara server LiteLLM."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="URL PostgreSQL. Daca lipseste, se foloseste memorie in RAM.",
    )
    parser.add_argument("--session-id", default="memory-test-andrei")
    parser.add_argument("--memory-window", type=int, default=10)
    args = parser.parse_args()

    memory = build_conversation_memory(
        database_url=args.database_url,
        window=args.memory_window,
    )
    agent = QAAgent(MemoryProbeLLM(), memory=memory)

    first = agent.answer("Ma numesc Andrei.", session_id=args.session_id)
    second = agent.answer("Cum ma numesc?", session_id=args.session_id)
    isolated = agent.answer("Cum ma numesc?", session_id=f"{args.session_id}-izolat")

    print(f"Sesiune: {args.session_id}")
    print(f"1. {first}")
    print(f"2. {second}")
    print(f"3. sesiune izolata: {isolated}")

    assert second == "Te numesti Andrei."
    assert isolated == "Nu am aceasta informatie in memorie."
    print("Conversation memory test passed.")


if __name__ == "__main__":
    main()
