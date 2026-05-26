from prompts.registry import PromptRegistry
from tools import ToolWrapper
import tools.basic_tools  # important: înregistrează tool-urile
import json


class QAAgent:
    def __init__(self, llm, max_iterations: int = 5):
        self.llm = llm
        self.max_iterations = max_iterations
        self.prompt_registry = PromptRegistry(folder="prompts")

    def answer(self, question: str) -> str:
        planner_prompt = self.prompt_registry.render(
            "planner",
            tools_catalog=json.dumps(
                ToolWrapper.catalog(),
                ensure_ascii=False,
                indent=2,
            ),
        )

        messages = [
            {
                "role": "system",
                "content": planner_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.react_loop(messages, question)

    def react_loop(self, messages: list, question: str) -> str:
        tool_results_by_signature = {}
        last_tool_result = None
        observations = []

        for iteration in range(1, self.max_iterations + 1):
            response = self.llm.invoke(
                messages,
                tools=ToolWrapper.catalog(),
            )

            messages.append(self._assistant_message(response))

            tool_calls = getattr(response, "tool_calls", None)

            if not tool_calls:
                if observations:
                    return self._final_answer(question, observations, response.content)

                return response.content

            for tool_call in tool_calls:
                name, args = self._parse_tool_call(tool_call)
                tool_signature = (name, json.dumps(args, sort_keys=True))

                if tool_signature in tool_results_by_signature:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_results_by_signature[tool_signature],
                        }
                    )
                    continue

                result = ToolWrapper.call(name, args)

                if result.startswith("Eroare"):
                    return result

                if result == last_tool_result:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": str(result),
                        }
                    )
                    continue

                last_tool_result = result
                tool_results_by_signature[tool_signature] = result
                observations.append(
                    {
                        "tool": name,
                        "args": args,
                        "result": str(result),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result),
                    }
                )

        if observations:
            return self._final_answer(question, observations)

        return "Nu am putut produce un răspuns final. Te rog reformulează întrebarea."

    def _final_answer(
        self,
        question: str,
        observations: list[dict],
        planner_answer: str = "",
    ) -> str:
        observations_text = self._format_observations(observations)
        facts = self._run_prompt(
            "extract",
            question=question,
            observations=observations_text,
        )
        analysis = self._run_prompt(
            "analyst",
            question=question,
            facts=facts,
        )
        summary = self._run_prompt(
            "summary",
            question=question,
            facts=facts,
            analysis=analysis or planner_answer,
        )

        return summary or planner_answer or self._final_answer_from_tool_result(
            observations[-1]["result"]
        )

    def _run_prompt(self, name: str, **variables) -> str:
        prompt = self.prompt_registry.render(name, **variables)
        response = self.llm.invoke(
            [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": "Execută instrucțiunile.",
                },
            ],
            tools=None,
        )

        return response.content

    @staticmethod
    def _format_observations(observations: list[dict]) -> str:
        formatted = []

        for index, observation in enumerate(observations, start=1):
            formatted.append(
                "\n".join(
                    [
                        f"Observația {index}:",
                        f"Nume: {observation['tool']}",
                        f"Argumente: {json.dumps(observation['args'], ensure_ascii=False)}",
                        "Rezultat:",
                        observation["result"],
                    ]
                )
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _final_answer_from_tool_result(result: str) -> str:
        lines = []

        for line in result.splitlines():
            line = line.strip()

            if line.startswith("- ") and ": " in line:
                line = line.split(": ", 1)[1]

            if line[:1].isdigit() and ". " in line and ": " in line:
                line = line.split(": ", 1)[1]

            if line:
                lines.append(line)

        return "\n".join(lines) if lines else result

    @staticmethod
    def _assistant_message(response) -> dict:
        message = {
            "role": "assistant",
            "content": response.content,
        }

        if getattr(response, "tool_calls", None):
            message["tool_calls"] = response.tool_calls

        return message

    @staticmethod
    def _parse_tool_call(tool_call: dict) -> tuple[str, dict]:
        if "function" in tool_call:
            function = tool_call["function"]
            arguments = function.get("arguments") or "{}"

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            return function["name"], arguments

        return tool_call["name"], tool_call["args"]
