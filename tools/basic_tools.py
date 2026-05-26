import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from tools.registry import register_tool
from tools.params_models import (
    CalculatorParams,
    DateTimeParams,
    KnowledgeSearchParams,
)


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_MONTHS_RO = {
    1: "ianuarie",
    2: "februarie",
    3: "martie",
    4: "aprilie",
    5: "mai",
    6: "iunie",
    7: "iulie",
    8: "august",
    9: "septembrie",
    10: "octombrie",
    11: "noiembrie",
    12: "decembrie",
}


def _safe_eval_math(expression: str) -> float | int:
    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Operator nepermis: {op_type.__name__}")

            left = eval_node(node.left)
            right = eval_node(node.right)

            return _ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Operator unar nepermis: {op_type.__name__}")

            operand = eval_node(node.operand)

            return _ALLOWED_OPERATORS[op_type](operand)

        raise ValueError("Expresie matematică nepermisă.")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed.body)


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluează expresii matematice simple atunci când este nevoie de calcule precise."""
    result = _safe_eval_math(params.expression)
    return str(result)


@register_tool
def get_current_datetime(params: DateTimeParams) -> str:
    """Returnează data și ora curentă pentru întrebări care depind de timp real."""
    now = datetime.now(ZoneInfo(params.timezone))
    month = _MONTHS_RO[now.month]
    return (
        f"Data curentă: {now.day} {month} {now.year}\n"
        f"Ora curentă: {now:%H:%M}\n"
        f"Fus orar: {params.timezone}"
    )


@register_tool
def search_knowledge_base(params: KnowledgeSearchParams) -> str:
    """Caută în baza locală informații despre suport, contact, email, urgențe, refund, facturare sau program."""
    documents = [
        {
            "title": "Program suport",
            "content": "Echipa de suport răspunde de luni până vineri între 09:00 și 18:00.",
        },
        {
            "title": "Politică refund",
            "content": "Refund-ul poate fi solicitat în maximum 14 zile de la achiziție.",
        },
        {
            "title": "Contact",
            "content": "Pentru întrebări urgente, utilizatorii pot trimite email la support@example.com.",
        },
        {
            "title": "Facturare",
            "content": "Factura este generată automat după confirmarea plății.",
        },
    ]

    query = params.query.lower()
    query_terms = set(query.replace("?", "").replace(".", "").split())

    scored_documents = []

    for document in documents:
        title = document["title"].lower()
        content = document["content"].lower()
        searchable_text = f"{title} {content}"

        score = 0

        for term in query_terms:
            if term in searchable_text:
                score += 1

        if score > 0:
            scored_documents.append((score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)

    results = [document for _, document in scored_documents[: params.max_results]]

    if not results:
        return (
            "Nu am găsit informații relevante în baza locală. "
            "Pot căuta doar despre suport, contact, email, urgențe, refund, facturare sau program."
        )

    formatted_results = []

    for index, document in enumerate(results, start=1):
        formatted_results.append(
            f"{index}. {document['title']}: {document['content']}"
        )

    return "\n".join(formatted_results)
