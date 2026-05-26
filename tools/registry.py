import inspect
from typing import Callable, Any

from pydantic import BaseModel


TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(func: Callable) -> Callable:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) != 1:
        raise TypeError(
            f"{func.__name__}: tool-ul trebuie să aibă exact un singur parametru."
        )

    annotation = params[0].annotation

    if annotation is inspect.Parameter.empty:
        raise TypeError(
            f"{func.__name__}: parametrul trebuie adnotat cu un model Pydantic BaseModel."
        )

    if not issubclass(annotation, BaseModel):
        raise TypeError(
            f"{func.__name__}: parametrul unic trebuie să fie de tip BaseModel."
        )

    docstring = (func.__doc__ or "").strip()

    if not docstring:
        raise ValueError(
            f"{func.__name__}: docstring obligatoriu — devine description vizibil pentru LLM."
        )

    if len(docstring) < 15:
        raise ValueError(
            f"{func.__name__}: docstring prea scurt. LLM-ul are nevoie de o descriere clară."
        )

    TOOL_REGISTRY[func.__name__] = {
        "func": func,
        "params_model": annotation,
        "description": docstring,
    }

    return func