from pydantic import BaseModel, Field


class CalculatorParams(BaseModel):
    expression: str = Field(
        description="Expresia matematică de evaluat, de exemplu: '2 + 3 * 4'.",
        min_length=1,
    )


class DateTimeParams(BaseModel):
    timezone: str = Field(
        default="Europe/Bucharest",
        description="Fusul orar pentru data și ora curentă.",
    )


class KnowledgeSearchParams(BaseModel):
    query: str = Field(
        description=(
            "Întrebarea sau termenii de căutat în baza locală de cunoștințe. "
            "Folosește pentru informații despre suport, contact, email, urgențe, "
            "refund, facturare sau program."
        ),
        min_length=2,
    )
    max_results: int = Field(
        default=3,
        description="Numărul maxim de documente relevante returnate.",
        ge=1,
        le=10,
    )