# Skillab Tema 1 - QA Agent

Acest proiect implementeaza un agent local de tip Q&A care raspunde la intrebari folosind un model compatibil OpenAI/LiteLLM si un set de tools locale.

Agentul foloseste un flux de tip ReAct: primeste intrebarea, decide daca are nevoie de tools, executa tool-urile relevante, apoi sintetizeaza raspunsul final pe baza observatiilor.

## Rulare

Comanda generala:

```bash
python run_agent.py "Intrebarea ta aici" --model gemini
```

Exemplu:

```bash
python run_agent.py "Pe cine pot contacta pentru a pune intrebari urgente? De asemenea, eu am cumparat astazi un curs, pana pe ce data pot face refund?" --model gemini
```

Pentru acest exemplu, agentul ar trebui sa foloseasca:

- `search_knowledge_base`, pentru informatii despre contact urgent si politica de refund.
- `get_current_datetime`, pentru a afla data curenta si a calcula limita celor 14 zile de refund.
- `calculator`, daca modelul decide ca are nevoie de un calcul numeric precis.

## Optiuni CLI

`run_agent.py` accepta urmatoarele optiuni:

| Optiune | Default | Descriere |
| --- | --- | --- |
| `question` | `Care este programul de suport?` | Intrebarea trimisa agentului. |
| `--model` | valoarea `LITELLM_MODEL` sau `ollama/mistral:7b` | Modelul trimis catre serverul OpenAI-compatible. |
| `--base-url` | valoarea `LITELLM_BASE_URL` sau `http://localhost:4000/v1` | URL-ul serverului LiteLLM/OpenAI-compatible. |
| `--timeout` | valoarea `LITELLM_TIMEOUT` sau `120` | Timeout HTTP, in secunde, pentru fiecare apel catre model. |
| `--max-retries` | valoarea `LITELLM_MAX_RETRIES` sau `2` | Numarul de retry-uri pentru raspunsuri HTTP 429. |

Exemplu cu URL explicit:

```bash
python run_agent.py "Care este programul de suport?" --model gemini --base-url http://localhost:4000/v1
```

## Variabile de mediu

Configurarea poate fi facuta si prin variabile de mediu:

```bash
export LITELLM_MODEL=gemini
export LITELLM_BASE_URL=http://localhost:4000/v1
export LITELLM_TIMEOUT=120
export LITELLM_MAX_RETRIES=2
```

## Tools disponibile

Tools sunt definite in `tools/basic_tools.py`, iar inregistrarea lor se face prin decoratorul `@register_tool`.

### `calculator`

Evalueaza expresii matematice simple atunci cand este nevoie de calcule precise.

Parametri:

- `expression`: expresia matematica de evaluat, de exemplu `2 + 3 * 4`.

Operatori permisi:

- adunare: `+`
- scadere: `-`
- inmultire: `*`
- impartire: `/`
- putere: `**`
- minus unar: `-x`

### `get_current_datetime`

Returneaza data si ora curenta pentru intrebari care depind de timp real.

Parametri:

- `timezone`: fusul orar folosit pentru data curenta. Default: `Europe/Bucharest`.

### `search_knowledge_base`

Cauta in baza locala de cunostinte informatii despre suport, contact, email, urgente, refund, facturare sau program.

Parametri:

- `query`: intrebarea sau termenii cautati.
- `max_results`: numarul maxim de documente relevante returnate. Default: `3`, minim `1`, maxim `10`.

Baza locala contine informatii despre:

- programul echipei de suport;
- politica de refund;
- contactul pentru intrebari urgente;
- facturare.

## Cum se adauga un tool nou

1. Defineste un model Pydantic pentru parametri in `tools/params_models.py`.
2. Creeaza functia in `tools/basic_tools.py`.
3. Adauga decoratorul `@register_tool` deasupra functiei.
4. Scrie un docstring clar, deoarece acesta devine descrierea vizibila pentru LLM.
5. Functia trebuie sa primeasca exact un singur parametru, iar parametrul trebuie sa fie un model Pydantic.

Exemplu minimal:

```python
from pydantic import BaseModel, Field
from tools.registry import register_tool


class ExampleParams(BaseModel):
    text: str = Field(description="Textul de procesat.")


@register_tool
def example_tool(params: ExampleParams) -> str:
    """Proceseaza un text simplu si returneaza rezultatul."""
    return params.text.upper()
```

## Arhitectura pe scurt

- `run_agent.py` parseaza argumentele CLI, configureaza clientul LiteLLM si porneste agentul.
- `agent.py` contine logica agentului, bucla ReAct si pasii de sintetizare a raspunsului.
- `prompts/` contine prompturile pentru planning, extractie, analiza si sumarizare.
- `tools/registry.py` gestioneaza inregistrarea tool-urilor.
- `tools/tool_wrapper.py` expune catalogul de tools catre model si executa tool-urile cerute.
- `tools/basic_tools.py` contine implementarile tool-urilor locale.

## Test rapid

Pentru verificare rapida:

```bash
python smoke_test.py
```
