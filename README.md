# Skillab Tema 1 - QA Agent

Agent local de Q&A construit peste un model compatibil OpenAI/LiteLLM, tools locale si extensii din lectiile L7/L8:

- ReAct agent cu tool calling;
- Conversation Memory persistenta in PostgreSQL;
- integrare optionala LangGraph;
- Anthropic Prompt Caching cu `cache_control: ephemeral`;
- Intent Classifier local cu scikit-learn.

## Setup rapid

Instaleaza dependintele:

```bash
pip install -r requirements.txt
```

Porneste PostgreSQL pentru memoria persistenta:

```bash
docker compose up -d postgres
```

Daca Docker spune ca exista deja containerul `skillab-postgres`, refoloseste-l:

```bash
docker start skillab-postgres
```

Seteaza variabilele uzuale:

```bash
export DATABASE_URL=postgresql+psycopg://skillab:skillab_dev@localhost:5432/skillab
export LITELLM_BASE_URL=http://localhost:4000/v1
export LITELLM_MODEL=gemini
```

Pentru rularea reala a agentului ai nevoie si de LiteLLM pornit pe portul `4000`. Verifica:

```bash
curl http://localhost:4000/health
```

## Rulare agent

Comanda generala:

```bash
python run_agent.py "Intrebarea ta aici" --model gemini
```

Exemplu:

```bash
python run_agent.py "Pe cine pot contacta pentru intrebari urgente? Pana cand pot face refund daca am cumparat astazi un curs?" --model gemini
```

Agentul poate folosi:

- `search_knowledge_base` pentru suport, contact, refund, facturare sau program;
- `get_current_datetime` pentru intrebari dependente de data curenta;
- `calculator` pentru calcule precise.

## Optiuni CLI

| Optiune | Default | Descriere |
| --- | --- | --- |
| `question` | `Care este programul de suport?` | Intrebarea trimisa agentului. |
| `--model` | `LITELLM_MODEL` sau `ollama/mistral:7b` | Modelul trimis catre LiteLLM/OpenAI-compatible. |
| `--base-url` | `LITELLM_BASE_URL` sau `http://localhost:4000/v1` | URL-ul serverului LiteLLM/OpenAI-compatible. |
| `--timeout` | `LITELLM_TIMEOUT` sau `120` | Timeout HTTP in secunde. |
| `--max-retries` | `LITELLM_MAX_RETRIES` sau `2` | Retry-uri pentru HTTP 429. |
| `--session-id` | `AGENT_SESSION_ID` sau `default` | Conversatia folosita pentru memorie. |
| `--database-url` | `DATABASE_URL` sau gol | PostgreSQL pentru long-term memory. Fara el, memoria este in RAM. |
| `--memory-window` | `AGENT_MEMORY_WINDOW` sau `10` | Cate mesaje recente se injecteaza in context. |
| `--anthropic-prompt-cache` | `ANTHROPIC_PROMPT_CACHE` sau dezactivat | Activeaza `cache_control: ephemeral`. |
| `--fixed-context-file` | `FIXED_CONTEXT_FILE` sau gol | Fisier cu context static inclus in prefixul cache-uit. |
| `--show-cache-metrics` | `SHOW_CACHE_METRICS` sau dezactivat | Afiseaza metrici pentru prompt caching. |

## Conversation Memory (L8)

Memoria are doua niveluri:

- short-term: ultimele `AGENT_MEMORY_WINDOW` mesaje sunt injectate in context;
- long-term: mesajele `user` si `assistant` sunt salvate in PostgreSQL in `sessions` si `chat_messages`;
- izolarea se face prin `session_id`.

Test fara LiteLLM, doar pentru memorie:

```bash
export DATABASE_URL=postgresql+psycopg://skillab:skillab_dev@localhost:5432/skillab
python test_conversation_memory.py
```

Output asteptat:

```text
2. Te numesti Andrei.
Conversation memory test passed.
```

Test complet cu agentul real:

```bash
export DATABASE_URL=postgresql+psycopg://skillab:skillab_dev@localhost:5432/skillab
python run_agent.py "Ma numesc Andrei." --session-id andrei --model gemini
python run_agent.py "Cum ma numesc?" --session-id andrei --model gemini
```

Al doilea raspuns ar trebui sa confirme numele. Pentru izolarea sesiunilor:

```bash
python run_agent.py "Cum ma numesc?" --session-id alta-sesiune --model gemini
```

### LangGraph

Integrarea este in `langgraph_agent.py`. Nodul `agent` primeste ultimul mesaj user din state, apeleaza `QAAgent.answer(..., session_id=...)`, iar memoria persistenta este incarcata si salvata in acelasi flow.

```python
from agent import QAAgent
from langgraph_agent import build_agent_graph
from memory import build_conversation_memory
from run_agent import LiteLLMClient

memory = build_conversation_memory(
    database_url="postgresql+psycopg://skillab:skillab_dev@localhost:5432/skillab",
    window=10,
)
agent = QAAgent(LiteLLMClient(model="gemini"), memory=memory)
graph = build_agent_graph(agent)

graph.invoke({
    "session_id": "andrei",
    "messages": [{"role": "user", "content": "Ma numesc Andrei."}],
})
```

## Prompt Caching Anthropic (L8)

Pentru modele Anthropic/Claude, agentul poate marca prefixul static cu:

```json
{"cache_control": {"type": "ephemeral"}}
```

Prefixul cache-uit include:

- system prompt-ul planner-ului;
- catalogul fix de tool-uri;
- contextul fix din `--fixed-context-file`, daca exista.

Nu sunt puse in cache: intrebarea user-ului, istoricul conversatiei si rezultatele tool-urilor.

Test recomandat, in acelasi proces:

```bash
python test_prompt_cache.py \
  --model anthropic/claude-sonnet-4-20250514 \
  --base-url http://localhost:4000/v1
```

La primul apel cauta:

```text
cache_creation_input_tokens: > 0
cache_read_input_tokens: 0
```

La al doilea apel cauta:

```text
cache_read_input_tokens: > 0
estimated_saved_input_tokens: ...
latency_reduction: ...
```

Conditii importante:

- LiteLLM trebuie sa fie configurat cu `ANTHROPIC_API_KEY`;
- modelul trebuie sa fie Anthropic/Claude, nu Gemini/Ollama;
- prefixul static trebuie sa treaca pragul minim de tokeni Anthropic;
- al doilea apel trebuie facut rapid, deoarece cache-ul ephemeral expira.

Rulare prin agent:

```bash
python run_agent.py "Care sunt termenii principali?" \
  --model anthropic/claude-sonnet-4-20250514 \
  --anthropic-prompt-cache \
  --fixed-context-file ./context.txt \
  --show-cache-metrics
```

## Intent Classifier cu scikit-learn (L7)

Classifier local pentru intentii:

- date: `data/intent_queries.csv`;
- labels: `search`, `extract`, `summarize`;
- features: `TfidfVectorizer` cu unigram + bigram;
- model: `LogisticRegression`;
- output model: `models/intent_classifier.pkl`.

Antrenare:

```bash
python intent_classifier.py --dataset data/intent_queries.csv --model-path models/intent_classifier.pkl
```

Benchmark fara LLM:

```bash
python compare_intent_classifiers.py --retrain --skip-llm
```

Benchmark classifier vs LLM:

```bash
python compare_intent_classifiers.py --retrain \
  --model gemini \
  --base-url http://localhost:4000/v1
```

Raportul include:

- `accuracy` pe holdout stratificat;
- `avg_latency_ms` si `p95_latency_ms`;
- cost estimat pentru LLM din `usage`;
- speedup de latenta si economie de cost pentru classifier.

## Tools disponibile

Tools sunt definite in `tools/basic_tools.py` si inregistrate prin `@register_tool`.

### `calculator`

Evalueaza expresii matematice simple.

Parametri:

- `expression`: expresia matematica, de exemplu `2 + 3 * 4`.

Operatori permisi: `+`, `-`, `*`, `/`, `**`, minus unar.

### `get_current_datetime`

Returneaza data si ora curenta.

Parametri:

- `timezone`: fusul orar folosit. Default: `Europe/Bucharest`.

### `search_knowledge_base`

Cauta in baza locala de cunostinte informatii despre suport, contact, email, urgente, refund, facturare sau program.

Parametri:

- `query`: intrebarea sau termenii cautati;
- `max_results`: numarul maxim de documente relevante. Default: `3`, minim `1`, maxim `10`.

## Arhitectura

- `agent.py`: logica ReAct, tool loop, sinteza finala si memory injection;
- `run_agent.py`: CLI, client LiteLLM, prompt caching si metrics;
- `memory.py`: memorie in RAM si PostgreSQL;
- `langgraph_agent.py`: integrare LangGraph;
- `prompt_cache.py`: metrici pentru prompt caching;
- `intent_classifier.py`: training si predictie intent classifier;
- `compare_intent_classifiers.py`: benchmark LLM vs scikit-learn;
- `tools/`: tool registry si tool implementations;
- `prompts/`: planner, extract, analyst si summary prompts;
- `data/intent_queries.csv`: dataset pentru intent classifier.

## Teste

Test general:

```bash
python smoke_test.py
```

Test memorie fara LiteLLM:

```bash
python test_conversation_memory.py
```

Test prompt caching Anthropic:

```bash
python test_prompt_cache.py --model anthropic/claude-sonnet-4-20250514
```

Test intent classifier:

```bash
python compare_intent_classifiers.py --retrain --skip-llm
```

## Troubleshooting

`Connection refused` la `localhost:4000` inseamna ca LiteLLM nu ruleaza:

```bash
curl http://localhost:4000/health
```

Conflict Docker pe `skillab-postgres` inseamna ca exista deja containerul:

```bash
docker ps -a --filter name=skillab-postgres
docker start skillab-postgres
```

Daca prompt caching raporteaza mereu `0` tokeni cache-uiti, verifica:

- modelul este Anthropic/Claude;
- `ANTHROPIC_API_KEY` este setat in LiteLLM;
- prefixul static este suficient de mare;
- faci doua apeluri apropiate in timp.
