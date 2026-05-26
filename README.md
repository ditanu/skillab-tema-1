# QA Agent

Agentul folosește un server compatibil OpenAI/LiteLLM la `http://localhost:4000/v1`.
Modelul implicit este `ollama/mistral:7b`.

## Rulare cu Mistral prin infrastructura SkillLab

Pornește infrastructura comună SkillLab:

```bash
docker-compose up -d
```

Apoi rulează agentul:

```bash
python run_agent.py "Care este programul de suport?"
```

Poți suprascrie modelul sau URL-ul:

```bash
python run_agent.py "Care este programul de suport?" \
  --model ollama/mistral:7b \
  --base-url http://localhost:4000/v1
```

Sau prin variabile de mediu:

```bash
export LITELLM_MODEL="ollama/mistral:7b"
export LITELLM_BASE_URL="http://localhost:4000/v1"
python run_agent.py "Care este programul de suport?"
```

Dacă serverul răspunde că modelul nu există, verifică dacă `ollama-init` a descărcat `mistral:7b` sau rulează:

```bash
docker exec -it skillab-ollama ollama pull mistral:7b
```
