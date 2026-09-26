# LLM Router Toolkit

A lightweight Python toolkit for routing LLM prompts to the most appropriate model based on **semantic intent matching** and **speculative cascade execution**. Built to optimize for cost, latency, and response quality by dynamically choosing between models like GPT-4o-mini, GPT-4o, and Claude 3.5 Sonnet.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example Output](#example-output)
- [Tuning the Semantic Router](#tuning-the-semantic-router)
- [Requirements](#requirements)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

An LLM router acts as an intelligent proxy between incoming prompts and a pool of foundation models. Instead of sending every request to one expensive model, it decides — per request — which model is best suited for the job.

This repo implements two routing strategies:

| Strategy | File | Mechanism |
| :--- | :--- | :--- |
| **Semantic Routing** | `router/semantic_router.py` | Embeds the prompt and compares it via cosine similarity against pre-defined intent clusters |
| **Cascade Routing** | `router/cascade_router.py` | Sends every prompt to a cheap model first, validates the response, and escalates to a stronger model only on failure |

Both strategies share a common interface defined in `router/base.py`.

---

## How It Works

### Semantic Router
1. Each intent cluster (e.g. `simple_retrieval`, `coding_architecture`, `creative_writing`) is defined with a handful of representative example prompts.
2. On startup, the router encodes every example sentence into an embedding using `sentence-transformers` (`all-MiniLM-L6-v2` by default).
3. When a new prompt comes in, it's embedded and compared against every stored example via cosine similarity.
4. The prompt is routed to whichever cluster's best-matching example scored highest — as long as that score clears a confidence `threshold`. If nothing clears the threshold, it falls back to a default high-capability model.

### Cascade Router
1. Every prompt is first dispatched to a cheap, fast model (e.g. `gpt-4o-mini`).
2. The response is passed through an `evaluator_fn` — a function you define (e.g. checking for uncertainty phrases, validating JSON structure, running a regex/unit test).
3. If the response **passes**, no further action is needed — you saved the cost of a bigger model.
4. If the response **fails**, the router escalates and returns a route to a stronger model (e.g. `gpt-4o`).

---

## Project Structure

```
llm-router-toolkit/
├── README.md
├── requirements.txt
├── .env                    # Your API keys (not committed)
├── .gitignore
├── main.py                 # Demo/entry point
└── router/
    ├── __init__.py
    ├── base.py              # Abstract base class + RouteDecision schema
    ├── semantic_router.py   # Embedding-based intent routing
    └── cascade_router.py    # Speculative cheap-model-first routing
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/anuj667/llm-router.git
cd llm-router
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> The first run will also download the `all-MiniLM-L6-v2` embedding model (~90MB) via Hugging Face. This happens once and is cached locally.

### 4. Set up environment variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

> Note: the current routing logic does not yet make live API calls — it only *decides* which model would be used. Keys are set up here for when live model calls are wired in (see [Roadmap](#roadmap)).

---

## Configuration

Routing clusters for the semantic router are defined directly in `main.py` inside the `SEMANTIC_ROUTES` dictionary:

```python
SEMANTIC_ROUTES = {
    "simple_retrieval": {
        "samples": [...],
        "target_model": "gpt-4o-mini",
        "provider": "openai",
        "cost_per_1k": 0.00015
    },
    ...
}
```

To add a new intent category, add a new key with:
- `samples`: a list of representative example prompts (more = better accuracy)
- `target_model`: the model to route to
- `provider`: the model's provider
- `cost_per_1k`: estimated cost per 1,000 tokens, for cost tracking

---

## Usage

Run the demo:
```bash
python main.py
```

This will:
1. Run several test prompts through the `SemanticRouter` and print which model each was routed to, along with confidence scores
2. Simulate a `CascadeRouter` flow: dispatch to a cheap model, then escalate if the (mocked) response fails validation

### Using the routers in your own code

```python
from router.semantic_router import SemanticRouter
from router.cascade_router import CascadeRouter

# Semantic routing
router = SemanticRouter(routes=SEMANTIC_ROUTES)
decision = router.route("Explain how transformers work in deep learning")
print(decision.model_name, decision.provider, decision.confidence)

# Cascade routing
def my_validator(response: str) -> bool:
    return len(response.strip()) > 20 and "I am not sure" not in response

cascade = CascadeRouter(evaluator_fn=my_validator)
initial = cascade.route("Summarize this contract")
# ... send prompt to initial.model_name, get a response ...
escalation = cascade.evaluate_or_escalate(response_text)
if escalation:
    # ... send prompt to escalation.model_name instead ...
```

---

## Example Output

```
--- 1. Semantic Router Test ---
Query: 'Where is the headquarters of Interpol?'
-> Route: gpt-4o-mini (openai) | Reason: Matched cluster: 'simple_retrieval' | Conf: 0.42

Query: 'Write a python script to parse binary heap memory dumps.'
-> Route: claude-3-5-sonnet (anthropic) | Reason: Matched cluster: 'coding_architecture' | Conf: 0.51

Query: 'Compose an existential poem about space.'
-> Route: claude-3-5-sonnet (anthropic) | Reason: Matched cluster: 'creative_writing' | Conf: 0.44

--- 2. Cascade Router Test ---
Initial Dispatch: gpt-4o-mini
Escalated to: gpt-4o (Reason: Cascade escalation: Low-tier output failed validation criteria.)
```

---

## Tuning the Semantic Router

Embedding similarity scores for short sentences are often lower than intuition suggests — a `threshold` of `0.3–0.4` is typically more realistic than `0.5+` for `all-MiniLM-L6-v2`. If routing accuracy feels off:

1. **Add more sample sentences** per cluster (5–10 is a reasonable minimum) — sparse clusters produce noisy centroids
2. **Lower or raise the `threshold`** passed to `.route()` depending on how conservative you want the fallback behavior to be
3. **Try a different embedding model** — e.g. `all-mpnet-base-v2` trades speed for higher accuracy

---

## Requirements

See `requirements.txt`. Core dependencies:

- `pydantic` — schema validation for routing decisions
- `sentence-transformers` — embedding generation for semantic routing
- `numpy`, `scikit-learn` — similarity computation
- `openai`, `anthropic` — SDKs for eventual live model calls
- `python-dotenv` — loads `.env` variables

---

## Roadmap

- [ ] Wire up live API calls to OpenAI/Anthropic instead of returning decisions only
- [ ] Add a `rule_router.py` for regex/keyword-based routing
- [ ] Add a `cost_optimizer.py` to track cumulative spend across routing decisions
- [ ] Externalize `SEMANTIC_ROUTES` into a `config.yaml` file instead of hardcoding in `main.py`
- [ ] Add unit tests for routing accuracy across a labeled prompt set
- [ ] Add embedding caching for repeated/identical queries

---
