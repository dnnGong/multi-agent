# Multi-Agent ML Chatbot and Evaluation

This folder contains two main scripts:

- `agents.py`: the multi-agent chatbot runtime.
- `eval.py`: the automated evaluation pipeline (test set generation + LLM-as-a-judge scoring).

## Project Structure

```text
src/
├── agents.py          # Multi-agent chatbot orchestration and CLI loop
├── eval.py            # Dataset generation + evaluation pipeline
├── test_set.json      # Input test dataset (generated or user-provided)
└── eval_results.json  # Evaluation output (summary + per-case details)
```

## How `agents.py` Works

`Head_Agent` orchestrates several sub-agents in a fixed route:

1. `Context_Rewriter_Agent`
2. `Obnoxious_Agent`
3. `Query_Agent(plan)`
4. `Query_Agent(search)` (if search is needed)
5. `Relevant_Documents_Agent`
6. `Answering_Agent`

### Routing Logic (per user turn)

1. Rewrite the latest user query into a standalone query using recent conversation history.
2. Detect obnoxious/rude input:
   - If detected, return refusal (`Refused: Obnoxious query detected.`).
3. Plan whether to search Pinecone (`SEARCH` or `NO_SEARCH`).
4. If searching:
   - Embed query and retrieve top-k documents from Pinecone.
   - Judge document relevance.
   - If not relevant, return refusal (`Refused: Retrieved documents are not relevant.`).
5. Generate final answer grounded in retrieved documents.

The script also prints the runtime path, e.g.:

```text
Context_Rewriter_Agent -> Obnoxious_Agent -> Query_Agent(plan) -> Query_Agent(search) -> Relevant_Documents_Agent -> Answering_Agent
```

## How `eval.py` Works

`eval.py` evaluates the chatbot behavior with six categories:

- `obnoxious` (10 cases)
- `irrelevant` (10 cases)
- `relevant` (10 cases)
- `small_talk` (5 cases)
- `hybrid` (8 cases)
- `multi_turn` (7 conversations)

Main components:

- `TestDatasetGenerator`: builds synthetic prompts (with fallback fixed prompts).
- `LLM_Judge`: behavior-only binary scoring (`score: 0/1`).
- `EvaluationPipeline`: runs all tests, stores per-case results, computes summary metrics.

## Prerequisites

Install dependencies in your environment:

```bash
pip install openai pinecone
```

Set required environment variables:

```bash
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
export PINECONE_INDEX_NAME="..."
```

Optional:

```bash
export PINECONE_NAMESPACE="ns-2500"
```

## Usage

Run chatbot (interactive CLI):

```bash
cd src
python agents.py
```

Run evaluation with an existing or auto-generated test set:

```bash
cd src
python eval.py --test_set test_set.json --out eval_results.json --judge_model gpt-4.1-nano
```

Generate test set only:

```bash
cd src
python eval.py --create_test_set --test_set test_set.json
```

## Output Format

`eval_results.json` contains:

- `summary`:
  - `total`, `passed`, `accuracy`
  - `by_category` with per-category totals and accuracies
- `results`:
  - full per-case records (`user_input`/`conversation`, `bot_response`, `agent_path`, `score`)

## Notes

- Default chat/completion model in both scripts is `gpt-4.1-nano`.
- Embedding model in `agents.py` is `text-embedding-3-small`.
- Pinecone namespace fallback logic is implemented in `Query_Agent` if the preferred namespace has no matches.
