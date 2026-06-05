# Multi-Agent ML Chatbot and Evaluation

This folder contains the multi-agent ML chatbot, a local RAG pipeline, a Streamlit UI, and an evaluation script.

- `agents.py`: the multi-agent chatbot runtime.
- `rag.py`: local RAG indexing and similarity search for `data/machine-learning.pdf`.
- `app_streamlit.py`: Streamlit chat frontend.
- `eval.py`: the automated evaluation pipeline (test set generation + LLM-as-a-judge scoring).

## Project Structure

```text
src/
├── agents.py          # Multi-agent chatbot orchestration and CLI loop
├── rag.py             # PDF chunking, embedding, JSON vector store, retrieval
├── app_streamlit.py   # Streamlit frontend
├── eval.py            # Dataset generation + evaluation pipeline
├── test_set.json      # Input test dataset (generated or user-provided)
└── eval_results.json  # Evaluation output (summary + per-case details)
data/
├── machine-learning.pdf
└── machine_learning_vector_store.json  # generated locally, not required before indexing
```

## How `agents.py` Works

`Head_Agent` orchestrates several sub-agents in a fixed route:

1. `Context_Rewriter_Agent`
2. `Obnoxious_Agent`
3. `Query_Agent(plan)`
4. `Query_Agent(search)` with local JSON RAG by default, or Pinecone when selected
5. `Relevant_Documents_Agent`
6. `Answering_Agent`

### Routing Logic (per user turn)

1. Rewrite the latest user query into a standalone query using recent conversation history.
2. Detect obnoxious/rude input:
   - If detected, return refusal (`Refused: Obnoxious query detected.`).
3. Plan whether to search the vector store (`SEARCH` or `NO_SEARCH`).
4. If searching:
   - Embed query and retrieve top-k chunks from the local JSON vector store or Pinecone.
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
cd /Users/gongjin/Downloads/LLM_course/multi-agent
pip install -r requirements.txt
```

Set the OpenAI key:

```bash
export OPENAI_API_KEY="..."
```

## Local RAG Usage

Build the local vector store from the copied ML textbook:

```bash
cd /Users/gongjin/Downloads/LLM_course/multi-agent
python src/rag.py --pdf data/machine-learning.pdf --out data/machine_learning_vector_store.json
```

Run chatbot in the CLI:

```bash
cd /Users/gongjin/Downloads/LLM_course/multi-agent
RAG_BACKEND=local python src/agents.py
```

Run the Streamlit frontend:

```bash
cd /Users/gongjin/Downloads/LLM_course/multi-agent
streamlit run src/app_streamlit.py
```

The frontend can also build/rebuild the local vector store from the sidebar.

## Pinecone Mode

The old Pinecone-backed RAG path is still supported:

```bash
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
export PINECONE_INDEX_NAME="machine-learning-textbook"
export PINECONE_NAMESPACE="ns-2500"
RAG_BACKEND=pinecone python src/agents.py
```

Run evaluation with an existing or auto-generated test set:

```bash
cd /Users/gongjin/Downloads/LLM_course/multi-agent/src
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
