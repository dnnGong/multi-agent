#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from pinecone import Pinecone


# =========================
# Utilities
# =========================

def _safe_strip(s: str) -> str:
    return (s or "").strip()

def _to_bool_from_text(t: str) -> Optional[bool]:
    """
    Accept: yes/no, true/false, 1/0
    """
    if t is None:
        return None
    x = t.strip().lower()
    if x in ["yes", "true", "1"]:
        return True
    if x in ["no", "false", "0"]:
        return False
    return None

def _extract_json_block(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from model output.
    """
    if not text:
        return None
    # naive: find first {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class RetrievedDoc:
    id: str
    score: float
    text: str
    meta: Dict[str, Any]


# =========================
# 1) Obnoxious Agent
# =========================

class Obnoxious_Agent:
    """
    Binary classifier: obnoxious or not.
    Must NOT use LangChain (we don't).
    """

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.prompt = (
            "You are an Obnoxious Query Detector.\n"
            "Return ONLY one token: YES or NO.\n"
            "YES = the user is being rude, insulting, hateful, harassing, or obscene toward a person/group.\n"
            "NO = otherwise.\n"
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        # response should be YES/NO
        t = response.strip().upper()
        if "YES" in t:
            return True
        if "NO" in t:
            return False
        # fallback
        b = _to_bool_from_text(response)
        return bool(b) if b is not None else False

    def check_query(self, query: str) -> bool:
        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Query:\n{query}\n\nAnswer YES or NO only."},
        ]
        r = self.client.chat.completions.create(
            model="gpt-4.1-nano",  # per your screenshot requirement
            messages=msg,
            temperature=0.0,
        )
        out = r.choices[0].message.content or ""
        print(f"[DEBUG][Obnoxious] raw={out!r}")
        return self.extract_action(out)


# =========================
# 2) Context Rewriter Agent
# =========================

class Context_Rewriter_Agent:
    """
    Rewrites latest user query into a standalone question given conversation history.
    Used for multi-turn. (Helps evaluation.)
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.prompt = (
            "You are a context rewriter.\n"
            "Given conversation history and the latest user query, rewrite the latest query into a standalone, "
            "unambiguous question suitable for document retrieval.\n"
            "If the query is already standalone, return it unchanged.\n"
            "Return ONLY the rewritten query text.\n"
        )

    def rephrase(self, user_history: List[Dict[str, str]], latest_query: str) -> str:
        # keep short context
        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in user_history[-8:]]
        )
        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Conversation:\n{history_text}\n\nLatest query:\n{latest_query}\n\nRewrite:"},
        ]
        r = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=msg,
            temperature=0.0,
        )
        out = _safe_strip(r.choices[0].message.content or "")
        print(f"[DEBUG][Rewriter] in={latest_query!r} out={out!r}")
        return out or latest_query


# =========================
# 3) Query Agent
# =========================

class Query_Agent:
    """
    - Creates embeddings for query
    - Queries Pinecone
    - Returns docs
    """

    def __init__(self, pinecone_index, openai_client: OpenAI, embeddings_model: str, namespace: str = "") -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings_model = embeddings_model
        self.namespace = (namespace or "").strip()
        self._cached_namespaces: Optional[List[str]] = None
        self._fallback_namespace_guesses = [
            "",
            "ns-2500",
            "ns2500",
            "ns-1000",
            "ns1000",
            "ns-500",
            "ns500",
        ]
        self.prompt = (
            "You are a query planner.\n"
            "Given the user query, decide whether to search the vector store.\n"
            "Return JSON: {\"action\": \"SEARCH\" or \"NO_SEARCH\", \"query\": \"...\"}.\n"
            "The indexed corpus is a machine-learning textbook.\n"
            "Use SEARCH for ML factual questions; use NO_SEARCH only for greetings/chitchat.\n"
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def _embed(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
            model=self.embeddings_model,
            input=text,
        )
        vec = emb.data[0].embedding
        return vec

    def _query_one_namespace(self, vec: List[float], k: int, namespace: str) -> List[RetrievedDoc]:
        kwargs: Dict[str, Any] = {
            "vector": vec,
            "top_k": k,
            "include_metadata": True,
        }
        ns = (namespace or "").strip()
        if ns:
            kwargs["namespace"] = ns
        res = self.index.query(**kwargs)

        docs: List[RetrievedDoc] = []
        for m in res.matches or []:
            meta = m.metadata or {}
            text = meta.get("text") or meta.get("content") or ""
            docs.append(RetrievedDoc(id=m.id, score=float(m.score), text=str(text), meta=dict(meta)))
        print(
            f"[DEBUG][Pinecone] namespace={ns or '<default>'} "
            f"k={k} got={len(docs)} top_scores={[d.score for d in docs[:3]]}"
        )
        return docs

    def _get_index_namespaces(self) -> List[str]:
        if self._cached_namespaces is not None:
            return self._cached_namespaces
        try:
            stats = self.index.describe_index_stats() or {}
            ns_map = stats.get("namespaces", {}) if isinstance(stats, dict) else {}
            self._cached_namespaces = list(ns_map.keys()) if isinstance(ns_map, dict) else []
            print(f"[DEBUG][Pinecone] index namespaces={self._cached_namespaces}")
        except Exception as e:
            print(f"[DEBUG][Pinecone] describe_index_stats failed: {e}")
            self._cached_namespaces = []
        return self._cached_namespaces

    def _candidate_namespaces(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for ns in [self.namespace] + self._get_index_namespaces() + self._fallback_namespace_guesses:
            n = (ns or "").strip()
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def query_vector_store(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        vec = self._embed(query)
        for ns in self._candidate_namespaces():
            docs = self._query_one_namespace(vec, k, ns)
            if docs:
                if ns != self.namespace:
                    print(
                        f"[DEBUG][Pinecone] using fallback namespace: "
                        f"{ns or '<default>'}"
                    )
                return docs
        return []

    def extract_action(self, response, query=None):
        """
        Expect JSON with action + query
        """
        j = _extract_json_block(response)
        if not j:
            # fallback: assume search
            return {"action": "SEARCH", "query": query or ""}
        action = str(j.get("action", "SEARCH")).upper()
        q = str(j.get("query", query or "")).strip()
        return {"action": action, "query": q}

    def plan(self, query: str) -> Dict[str, str]:
        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"User query:\n{query}\n\nReturn JSON only."},
        ]
        r = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=msg,
            temperature=0.0,
        )
        out = r.choices[0].message.content or ""
        print(f"[DEBUG][QueryPlan] raw={out!r}")
        return self.extract_action(out, query=query)


# =========================
# 4) Answering Agent
# =========================

class Answering_Agent:
    """
    Generate final response grounded in retrieved docs.
    """

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a helpful assistant. Answer the user using ONLY the provided documents.\n"
            "If the documents do not contain the answer, say you cannot find it in the provided documents.\n"
            "Be concise and clear.\n"
        )

    def generate_response(self, query: str, docs: List[RetrievedDoc], conv_history: List[Dict[str, str]], k: int = 5) -> str:
        docs = docs[:k]
        ctx = "\n\n".join(
            [f"[Doc {i+1} | score={d.score:.4f}]\n{d.text}" for i, d in enumerate(docs)]
        )
        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in conv_history[-6:]]
        )

        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Conversation (recent):\n{history_text}\n\nUser query:\n{query}\n\nDocuments:\n{ctx}\n\nAnswer:"},
        ]
        r = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=msg,
            temperature=0.2,
        )
        out = _safe_strip(r.choices[0].message.content or "")
        print(f"[DEBUG][Answer] len_docs={len(docs)} out_len={len(out)}")
        return out


# =========================
# 5) Relevant Documents Agent
# =========================

class Relevant_Documents_Agent:
    """
    Judge whether retrieved docs are relevant for answering.
    Must NOT use LangChain (we don't).
    """

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a relevance judge.\n"
            "Given a user query and retrieved documents, decide whether the documents contain information "
            "useful to answer the query.\n"
            "Return ONLY YES or NO.\n"
        )

    def get_relevance(self, query: str, docs: List[RetrievedDoc]) -> str:
        ctx = "\n\n".join([f"[Doc {i+1}]\n{d.text[:1200]}" for i, d in enumerate(docs[:5])])
        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Query:\n{query}\n\nDocs:\n{ctx}\n\nAnswer YES or NO only."},
        ]
        r = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=msg,
            temperature=0.0,
        )
        out = (r.choices[0].message.content or "").strip().upper()
        print(f"[DEBUG][RelJudge] raw={out!r}")
        return "YES" if "YES" in out else "NO"


# =========================
# 6) Head Agent
# =========================

class Head_Agent:
    def __init__(self, openai_key: str, pinecone_key: str, pinecone_index_name: str, pinecone_namespace: str = "ns-2500") -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_namespace = (pinecone_namespace or "").strip()

        self.client = OpenAI(api_key=self.openai_key)

        pc = Pinecone(api_key=self.pinecone_key)
        self.index = pc.Index(self.pinecone_index_name)

        # runtime state
        self.conv_history: List[Dict[str, str]] = []
        self.agent_path: List[str] = []

        self.setup_sub_agents()

    def setup_sub_agents(self):
        # embeddings model: pick what you used when building pinecone index
        self.embeddings_model = "text-embedding-3-small"

        self.obnoxious = Obnoxious_Agent(self.client)
        self.rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(
            self.index,
            self.client,
            self.embeddings_model,
            namespace=self.pinecone_namespace,
        )
        self.rel_agent = Relevant_Documents_Agent(self.client)
        self.answer_agent = Answering_Agent(self.client)

    def _looks_like_ml_query(self, query: str) -> bool:
        q = (query or "").lower()
        ml_terms = [
            "machine learning", "ml", "logistic regression", "overfitting", "underfitting",
            "gradient descent", "learning rate", "regularization", "cross-validation",
            "confusion matrix", "precision", "recall", "f1", "support vector machine",
            "svm", "perceptron", "neural network", "bias-variance", "bias variance",
            "classification", "regression",
        ]
        return any(t in q for t in ml_terms)

    def _reset_path(self):
        self.agent_path = []

    def _log_step(self, name: str):
        self.agent_path.append(name)

    def _refuse(self, reason: str) -> str:
        return f"Refused: {reason}"

    def run_one_turn(self, user_query: str) -> Tuple[str, List[str]]:
        """
        Return: (assistant_text, agent_path)
        """
        self._reset_path()

        # 1) rewrite
        self._log_step("Context_Rewriter_Agent")
        rewritten = self.rewriter.rephrase(self.conv_history, user_query)

        # 2) obnoxious check
        self._log_step("Obnoxious_Agent")
        is_bad = self.obnoxious.check_query(rewritten)
        if is_bad:
            return self._refuse("Obnoxious query detected."), list(self.agent_path)

        # 3) query planning + search
        self._log_step("Query_Agent(plan)")
        plan = self.query_agent.plan(rewritten)
        action = plan.get("action", "SEARCH")
        q_for_search = plan.get("query", rewritten)

        if action == "NO_SEARCH" and self._looks_like_ml_query(rewritten):
            print("[DEBUG][QueryPlan] override NO_SEARCH -> SEARCH for ML query")
            action = "SEARCH"

        if action == "NO_SEARCH":
            # In this assignment, NO_SEARCH likely corresponds to greetings/small talk
            self._log_step("Answering_Agent")
            ans = self.answer_agent.generate_response(rewritten, [], self.conv_history, k=0)
            return ans, list(self.agent_path)

        self._log_step("Query_Agent(search)")
        docs = self.query_agent.query_vector_store(q_for_search, k=5)

        # 4) relevance judge
        self._log_step("Relevant_Documents_Agent")
        rel = self.rel_agent.get_relevance(rewritten, docs)
        if rel != "YES":
            return self._refuse("Retrieved documents are not relevant."), list(self.agent_path)

        # 5) answer
        self._log_step("Answering_Agent")
        ans = self.answer_agent.generate_response(rewritten, docs, self.conv_history, k=5)
        return ans, list(self.agent_path)

    def main_loop(self):
        """
        Simple CLI chatbot loop. Records history and prints agent path each turn.
        """
        print("Multi-agent chatbot. Type 'exit' to quit.\n")
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                break

            # store user in history
            self.conv_history.append({"role": "user", "content": user_query})

            ans, path = self.run_one_turn(user_query)

            # store assistant in history
            self.conv_history.append({"role": "assistant", "content": ans})

            print(f"\nAssistant: {ans}")
            print(f"[AgentPath] {' -> '.join(path)}\n")


def main():
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    pinecone_key = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "")
    pinecone_namespace = os.environ.get("PINECONE_NAMESPACE", "ns-2500")

    if not openai_key or not pinecone_key or not pinecone_index_name:
        raise RuntimeError(
            "Please set env vars: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME"
        )

    bot = Head_Agent(openai_key, pinecone_key, pinecone_index_name, pinecone_namespace=pinecone_namespace)
    bot.main_loop()


if __name__ == "__main__":
    main()
