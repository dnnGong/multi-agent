#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "machine-learning.pdf"
DEFAULT_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "machine_learning_vector_store.json"


@dataclass
class LocalSearchResult:
    id: str
    score: float
    text: str
    meta: Dict[str, Any]


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


PathLike = Union[os.PathLike, str]


def extract_pdf_pages(pdf_path: PathLike) -> List[Dict[str, Any]]:
    """
    Extract page-level text from the ML textbook.

    Uses pypdf first, then PyPDF2 for older environments. The project keeps this
    dependency light because Streamlit only needs the built JSON store at runtime.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader_cls = None
    try:
        from pypdf import PdfReader  # type: ignore

        reader_cls = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader_cls = PdfReader
        except Exception as exc:
            raise RuntimeError(
                "PDF extraction requires pypdf. Install it with `pip install pypdf` "
                "or use an already-built vector store JSON."
            ) from exc

    reader = reader_cls(str(path))
    pages: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": page_idx, "text": text})
    if not pages:
        raise RuntimeError(f"No extractable text found in {path}")
    return pages


def chunk_pages(
    pages: Iterable[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: List[Dict[str, Any]] = []
    for page in pages:
        text = page["text"]
        page_no = page["page"]
        start = 0
        local_idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = _clean_text(text[start:end])
            if len(piece) >= 80:
                chunks.append(
                    {
                        "id": f"page-{page_no:04d}-chunk-{local_idx:03d}",
                        "text": piece,
                        "metadata": {
                            "page": page_no,
                            "chunk": local_idx,
                            "source": str(DEFAULT_PDF_PATH.name),
                        },
                    }
                )
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
            local_idx += 1
    return chunks


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> List[List[float]]:
    vectors: List[List[float]] = []
    for batch in _batched(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


class LocalVectorStore:
    def __init__(
        self,
        store_path: PathLike = DEFAULT_STORE_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.store_path = Path(store_path)
        self.embedding_model = embedding_model
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None

    @property
    def is_loaded(self) -> bool:
        return bool(self.documents) and self.embeddings is not None

    def load(self) -> "LocalVectorStore":
        if not self.store_path.exists():
            raise FileNotFoundError(f"Vector store not found: {self.store_path}")
        with self.store_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.embedding_model = payload.get("embedding_model", self.embedding_model)
        self.documents = payload["documents"]
        self.embeddings = np.array(payload["embeddings"], dtype=np.float32)
        self._norms = np.linalg.norm(self.embeddings, axis=1)
        return self

    def save(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedding_model,
            "documents": self.documents,
            "embeddings": self.embeddings.tolist(),
        }
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def build_from_pdf(
        self,
        client: OpenAI,
        pdf_path: PathLike = DEFAULT_PDF_PATH,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        batch_size: int = 64,
    ) -> "LocalVectorStore":
        pages = extract_pdf_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise RuntimeError("No chunks generated from the PDF")
        vectors = embed_texts(
            client,
            [chunk["text"] for chunk in chunks],
            model=self.embedding_model,
            batch_size=batch_size,
        )
        self.documents = chunks
        self.embeddings = np.array(vectors, dtype=np.float32)
        self._norms = np.linalg.norm(self.embeddings, axis=1)
        self.save()
        return self

    def search(
        self,
        client: OpenAI,
        query: str,
        k: int = 5,
    ) -> List[LocalSearchResult]:
        if not self.is_loaded:
            self.load()
        if self.embeddings is None or self._norms is None:
            return []

        query_vec = np.array(
            embed_texts(client, [query], model=self.embedding_model, batch_size=1)[0],
            dtype=np.float32,
        )
        query_norm = float(np.linalg.norm(query_vec))
        if math.isclose(query_norm, 0.0):
            return []

        scores = (self.embeddings @ query_vec) / (self._norms * query_norm + 1e-12)
        top_idx = np.argsort(scores)[::-1][:k]
        results: List[LocalSearchResult] = []
        for idx in top_idx:
            doc = self.documents[int(idx)]
            results.append(
                LocalSearchResult(
                    id=str(doc["id"]),
                    score=float(scores[int(idx)]),
                    text=str(doc["text"]),
                    meta=dict(doc.get("metadata", {})),
                )
            )
        return results


def build_store_from_args(args: argparse.Namespace) -> Path:
    client = OpenAI(api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    store = LocalVectorStore(args.out, embedding_model=args.embedding_model)
    store.build_from_pdf(
        client=client,
        pdf_path=args.pdf,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    return Path(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local RAG vector store from the ML textbook PDF.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH), help="Path to machine-learning.pdf")
    parser.add_argument("--out", default=str(DEFAULT_STORE_PATH), help="Output JSON vector store path")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--openai_api_key", default=None)
    args = parser.parse_args()
    out = build_store_from_args(args)
    print(f"Built local vector store: {out}")


if __name__ == "__main__":
    main()
