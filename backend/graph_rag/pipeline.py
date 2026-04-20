"""
graph_rag.pipeline
------------------

End-to-end Graph RAG pipeline:

    user_question
        -> router (intent + graph context)
        -> LLM (Ollama llama3.2:3b)
        -> answer + debug metadata

Typical usage
-------------

    from graph_rag.pipeline import get_pipeline

    pipeline = get_pipeline()

    result = pipeline.answer(
        "Which biomarkers in CSF decrease in Alzheimer's disease?",
        return_context=True,
    )

    print("Intent:", result["intent_type"])
    print("Answer:\n", result["answer"])
    # Optionally inspect:
    # print(result["context"])
    # print(result["debug"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .router import build_context_for_question, RouteResult
from .retriever import GraphRetriever, get_retriever
from .llm_client import LLMClient, get_llm_client


# ---------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------


@dataclass
class GraphRAGPipeline:
    """
    End-to-end Graph RAG pipeline.

    Responsibilities
    ----------------
    - Take a user question (string).
    - Use the router to:
        * classify the intent
        * build a graph-derived textual context.
    - Call the LLM with that context to generate an answer.
    - Return a structured dict with answer + metadata.

    Parameters
    ----------
    retriever:
        GraphRetriever instance (Neo4j).
    llm_client:
        LLMClient instance (Ollama llama3.2:3b by default).
    """

    retriever: GraphRetriever
    llm_client: LLMClient

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 400,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a user question using the Alzheimer KG + LLM.

        Steps
        -----
        1) Route the question to an intent + retrieval strategy.
        2) Build graph-derived context (currently AD-centered).
        3) Call the LLM with that context using a RAG-style prompt.
        4) Return answer + metadata.
        """

        # 1) Route question → intent + context
        route: RouteResult = build_context_for_question(
            question=question,
            retriever=self.retriever,
        )

        # 2) Short-circuit: no LLM call for out-of-scope or missing-KG cases
        if route.context == "__OUT_OF_SCOPE__":
            answer_text = (
                "This assistant only answers questions about Alzheimer's disease — "
                "biomarkers, therapeutics, clinical phenotypes, affected biological "
                "pathways, and related genes/proteins. Please rephrase your question "
                "with that focus."
            )
        elif "does not appear to contain an Alzheimer's" in route.context:
            answer_text = route.context
        else:
            # Build an enriched question that encodes intent + safety constraints
            intent_label = route.intent.type.name
            prompt_question = (
                "You are an assistant answering questions *strictly* based on the "
                "Alzheimer’s disease graph context provided separately.\n\n"
                f"Query type (intent): {intent_label}\n"
                "Rules:\n"
                "1. Use only the information in the context. Do NOT invent biomarkers, "
                "drugs, genes, or symptoms that are not explicitly listed.\n"
                "2. If the context does not contain enough information to answer "
                "part of the question, say so explicitly.\n"
                "3. When listing items (biomarkers, drugs, pathways, phenotypes), "
                "only mention entities that you see in the context text.\n\n"
                f"User question: {question}"
            )

            # If temperature is not given, you can force 0.0 here for max determinism
            effective_temp = 0.0 if temperature is None else temperature

            answer_text: str = self.llm_client.simple_qa(
                question=prompt_question,
                context=route.context,
                temperature=effective_temp,
                max_tokens=max_tokens,
            )

        # 3) Package result
        result: Dict[str, Any] = {
            "question": question,
            "answer": answer_text,
            "intent_type": route.intent.type.name,
            "intent_notes": route.intent.notes,
            "strategy": route.strategy_name,
            "debug": route.debug,
        }

        if return_context:
            result["context"] = route.context
            if route.raw_data:
                result["evidence"] = route.raw_data

        return result


# ---------------------------------------------------------------------
# Singleton-style accessor
# ---------------------------------------------------------------------


_pipeline: Optional[GraphRAGPipeline] = None


def get_pipeline() -> GraphRAGPipeline:
    """
    Get (and lazily create) the project-wide GraphRAGPipeline.

    Uses the shared GraphRetriever + LLMClient singletons.
    """
    global _pipeline
    if _pipeline is None:
        retriever = get_retriever()
        llm_client = get_llm_client()
        _pipeline = GraphRAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
        )
    return _pipeline


# ---------------------------------------------------------------------
# Pydantic request / response models (imported by main.py)
# ---------------------------------------------------------------------


class QuestionRequest(BaseModel):
    question: str
    temperature: Optional[float] = None
    max_tokens: int = 400
    return_context: bool = False


class AnswerResponse(BaseModel):
    answer: str
    intent_type: str
    intent_notes: Optional[str]
    strategy: str
    context: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None