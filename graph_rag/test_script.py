########################################################
########################################################




# from neo4j import GraphDatabase

# driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))
# with driver.session(database="neo4j") as session:
#     result = session.run("RETURN 'Connection Successful' AS msg")
#     print(result.single()["msg"])




########################################################
########################################################




# from graph_rag.llm_client import get_llm_client
# from graph_rag.retriever import get_retriever


# def main() -> None:
#     retriever = get_retriever()
#     llm = get_llm_client()

#     print("[test_script] Building Alzheimer's Disease context from Neo4j...")
#     context = retriever.build_ad_context()

#     lines = context.splitlines()
#     print("\n=== CONTEXT PREVIEW (first 40 lines) ===")
#     for line in lines[:40]:
#         print(line)
#     print(f"\n[test_script] Context has {len(lines)} lines total.\n")

#     question = (
#         "Using ONLY the biomarkers explicitly listed under the section "
#         "'Biomarkers associated with AD (HAS_BIOMARKER)', summarize which "
#         "biomarkers in CSF or plasma/serum tend to increase or decrease in "
#         "Alzheimer's disease. Do not mention any biomarker that is not "
#         "present in that section."
#     )

#     print("=== QUESTION ===")
#     print(question)
#     print("\n[test_script] Sending question + context to LLM (llama3.2:3b via Ollama)...\n")

#     answer = llm.simple_qa(
#         question=question,
#         context=context,
#         temperature=0.0,      # <---- IMPORTANT
#         max_tokens=400,
#     )

#     print("=== LLM ANSWER ===\n")
#     print(answer)
#     print("\n[test_script] Done.")


# if __name__ == "__main__":
#     main()




########################################################
########################################################




# from graph_rag.router import build_context_for_question, describe_route
# from graph_rag.llm_client import get_llm_client

# question = "What are the main biomarkers in CSF that decrease in Alzheimer's disease?"
# route = build_context_for_question(question)

# print(describe_route(route))         # see which intent was picked
# answer = get_llm_client().simple_qa(
#     question=question,
#     context=route.context,
#     temperature=0.0,
# )
# print(answer)




########################################################
########################################################




# """
# graph_rag.test_script
# ---------------------

# Minimal end-to-end test of the Graph RAG workflow:

#     Question -> Router/Intent -> Graph Retrieval -> LLM Answer

# Run from project root:

#     python -m graph_rag.test_script
# """

# from __future__ import annotations

# from typing import List

# from graph_rag.pipeline import get_pipeline


# def run_single_question(question: str, *, show_context: bool = False) -> None:
#     """
#     Helper to run a single question through the GraphRAGPipeline
#     and pretty-print the result.
#     """
#     pipeline = get_pipeline()

#     print("\n" + "=" * 80)
#     print("QUESTION:")
#     print(question)
#     print("=" * 80)

#     # Call full pipeline
#     result = pipeline.answer(
#         question=question,
#         temperature=0.0,      # deterministic-ish for debugging
#         max_tokens=400,
#         return_context=show_context,
#     )

#     # Basic metadata
#     print(f"\nIntent:   {result['intent_type']}")
#     print(f"Notes:    {result['intent_notes']}")
#     print(f"Strategy: {result['strategy']}")

#     # Answer
#     print("\nANSWER:\n")
#     print(result["answer"])

#     # Optional: context preview
#     if show_context and "context" in result:
#         lines = result["context"].splitlines()
#         print("\n--- CONTEXT PREVIEW (first 40 lines) ---")
#         for line in lines[:40]:
#             print(line)
#         print(f"\n[Context has {len(lines)} total lines]\n")

#     # Optional: debug info
#     if result.get("debug"):
#         print("--- DEBUG ---")
#         for k, v in result["debug"].items():
#             print(f"{k}: {v}")


# def main() -> None:
#     """
#     Run a small battery of questions to verify that:

#     - Intent classification is reasonable
#     - Graph context builds without errors
#     - LLM returns a non-empty answer
#     """
#     questions: List[str] = [
#         # Biomarker-style (BIOMARKER intent)
#         "Which biomarkers in CSF or plasma/serum tend to decrease in "
#         "Alzheimer's disease?",

#         # Phenotype / symptoms (PHENOTYPE intent)
#         "What are the main clinical symptoms or phenotypes of Alzheimer's disease?",

#         # Drug / trial status (DRUG_TRIAL intent)
#         "Which Alzheimer's drugs are in phase 3 or approved, and which pathways "
#         "do they affect?",
#     ]

#     # For the first question, also show context preview
#     for i, q in enumerate(questions):
#         show_context = (i == 0)
#         run_single_question(q, show_context=show_context)


# if __name__ == "__main__":
#     main()




########################################################
########################################################




"""
graph_rag.test_script
---------------------

Quick end-to-end smoke test for the Graph RAG pipeline.

Usage (from project root):

    (venv) $ python -m graph_rag.test_script

This will:
- Build the Alzheimer’s ultra-compact context from Neo4j
- Route a few different questions (biomarker, phenotype, drug/trial, general)
- Call the local LLM via Ollama
- Print intent, answer, and a short context preview
"""

from __future__ import annotations

from typing import List

from .pipeline import get_pipeline


SEPARATOR = "=" * 80


def _print_context_preview(context: str, max_lines: int = 40) -> None:
    """Print the first `max_lines` of the context for inspection."""
    lines: List[str] = context.splitlines()
    print("\n--- CONTEXT PREVIEW (first {} lines) ---".format(max_lines))
    for line in lines[:max_lines]:
        print(line)
    print(f"\n[Context has {len(lines)} total lines]\n")


def run_single_question(question: str, *, show_context: bool = True) -> None:
    """Run the full pipeline for a single question and pretty-print output."""
    pipe = get_pipeline()

    print(SEPARATOR)
    print("QUESTION:")
    print(question)
    print(SEPARATOR)
    print("")

    result = pipe.answer(
        question,
        temperature=0.0,        # make behavior as deterministic as possible
        max_tokens=400,
        return_context=show_context,
    )

    print(f"Intent:   {result['intent_type']}")
    print(f"Notes:    {result['intent_notes']}")
    print(f"Strategy: {result['strategy']}")
    print("")

    print("ANSWER:\n")
    print(result["answer"])
    print("")

    if show_context and "context" in result:
        _print_context_preview(result["context"])

    print("--- DEBUG ---")
    for k, v in result["debug"].items():
        print(f"{k}: {v}")
    print("")


def main() -> None:
    """
    Run a small battery of diverse questions to exercise the pipeline.

    - Biomarker-specific (CSF / plasma, decreased)
    - Phenotype / symptoms
    - Drug & trial status + pathways
    - General high-level question
    """
    questions = [
        # 1) Highly specific biomarker query
        "Which biomarkers in CSF or plasma/serum tend to decrease in Alzheimer's disease?",

        # 2) Clinical phenotypes
        "What are the main clinical symptoms or phenotypes of Alzheimer's disease?",

        # 3) Drugs in phase 3 / approved + pathways
        "Which Alzheimer's drugs are in phase 3 or approved, and which pathways do they affect?",

        # 4) Very high-level / vague question
        "Give me a high-level overview of key biomarkers, drugs, and pathways in Alzheimer's disease.",
    ]

    for q in questions:
        run_single_question(q, show_context=True)


if __name__ == "__main__":
    main()




########################################################
########################################################