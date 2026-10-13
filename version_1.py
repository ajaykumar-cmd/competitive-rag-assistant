"""
version_1.py - Exam question generator using LangChain.

Providers: gemini (default), openai, anthropic
Set in .env:
    PROVIDER=gemini
    GOOGLE_API_KEY=...        # for gemini
    OPENAI_API_KEY=...        # for openai
    ANTHROPIC_API_KEY=...     # for anthropic

Install:
    pip install langchain langchain-core langchain-google-genai python-dotenv pydantic
    pip install langchain-openai        # optional
    pip install langchain-anthropic     # optional
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "gemini").lower()
print(f"Provider: {PROVIDER}")

# ---------------------------------------------------------------------------
# Model setup (with retries + fallbacks to survive 503 / 429 errors)
# ---------------------------------------------------------------------------

# (primary model, [fallback models]) for each provider
MODELS = {
    "gemini": ("gemini-3.8-flash", ["gemini-3.7-flash", "gemini-3.5-flash-lite"]),
    "openai": ("gpt-4o-mini", ["gpt-4o"]),  # check OpenAI's docs for current names
    "anthropic": ("claude-sonnet-5-5", ["claude-haiku-5-5"]),
}


def _build_model(provider: str, model_name: str):
    """Create a single chat model with built-in retry/backoff."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            max_retries=5,  # retries 503/429 with backoff
            timeout=60,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=0.3, max_retries=5, timeout=60)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=0.3, max_retries=5, timeout=60)
    raise ValueError(f"Unknown provider: {provider}")


def get_llm(schema: Optional[type] = None):
    """
    Return a runnable LLM: primary model with fallback models.
    If `schema` (a Pydantic class) is given, each model returns structured output.
    """
    primary_name, fallback_names = MODELS[PROVIDER]
    # Allow overriding the primary model from .env, e.g. GEMINI_MODEL=gemini-2.5-pro
    primary_name = os.getenv(f"{PROVIDER.upper()}_MODEL", primary_name)

    models = [_build_model(PROVIDER, n) for n in [primary_name] + fallback_names]
    if schema is not None:
        models = [m.with_structured_output(schema) for m in models]

    primary, *backups = models
    return primary.with_fallbacks(backups) if backups else primary


# ---------------------------------------------------------------------------
# 1. Basic chain: prompt -> LLM -> string
# ---------------------------------------------------------------------------

text_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert exam coach. Be accurate, concise and exam-focused.",
        ),
        (
            "human",
            "Exam: {exam}\nTopic: {topic}\n\n"
            "Give a short study summary of this topic (key points, important "
            "articles/terms, and common exam traps) in under 250 words.",
        ),
    ]
)


def run_text_chain(exam: str, topic: str) -> str:
    chain = text_prompt | get_llm() | StrOutputParser()
    return chain.invoke({"exam": exam, "topic": topic})


# ---------------------------------------------------------------------------
# 2. Structured output chain: generate MCQs as typed objects
# ---------------------------------------------------------------------------


class MCQ(BaseModel):
    question: str = Field(description="The question text")
    options: List[str] = Field(description="Exactly four answer options")
    answer: str = Field(description="The correct option, copied exactly from options")
    explanation: str = Field(description="One or two lines explaining the answer")


class MCQSet(BaseModel):
    exam: str
    topic: str
    questions: List[MCQ]


mcq_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert question-paper setter."),
        (
            "human",
            "Create {n} multiple-choice questions for {exam} on the topic "
            "'{topic}'. Each must have four options and one correct answer.",
        ),
    ]
)


def run_mcq_chain(exam: str, topic: str, n: int = 3) -> MCQSet:
    chain = mcq_prompt | get_llm(schema=MCQSet)
    return chain.invoke({"exam": exam, "topic": topic, "n": n})


# ---------------------------------------------------------------------------
# 3. Streaming chain: print tokens as they arrive
# ---------------------------------------------------------------------------


def run_stream_chain(exam: str, topic: str) -> None:
    chain = text_prompt | get_llm() | StrOutputParser()
    for chunk in chain.stream({"exam": exam, "topic": topic}):
        print(chunk, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_mcqs(result: MCQSet) -> None:
    for i, q in enumerate(result.questions, 1):
        print(f"\nQ{i}. {q.question}")
        for letter, opt in zip("ABCD", q.options):
            print(f"   {letter}) {opt}")
        print(f"   Answer: {q.answer}")
        print(f"   Why: {q.explanation}")


if __name__ == "__main__":
    EXAM = "UPSC Prelims"
    TOPIC = "Fundamental Rights"

    try:
        print("\n=== 1. Basic chain ===")
        print(run_text_chain(EXAM, TOPIC))

        print("\n=== 2. Structured output (MCQs) ===")
        print_mcqs(run_mcq_chain(EXAM, TOPIC, n=3))

        print("\n=== 3. Streaming ===")
        run_stream_chain(EXAM, TOPIC)

    except Exception as e:
        print(f"\nRequest failed after retries and fallbacks: {e}")
        print("Try again in a few minutes, or set PROVIDER to another provider in .env.")