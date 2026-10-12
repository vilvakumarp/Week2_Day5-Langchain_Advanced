# 🥗 Healthy Breakfast Generator

A Streamlit application built with **LangChain, OpenAI, and LangSmith** that generates personalized healthy breakfast ideas based on user preferences.

## Features

- **User Preferences:** Diet, health goal, cuisine, preparation time, allergies, and custom preferences.
- **LangChain Messages:** Uses `SystemMessage` and `HumanMessage` to structure prompts.
- **Guardrails:** Validates user inputs and generated outputs.
- **Structured Evaluation:** Uses Pydantic to return a quality score, pass/fail status, and feedback.
- **LangSmith Tracing:** Tracks application execution and model calls.

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI (`gpt-4o-mini`)
- LangSmith
- Pydantic

## Workflow

1. **User Input:** Select breakfast preferences in the Streamlit UI.
2. **Input Guardrails:** Validate preferences and reject basic prompt-injection patterns.
3. **Prompt Construction:** Build `SystemMessage` and `HumanMessage`.
4. **LLM Generation:** Call `llm.invoke(messages)` to generate five breakfast ideas.
5. **Output Guardrails:** Check the number of ideas, descriptions, and explicitly excluded ingredients.
6. **Evaluation:** Use a second LLM call with structured Pydantic output to assess quality.
7. **Display Results:** Show breakfast ideas, evaluation score, and feedback.
8. **LangSmith Tracing:** Inspect execution traces in the LangSmith dashboard.

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd breakfast
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```dotenv
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=breakfast-assignment
```

### 5. Run the application

```bash
streamlit run breakfast.py
```

## LangSmith Monitoring

Open [LangSmith](https://smith.langchain.com/) and select the `breakfast-assignment` project to inspect traces, inputs, outputs, latency, and token usage when available.

## Important Note

The ingredient guardrail uses basic text matching and does not guarantee allergy safety. Always verify ingredient labels and cross-contact risks independently.

---

**Learning Objectives:** LangChain messages, model invocation, prompt engineering, guardrails, structured output, LLM evaluation, and LangSmith tracing.
