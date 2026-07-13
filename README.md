# CascadeAgent (AMD Track 2)

A general-purpose routing agent designed for the AMD Developer Hackathon (Act II, Track 2). It dynamically routes tasks to optimal model tiers on Fireworks AI or resolves them locally using zero-cost code solvers.

## Core Architecture & Features

- **Dynamic Heuristic Classification**: Classifies incoming tasks into 8 distinct categories (Factual, Math, Sentiment, Summarization, NER, Code Debugging, Logical Reasoning, and Code Generation) using fast, zero-cost keyword and regex heuristics.
- **0-Token Cost Optimization (Gemma-First)**: Routes simple text, math, and logical categories to the cheapest **Gemma models first** (0-token billing) to maximize the track-specific quality score, falling back to premium tiers only if needed.
- **Specialized Premium Tiering**: 
  - **Kimi (code-specialized)** is prioritized for Code Generation and Code Debugging tasks to ensure high algorithmic accuracy.
  - **MiniMax** is prioritized for Named Entity Recognition (NER) to ensure strict compliance with structured JSON formats.
- **Zero-Cost Local Solvers**: Resolves simple arithmetic expressions directly in Python using a safe AST-based evaluator, bypassing external API calls entirely when confident.
- **Robust Sanitization Layer**: Strips Gemma thinking blocks, normalizes output JSON schemas, enforces formatting constraints (e.g. sentiment formats, bullet markers, removal of commas in large numbers), and handles unclosed code blocks defensively.

## Setup & Running

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running Locally

To run the agent locally, configure the following environment variables:

- `FIREWORKS_API_KEY`: Your Fireworks API Key.
- `FIREWORKS_BASE_URL`: The Fireworks API base URL.
- `ALLOWED_MODELS`: Comma-separated list of allowed model IDs.

Place tasks in `input/tasks.json` and run:

```bash
python -m app.main
```

Results will be written to `output/results.json`.

## Docker Usage

Build the image:

```bash
docker build --tag cascade-agent-amd:latest .
```

Run the container:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=xxx \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=model-1,model-2 \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  cascade-agent-amd:latest
```
