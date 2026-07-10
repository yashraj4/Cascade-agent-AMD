# Cascade Agent AMD

A general-purpose routing agent that dynamically routes tasks to local solvers or to the Fireworks API based on task category.

## Features

- **Dynamic Task Classification**: Classifies incoming tasks into distinct categories using light heuristics.
- **Local Solvers**: Uses lightweight local processing (VADER for sentiment analysis, spaCy for named entity recognition, and AST-based arithmetic evaluation) to resolve tasks without external API overhead.
- **Fireworks Integration**: Routes complex tasks to optimal model tiers via the Fireworks API.
- **Containerized Deployment**: Ready to be run inside a Docker container.

## Setup & Running

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
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
