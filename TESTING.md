# Testing Guide

This guide explains how to test the Cascade Agent locally.

## Unit Testing

You can verify the classification and the AST-based math solver directly using Python:

```bash
pip install -r requirements.txt

# Test category classification
python3 -c "
from app.classifier import classify
print(classify('What is the capital of France?'))
print(classify('Write a function that reverses a string'))
"

# Test local math solver
python3 -c "
from app.local_solvers import solve_simple_arithmetic
print(solve_simple_arithmetic('What is 24 * 7?'))
"
```

## Integration Testing (Mock Server)

You can run the app against a mock API server to verify network client code, formatting, and general routing.

1. Start the mock server:
   ```bash
   pip install flask
   python mock_fireworks_server.py
   ```

2. Run the agent against the mock server:
   ```bash
   export FIREWORKS_API_KEY="test-key"
   export FIREWORKS_BASE_URL="http://localhost:8000/v1"
   export ALLOWED_MODELS="mock/small-8b,mock/large-70b"

   python -m app.main
   ```

3. Validate the output file format:
   ```bash
   python validate_submission.py input/tasks.json output/results.json
   ```

## Standardized Evaluation

To run the local 120-task evaluation suite against the real Fireworks API:

```bash
python run_eval.py
```

## Docker Testing

To test the agent inside the Docker container:

```bash
# Build the image
docker build --tag general-agent:test .

# Run the container
docker run --rm \
  -e FIREWORKS_API_KEY=test-key \
  -e FIREWORKS_BASE_URL=http://host.docker.internal:8000/v1 \
  -e ALLOWED_MODELS=mock/small-8b,mock/large-70b \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  general-agent:test
```
