import os
import subprocess
import time

os.environ["ALLOWED_MODELS"] = "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
os.environ["FIREWORKS_API_KEY"] = "test"
os.environ["FIREWORKS_BASE_URL"] = "http://localhost:8000"

server = subprocess.Popen(["python", "mock_fireworks_server.py"])
time.sleep(2)

try:
    res = subprocess.run(["python", "-m", "app.main"], capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)
finally:
    server.terminate()
