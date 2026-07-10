import subprocess
subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', 'Fix NER JSON formatting and add exponential backoff for API rate limits'])
subprocess.run(['git', 'push', 'origin', 'main'])
subprocess.run(['git', 'tag', 'v2'])
subprocess.run(['git', 'push', 'origin', 'v2'])
