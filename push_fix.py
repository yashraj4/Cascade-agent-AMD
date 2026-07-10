import subprocess
subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', 'Restore Dockerfile and requirements.txt'])
subprocess.run(['git', 'push', 'origin', 'main'])
subprocess.run(['git', 'tag', '-f', 'v1'])
subprocess.run(['git', 'push', '-f', 'origin', 'v1'])
