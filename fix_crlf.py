import sys
for f in ['requirements.txt', 'Dockerfile']:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            data = file.read()
        with open(f, 'w', encoding='utf-8', newline='\n') as file:
            file.write(data.replace('\r\n', '\n'))
        print(f"Fixed {f}")
    except Exception as e:
        print(f"Error {f}: {e}")
