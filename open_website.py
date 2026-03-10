import subprocess
import os
import re
import time

print("Starting public tunnel...")
# Start localtunnel
process = subprocess.Popen(['npx', '-y', 'localtunnel', '--port', '5000', '--subdomain', 'parkisense-live-app'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Wait for the URL to appear
url = None
for i in range(15):  # wait up to 15 seconds
    line = process.stdout.readline()
    if line:
        print("Output:", line.strip())
        match = re.search(r'(https://[a-zA-Z0-9-]+\.loca\.lt)', line)
        if match:
            url = match.group(1)
            break
    time.sleep(1)

if url:
    print(f"Opening public URL in your browser: {url}")
    os.startfile(url)
    
    print("\nIMPORTANT: Keep this terminal open to keep the website online!")
    process.wait()
else:
    print("Could not get a public URL. Opening local version instead...")
    os.startfile('http://127.0.0.1:5000')
