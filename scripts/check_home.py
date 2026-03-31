import requests
r = requests.get('http://127.0.0.1:5000/home')
print('Status', r.status_code)
print('Contains Test User?', 'Test User' in r.text)

idx = r.text.find('Test User')
if idx!=-1:
    print(r.text[idx-60:idx+60])
else:
    print('Name not present in HTML')
