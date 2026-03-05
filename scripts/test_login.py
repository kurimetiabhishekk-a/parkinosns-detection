import requests
s = requests.Session()
url = 'http://127.0.0.1:5000/login'
resp = s.post(url, data={'email':'test@local','password':'password123'}, allow_redirects=False)
print('POST status', resp.status_code)
print('Headers:', resp.headers)
if resp.status_code in (301,302,303,307,308):
    loc = resp.headers.get('Location')
    print('Redirect to', loc)
    follow = s.get('http://127.0.0.1:5000'+loc)
    print('Follow status', follow.status_code)
    print('Follow length', len(follow.text))
else:
    print('Body length', len(resp.text))
