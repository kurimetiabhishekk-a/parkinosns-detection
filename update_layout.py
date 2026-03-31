import sys
import re

with open('templates/layout.html', 'r', encoding='utf-8') as f:
    content = f.read()

old_head = 

new_head = 

content = content.replace(old_head, new_head)

old_js = 

new_js = 

content = content.replace(old_js, new_js)

new_topbar = 

content = re.sub(r'<!-- Topbar Search -->.*?</ul>', new_topbar, content, flags=re.DOTALL)

old_content = 

new_content = 

content = content.replace(old_content, new_content)

with open('templates/layout.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('layout.html updated')
