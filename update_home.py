import sys
import re

with open('templates/home.html', 'r', encoding='utf-8') as f:
    content = f.read()

hero_banner = 

content = re.sub(r'^{% extends [^}]*%}.*?(<div class="row">)', hero_banner + r'\n\1', content, flags=re.DOTALL | re.MULTILINE)

old_cards_regex = r'<div class="row">.*?<!-- Content Row -->'

new_cards = 

content = re.sub(old_cards_regex, new_cards, content, flags=re.DOTALL)

with open('templates/home.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('home.html updated')
