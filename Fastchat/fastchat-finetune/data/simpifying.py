import json
with open('autodroid.json', 'r') as f:
    convers = json.load(f)
convers_new = convers[:1000]

with open('autodroid_simple.json', 'w') as f:
    json.dump(convers_new, f)

