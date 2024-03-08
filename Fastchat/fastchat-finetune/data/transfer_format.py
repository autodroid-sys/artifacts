# import json
# with open('questions1.json', 'r') as f:
#     questions = json.load(f)

# dummy_questions, mappings = [], {}
# question_id = 0
# for k, v in questions.items():
#     for kk, vv in v.items():
#         for quesid, ques in enumerate(vv):
#             mappings[question_id] = [k, kk, quesid]
#             question_id += 1
#             dummy_questions.append(ques)

# with open('dummy_questions1.json', 'w') as f:
#     json.dump(dummy_questions, f)

# with open('question_mappings1.json', 'w') as f:
#     json.dump(mappings, f)

import json
with open('autodroid2.json', 'r') as f:
    questions = json.load(f)
alpaca_data = []


for v in questions:
    question = v['conversations'][0]['value']
    answer = v['conversations'][1]['value']
    alpaca_data.append({
    'instruction': question,
    'input': '',
    'output': answer
    })
print(len(alpaca_data))
with open('alpaca_ft_data.json', 'w') as f:
    json.dump(alpaca_data, f)

