import json
import re

def extract_outputv1(text):
    pattern = r"id=(\d+)"
    match = re.search(pattern, text)
    if match:
        id_value = int(match.group(1))
    else:
        # print(text)
        # id_value = input("id not found, please input: ")
        id_value = -1
    
    input_text = 'N/A'

    match = re.search(r"input text=([^\s]+)", text)
    if match:
        input_text = match.group(1)
        # pdb.set_trace()
    else:
        match = re.search(r"input=([^\s]+)", text)
        if match:
            input_text = match.group(1)
        # else:
        #     print('input wrong')
        #     print(text)
        #     pdb.set_trace()
    
    return id_value, 'tap', input_text


with open('../fastchat-finetune/autodroid_output.json', 'r') as f:
    vicuna_data = json.load(f)


with open('ground_truth.json', 'r') as f:
    gt = json.load(f)


correct_num, all_num = 0, 0
end_right_num, choose_right_num, input_right_num = 0, 0, 0
end_num, choose_num, input_num = 0, 0, 0
task_num, task_complete_num = 0, 0
for app, app_task in gt.items():
    for task_str, task_data in app_task.items():
        task_num += 1
        task_correct = True
        task_name = task_data['task']
        task_profile = task_data['profile']
        for stepid, step in enumerate(task_profile):
            # mapping_str_list = [app, task_str, str(stepid)]
            # mapping_str = '-'.join(mapping_str_list)
            # try:
            #     vicuna_output_id = reversed_mapping[mapping_str]
            # except:
            #     print(mapping_str, 'not found')
            #     continue
            vicuna_output = vicuna_data[app][task_str][stepid]
            id, action, input_text = extract_outputv1(vicuna_output)
            all_num += 1
            if id == -1 and step['label'][0] == -1:
                correct_num += 1
                end_right_num += 1
                end_num += 1
            elif id != -1 and step['label'][0] == -1:
                end_num += 1
            elif id == step['label'][0] and step['label'][1] == 'null':
                correct_num += 1
                choose_right_num += 1
                choose_num += 1
            elif id != step['label'][0] and step['label'][1] == 'null':
                choose_num += 1
                task_correct = False
            elif id == step['label'][0] and step['label'][1] in input_text:
                correct_num += 1
                input_right_num += 1
                input_num += 1
            elif step['label'][1] != 'null':
                input_num += 1
                task_correct = False
            else:
                print(vicuna_output, '\n*******************')
                print(step['label'], '\n')
        if task_correct:
            task_complete_num += 1

print(f'action accuracy: {correct_num/all_num}\nend accuracy: {end_right_num/end_num}\ntap accuracy: {choose_right_num/choose_num}\ninput accuracy: {input_right_num/input_num}\ntask completion rate: {task_complete_num/task_num}')