import pdb
import re
import ast
import requests
from rich.console import Console
import time
import random
import json
import networkx as nx
import yaml
from openai import OpenAI
import os
CONSOLE = Console()

def get_id_from_view_desc(view_desc):
    '''
    given a view(UI element), get its ID
    '''
    return int(re.findall(r'id=(\d+)', view_desc)[0])

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_view_without_id(view_desc):
    '''
    remove the id from the view
    '''
    element_id_pattern = r'element \d+: (.+)'
    view_without_id = re.findall(element_id_pattern, view_desc)
    if view_without_id:
        return view_without_id[0]
    
    id = re.findall(r'id=(\d+)', view_desc)[0]
    id_string = ' id=' + id
    return re.sub(id_string, '', view_desc)

def remove_bbox(view_desc):
    return re.sub(r' bound_box=\d+,\d+,\d+,\d+', '', view_desc)

def get_height_width(view_desc):
    # Regular expression to extract the numbers from the 'bound_box' attribute
    numbers = re.findall(r'bound_box=(\d+),(\d+),(\d+),(\d+)', view_desc)
    # Extract the numbers if a match is found (assuming there's only one bound_box in the string)
    bound_box_numbers = numbers[0] if numbers else None
    x1, y1, x2, y2 = int(bound_box_numbers[0]), int(bound_box_numbers[1]), int(bound_box_numbers[2]), int(bound_box_numbers[3])
    width, height = int(bound_box_numbers[2]) - int(bound_box_numbers[0]), int(bound_box_numbers[3]) - int(bound_box_numbers[1])
    return height, width

def wrap_element_desc(element_desc):
    '''
    remove id and bbox from the element description, and wrap the element description with a 'hxw=' tag
    '''
    element_desc_without_id = get_view_without_id(element_desc)
    element_desc_without_id_and_bbox = remove_bbox(element_desc_without_id)
    element_height, element_width = get_height_width(element_desc)
    element_desc_final = f'{element_desc_without_id_and_bbox} hxw={element_height}x{element_width}'
    return element_desc_final

def load_txt_file(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    return file_contents


def load_json_file(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json_file(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f)


def query_gpt(prompt):
    URL = 'https://gpt.yanghuan.site/api/openai/v1/chat/completions'
    body = {"model": "gpt-3.5-turbo-1106", "messages": [{"role": "user", "content": prompt}], "stream": False,
            "temperature": 0.5}
    headers = {'Content-Type': 'application/json', 'path': '/api/openai/v1/chat/completions',
               'Authorization': 'Bearer ak-LgOVMZ0nHqetz55TjSBlZia9u7QvYxB2kNvNlx8ACsjkZexD'}
    answers = requests.post(url=URL, json=body, headers=headers)
    print(answers.text)
    index = answers.text.find('{')
    dict_responses = ast.literal_eval(answers.text[index:])
    return dict_responses['choices'][0]['message']['content']


def debug(prompt: str, console: Console, identifier=""):
    url = 'https://gpt.yanghuan.site/api/openai/v1/chat/completions'
    retry = 0
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.25,
    }
    headers = {
        "Content-Type": "application/json",
        "path": "/api/openai/v1/chat/completions",
        "Authorization": "Bearer ak-LgOVMZ0nHqetz55TjSBlZia9u7QvYxB2kNvNlx8ACsjkZexD",
    }
    while retry < 12:
        try:
            result = requests.post(url=url, json=body, headers=headers)
            index = result.text.find('{')
            dict_response = ast.literal_eval(result.text[index:])
            res = dict_response["choices"][0]["message"]["content"]
            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )
            break
        except:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times."
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    else:
        if identifier:
            console.log(f"Task [red]{identifier}[/red] fails. Shutdown threadpool.")
        return None
    return res


def extract_control_properties(view_description):
    '''
    :param view_description: a natural language string of a control
    :return: view_properties: {'type': type, 'desc': desc, 'text': text}
    '''
    type_pattern = r'<([^>]+)>'
    matches = re.findall(type_pattern, view_description)
    if ' ' in matches[0]:
        type_pattern = r"<(.*?)\s"
        matches = re.findall(type_pattern, view_description)
    control_type = matches[0]

    desc_pattern = r"text=(.*?)\s"
    match = re.search(desc_pattern, view_description)
    if match:
        desc = match.group(1)
    else:
        desc_pattern = r'text=(.*?)>'
        match = re.search(desc_pattern, view_description)
        if match:
            desc = match.group(1)
        else:
            desc = ''

    text_pattern = r">(.*?)<\/"
    match = re.search(text_pattern, view_description)
    if match:
        text = match.group(1)
    else:
        text = ''

    return {'type': control_type, 'desc': desc, 'text': text}


def get_control_code(view_properties, input_content=None, from_description=True):
    if from_description:
        view_properties = extract_control_properties(view_properties)
    
    desc = view_properties['text'] or view_properties['desc']

    if view_properties['type'] == 'input':
        action_desc = f'find_element_by_xpath("//android.widget.EditText[contains(@content-desc, \'{desc}\')]").sendKeys("{input_content}")\n'.replace(
            "''", "'")
        # action_desc += f'input_box.sendKeys("{input_content}")\n'.replace("''", "'")
        # action_desc = f'input_box=find_element_by_xpath("//android.widget.EditText[contains(@content-desc, \'{desc}\')]")\n'.replace(
        #     "''", "'")
        # action_desc += f'input_box.sendKeys("{input_content}")\n'.replace("''", "'")
    elif view_properties['type'] == 'button':

        action_desc = f'find_element_by_xpath("//android.widget.Button[contains(@content-desc, \'{desc}\')]").click()\n'.replace(
            "''", "'")
        # action_desc += f'button.click()\n'.replace("''", "'")
    elif view_properties['type'] == 'checkbox':

        action_desc = f'find_element_by_xpath("//android.widget.CheckBox[contains(@content-desc, \'{desc}\')]").click()\n'.replace(
            "''", "'")
        # action_desc += f'checkbox.click()\n'.replace("''", "'")

    return action_desc


def get_control_codev2(view_properties, input_content=None, from_description=True, prefix_actions=True):
    '''
    clickButton('Button');
    checkCheckbox('Checkbox');
    uncheckCheckbox('Checkbox');
    setInputValue('usernameInput', '张三');
    scrollUp();
    '''
    if from_description:
        view_properties = extract_control_properties(view_properties)
    desc = view_properties['text'] or view_properties['desc'] .replace("''", "'")
    if view_properties['type'] == 'input':
        if prefix_actions and input_content:
            action_desc = f'setInputValue("{desc}", "{input_content}");'
        else:
            action_desc = f'<input type="{desc}">'
    elif view_properties['type'] == 'button':
        action_desc = f'click <button>{desc}</button>;' if prefix_actions else f'<button>{desc}</button>'
    elif view_properties['type'] == 'checkbox':
        action_desc = f'click <checkbox>{desc}</checkbox>;' if prefix_actions else f'<checkbox>{desc}</checkbox>'
    return action_desc

def debug_query_gptv2(prompt: str, model_name: str, retry_times=12):
    # print(prompt)
    client = OpenAI(
        base_url='https://api.openai-proxy.org/v1',
        # This is the default and can be omitted
        api_key='sk-otu6p5ykBAemvGndQbbed7mSKD5deXNESjo1i6TF482pjT4Y'
    )
    completion = client.chat.completions.create(
    messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        timeout=15
    )
    res = completion.choices[0].message.content
    return res


def query_gptv2(prompt: str, model_name: str, retry_times=12):
    # print(prompt)
    client = OpenAI(
        base_url='https://api.openai-proxy.org/v1',
        # This is the default and can be omitted
        api_key='sk-otu6p5ykBAemvGndQbbed7mSKD5deXNESjo1i6TF482pjT4Y'
    )
    retry = 0
    while retry < retry_times:
        try:
            completion = client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                timeout=15
            )
            res = completion.choices[0].message.content
            break
        except:
            retry += 1
            print(f'\n\n\nWARNING: API failed {retry} times. Retrying...\n\n\n')
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    return res


def queryGPT(prompt, console=CONSOLE, identifier='', model_name="gpt-3.5-turbo"):
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-05-15",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    retry = 0
    while retry < 12:
        try:
            result = client.chat.completions.create(
                model="gpt-4-32k",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            # res = result["choices"][0]["message"]["content"]
            # res = result.model_dump_json(indent=2)
            res = result.choices[0].message.content
            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )
            break
        except:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times."
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    else:
        if identifier:
            console.log(f"Task [red]{identifier}[/red] fails. Shutdown threadpool.")
        return None
    return res

def debug(prompt: str, console: Console, identifier=""):
    client = OpenAI(
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-otu6p5ykBAemvGndQbbed7mSKD5deXNESjo1i6TF482pjT4Y'
    )
    retry = 0
    while retry < 1:
        try:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo-16k",
                timeout=15
            )
            res = completion.choices[0].message.content
            
            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )
            break
        except:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times."
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    else:
        if identifier:
            console.log(f"Task [red]{identifier}[/red] fails. Shutdown threadpool.")
        return None
    return res


def retry_query_gptv2(prompt: str, console=CONSOLE, identifier="", model_name="gpt-3.5-turbo"):
    import openai
    openai.api_base = 'https://api.openai-proxy.org/v1'
    openai.api_key = 'sk-otu6p5ykBAemvGndQbbed7mSKD5deXNESjo1i6TF482pjT4Y'
    retry = 0
    while retry < 12:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    # {"role": "system", "content": "You are a personal agent running in a smartphone."},
                    {"role": "user", "content": prompt},
                ],
                timeout=15
            )
            res = response["choices"][0]["message"]["content"]
            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )
            break
        except:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times."
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    else:
        if identifier:
            console.log(f"Task [red]{identifier}[/red] fails. Shutdown threadpool.")
        return None
    return res

def split_state(state_desc):
    views = state_desc.split('>\n')
    for viewid in range(len(views)):
        if views[viewid][-1] != '>':
            views[viewid] = views[viewid] + '>'
    return views

def get_scroll_description(scroll_control, from_yaml=True):
    if from_yaml:
        if 'scroller' in scroll_control and '</div>' in scroll_control:
            if 'scroll up' in scroll_control:
                return 'Scroll up'
            else:
                return 'Scroll down'
        else:
            return False
    else: # from txt file
        if '</scrollbar>' in scroll_control:
            return True
        else:
            return False

def get_button_properties(state_desc, choice_idx):
    views = state_desc.split('>\n')
    for view in views:
        pattern = r'id=(\d+)'

        match = re.findall(pattern, view)
        if int(match[0]) == choice_idx:

            type_pattern = r"<(.*?)\s"
            type = re.findall(type_pattern, view)[0]

            desc_pattern = r"text=(.*?)\s"
            match = re.search(desc_pattern, view)
            if match:
                desc = match.group(1)
            else:
                desc_pattern = r'text=(.*?)>'
                match = re.search(desc_pattern, view)
                if match:
                    desc = match.group(1)
                else:
                    desc = ''

            text_pattern = r">(.*?)<\/"
            match = re.search(text_pattern, view)
            if match:
                text = match.group(1)
            else:
                text = ''

            return {'type': type, 'desc': desc, 'text': text}
    return 'missed'

def get_onclick_comment(event: str, jumped_state_id: int, former_comment='', state_desc='state'):
    match = re.search(r'InputText: (.*)', event)
    if match:
        input_text = match.group(1)
        comment = f' // on set_text (\'{input_text}\'), go to {state_desc} {jumped_state_id}' if former_comment == '' else former_comment + f'; on set_text (\'{input_text}\'), go to {state_desc} {jumped_state_id}'
    else:
        if 'scroll down' in event.lower():
            comment = f' // on scroll down, go to {state_desc} {jumped_state_id}'
        elif 'scroll up' in event.lower():
            comment = f' // on scroll up, go to {state_desc} {jumped_state_id}'
        elif 'Long_touch' in event:
            comment = f' // on long touch, go to {state_desc} {jumped_state_id}'
        elif 'Select' in event:
            comment = f' // on select, go to {state_desc} {jumped_state_id}'
        elif 'Unselect' in event:
            comment = f' // on unselect, go to {state_desc} {jumped_state_id}'
        else:
            comment = f' // on touch, go to {state_desc} {jumped_state_id}' if former_comment == '' else former_comment + f'; on touch, go to {state_desc} {jumped_state_id}'
    return comment

def get_tasks_from_json(jsonpath):
    data = load_json_file(jsonpath)
    def get_task_from_string(text):
        pattern = r"task: (.*?),"
        matches = re.findall(pattern, text)
        return matches

    all_tasks = []
    for k, v in data.items():
        if isinstance(v, list):
            all_tasks += [task_desc['task'] for task_desc in v]
            continue
        else:
            all_tasks += get_task_from_string(v)
    return all_tasks

def get_path_to_state(graph, start_str, end_str):
    '''
    get the UI nodes and the action sequence that can lead to the state
    '''
    if nx.has_path(graph, start_str, end_str):
        path_nodes = nx.shortest_path(graph, start_str, end_str)
        action_nodes = [graph[path_nodes[i]][path_nodes[i+1]]['event'][0] for i in range(len(path_nodes) - 1)]#list(self.raw_memory_graph.edges(path_nodes, data=True))
        return path_nodes, action_nodes
    else:
        return None, None
    
def make_date_dir(parent_path='output'):
    from datetime import datetime
    current_date = str(datetime.now().date())
    import os
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    directory = os.path.join(parent_path, current_date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def add_edge_from_yaml(records, page_state_str, pageid, utg, event, page_choice_id):
    # if pageid < len(records) - 1:
    next_page_str = records[pageid + 1]['state_str']
    if utg.has_edge(page_state_str, next_page_str):
        events = utg[page_state_str][next_page_str]['event']
        if [event, page_choice_id, page_state_str] not in events:
            events.append([event, page_choice_id, page_state_str])  # len(pure_elements) - 1 indicates the id of the current element
        utg.remove_edge(page_state_str, next_page_str)
    else:
        events = [[event, page_choice_id, page_state_str]]
    utg.add_edge(page_state_str, next_page_str, event=events, label=str(events))
    return utg

def remove_restart_goback(state_desc):
    '''
    can deal with either string state_desc or a list of views
    '''
    if isinstance(state_desc, str):
        views = split_state(state_desc)
    else:
        views = state_desc
    new_views = []
    for view in views:
        if 'restart</button>' not in view and 'go back' not in view:
            new_views.append(view)
    if isinstance(state_desc, list):
        return new_views
    if isinstance(state_desc, str):
        new_state_desc = '\n'.join(new_views)
        return new_state_desc

def remove_restart_revise_goback(state_desc, height=2280, width=1080):
    '''
    remove restart button, revise goback button
    '''
    if isinstance(state_desc, str):
        views = split_state(state_desc)
    else:
        views = state_desc
    new_views = []
    for view in views:
        if 'restart</button>' not in view:
            if 'go back' not in view:
                new_views.append(view)
            else:
                new_views.append(view.replace('bound_box=0,0,0,0', f'bound_box=0,{height-48},{width/3},{height}'))
    if isinstance(state_desc, list):
        return new_views
    if isinstance(state_desc, str):
        new_state_desc = '\n'.join(new_views)
        return new_state_desc

def convert_gpt_answer_to_json(answer, model_name):
    convert_prompt = f'''
Convert the following data into JSON format, ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).

data:
{answer}

**Please do not output any content other than the JSON format.**
'''
    try:
        converted_answer = json.loads(answer)
    except:
        print('*'*10, 'converting', '*'*10, '\n', answer, '\n', '*'*50)
        converted_answer = query_gptv2(convert_prompt, model_name)
        print('*'*10, 'converted v1', '*'*10, '\n', converted_answer, '\n', '*'*10)
        if isinstance(converted_answer, str):
            try:
                converted_answer = json.loads(converted_answer)
            except:
                new_convert = f'''
Convert the following data into JSON format, ensuring it's valid for Python parsing (pay attention to single/double quotes in the strings).

data:
{answer}

The former answer you returned:
{converted_answer}
is wrong and can not be parsed in python. Please check it and convert it properly!

**Please do not output any content other than the JSON format!!!**
'''
                converted_answer = query_gptv2(new_convert, model_name)
                print('*'*10, 'converted v2', '*'*10, '\n', converted_answer, '\n', '*'*10)
                if isinstance(converted_answer, str):
                    converted_answer = json.loads(converted_answer)
    return converted_answer

def get_int_from_str(input_string):
    # Use regular expression to find digits in the string
    match = re.search(r'\d+', input_string)

    if match:
        # Extract the matched integer
        extracted_integer = int(match.group())
        # print("Extracted integer:", extracted_integer)
    else:
        pdb.set_trace()
    return extracted_integer

if __name__ == '__main__':
    with open('prompts/task_query.txt', 'r') as file:
        prompt = file.read()
    print(prompt)
    print(query_gptv2(prompt, model_name='gpt-4-32k'))
