import copy
import datetime
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

import yaml
from rich import print
from rich.console import Console
from .utils import queryGPT
import tiktoken

CONSOLE = Console()


class Evaluator:
    def __init__(
        self,
        data_folder,
        log_root,
        memory_folder,
        use_memory,
        prompt_config,
        use_baseline=False,
        all_internal=False,
        external_mem=None,
        instance_name=None,
        specify_app=None,
        load_ckpt=False,
        load_result=False,
        disable_query=False,
    ) -> None:
        self.data_folder = data_folder
        self.log_root = log_root
        self.ckpt_folder = None
        self.memory_folder = memory_folder
        self.log_folder = None

        self.use_memory = use_memory
        self.use_baseline = use_baseline
        self.instance_name = instance_name
        self.load_ckpt = load_ckpt
        self.load_result = load_result
        self.external_mem = external_mem
        self.all_internal = all_internal
        self.disable_query = disable_query

        self.__cfg_structure = {}
        self.__profile_tree = {}
        self.__memory = {}
        self.__external_mem = {}
        self.__tokens = 0
        self.__prompts_data = {}

        init_data = self.to_dict()
        if self.instance_name is None:
            self.instance_name = hashlib.sha224(
                str(init_data).encode("utf-8")
            ).hexdigest()
        self.log_folder = Path(self.log_root) / self.instance_name
        if self.log_folder.exists():
            if self.load_ckpt:
                for txt in self.log_folder.glob("*.txt"):
                    txt.unlink()
            else:
                shutil.rmtree(self.log_folder)
        self.ckpt_folder = self.log_folder / "checkpoints"
        if not self.load_ckpt:
            self.log_folder.mkdir(parents=True, exist_ok=True)
            self.ckpt_folder.mkdir(parents=True, exist_ok=True)
            with open(self.log_folder / "init.json", "w") as f:
                json.dump(init_data, f)
        tmp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        (self.log_folder / f"{tmp}.txt").touch()
        if self.external_mem:
            with open(self.external_mem, "r") as f:
                self.__external_mem = json.load(f)
        with open(prompt_config, "r") as f:
            self.prompt_config = yaml.safe_load(f)
        for entry in Path(self.data_folder).iterdir():
            if entry.is_dir():
                self.__cfg_structure[entry.stem] = {}
        if self.use_memory:
            self.__memory = self.__load_memory()
        for app in self.__cfg_structure:
            if specify_app and app not in specify_app:
                continue
            self.__cfg_structure[app] = list(
                (Path(self.data_folder) / app).glob("*.yaml")
            )
            self.__profile_tree[app] = {}
            for cfg in self.__cfg_structure[app]:
                self.__profile_tree[app].update(self.__unitProcess(cfg))
        with open(self.log_folder / "profile_no_answer.json", "w") as f:
            json.dump(self.__profile_tree, f)
        if self.load_ckpt:
            with open(self.ckpt_folder / "prompts_data.json", "r") as f:
                tmp = json.load(f)
            with open(self.log_folder / "profile_no_answer.json", "r") as f:
                self.__profile_tree = json.load(f)
            for app in tmp.keys():
                if specify_app and app not in specify_app:
                    continue
                else:
                    self.__prompts_data[app] = tmp[app]
        return

    def __unitProcess(self, task_profile: Path) -> dict:
        with open(task_profile, "r") as f:
            task_data = yaml.safe_load(f)
        app = task_profile.parent.name
        if self.use_memory:
            task_data = self.__insert_memory(task_profile.parent.stem, task_data)
        name = task_data["task_name"]
        query_data = {"task": name, "profile": []}
        task_hash = hashlib.sha256(name.encode()).hexdigest()
        action_history = []
        for record in task_data["records"]:
            if not action_history:
                ah = f"\n- Start the {app} app."
            else:
                ah = "".join(action_history)
            temp = {
                "label": (record["Choice"], record["Input"]),
                "history": ah,
                "state": record["State"],
            }
            if record["Choice"] != -1:
                elements = re.findall(
                    "<[^<]*>[^<]*</[^>]*>", record["State"].replace("<br>", "@@@")
                )
                for e in elements:
                    if f"id={record['Choice']}" in e:
                        choice = re.sub(f"\s?id={record['Choice']}", "", e)
                        break
                choice = re.sub(" onclick=(.*?)>", ">", choice)
                if "checkbox" in choice:
                    if "checked=True" in choice:
                        new_action = f"\n- Uncheck: {choice.replace('@@@', '<br>')}"
                    elif "checked=False" in choice:
                        new_action = f"\n- Check: {choice.replace('@@@', '<br>')}"
                else:
                    new_action = f"\n- TapOn: {choice.replace('@@@', '<br>')}"
                if record["Input"] != "null":
                    new_action += f" InputText: {record['Input']}"
                new_action = new_action.replace("checked=True", "").replace(
                    "checked=False", ""
                )
                action_history.append(new_action)
            query_data["profile"].append(temp)
        query_data["step_num"] = len(query_data["profile"])
        return {task_hash: query_data}

    def __load_memory(self) -> dict:
        memory_cfgs = list(Path(self.memory_folder).glob("*.yaml"))
        available_apps = {e.stem: e for e in memory_cfgs}
        app_found = 0
        app_memory = {}
        for app in self.__cfg_structure:
            if app not in available_apps:
                continue
            with open(available_apps[app], "r") as f:
                memory = yaml.safe_load(f)
            app_memory[app] = memory
            app_found += 1
        if not app_found:
            CONSOLE.log(
                "You've specified to [cyan bold]load memory[/cyan bold]. But [red bold]no available app memory[/red bold] is found."
            )
        return app_memory

    def __insert_memory(self, app: str, task_data) -> dict:
        if app not in self.__memory:
            return task_data
        name = task_data["task_name"]
        task_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()
        related_elements = re.findall(
            "<[^<]*>[^<]*</[^>]*>",
            self.__external_mem[app][task_hash]["path"].replace("<br>", "@@@"),
        )
        for idx, rr in enumerate(related_elements):
            beg = re.findall("<([^>]*)>", rr)[0]
            sep = rr[rr.find(">") :]
            related_elements[idx] = "<" + beg.replace(" ", "[^>]*") + "[^>]*" + sep
        for idr, record in enumerate(task_data["records"]):
            if isinstance(record["new_state_str"], list):
                if len(record["new_state_str"]) == 1:
                    hash = record["new_state_str"][0]
                elif len(record["new_state_str"]) > 1:
                    hash = hashlib.sha256(record["State"].encode("utf-8")).hexdigest()
            elif isinstance(record["new_state_str"], str):
                hash = record["new_state_str"]
            elements = re.findall(
                "<[^<]*>[^<]*</[^>]*>", record["State"].replace("<br>", "@@@")
            )
            if hash not in self.__memory[app]:
                continue
            for ele_id, ele_hint in self.__memory[app][hash].items():
                leave = 1 - int(self.all_internal)
                if idr < len(related_elements) - 1 and not self.all_internal:
                    for idd in range(idr + 1):
                        rrre = related_elements[idd]
                        if re.match(rrre, elements[ele_id]):
                            leave = 0
                            break
                if not leave:
                    sub_ele = elements[ele_id].split(">")
                    sub_ele[0] += f" onclick=jump to a GUI about '{ele_hint}'"
                    elements[ele_id] = ">".join(sub_ele)
            task_data["records"][idr]["State"] = ("\n".join(elements)).replace(
                "@@@", "<br>"
            )
        return task_data

    def __insert_external_task(self, app: str, task_data):
        name = task_data["task_name"]
        task_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()
        related_elements = re.findall(
            "<[^<]*>[^<]*</[^>]*>",
            self.__external_mem[app][task_hash]["path"].replace("<br>", "@@@"),
        )
        similar_task = self.__external_mem[app][task_hash]["similar"]
        similar_task = re.sub("[tT]ask: ", "", similar_task)
        for idx, rr in enumerate(related_elements):
            beg = re.findall("<([^>]*)>", rr)[0]
            sep = rr[rr.find(">") :]
            related_elements[idx] = "<" + beg.replace(" ", "[^>]*") + "[^>]*" + sep
        for idr, record in enumerate(task_data["records"]):
            elements = re.findall(
                "<[^<]*>[^<]*</[^>]*>", record["State"].replace("<br>", "@@@")
            )
            for ide, ele in enumerate(elements):
                if idr < len(related_elements) and re.match(related_elements[idr], ele):
                    for idd in range(idr + 1):
                        rrre = related_elements[idd]
                        if re.match(rrre, ele):
                            sub_ele = ele.split(">")
                            sub_ele[
                                0
                            ] += f' onclick="finish the task of {similar_task}"'
                            elements[ide] = ">".join(sub_ele)
            task_data["records"][idr]["State"] = ("\n".join(elements)).replace(
                "@@@", "<br>"
            )
        return task_data

    def __fetchAnswers(self):
        CONSOLE.rule("START PROCESS")
        enc = tiktoken.get_encoding("cl100k_base")
        mem = {}
        if self.external_mem:
            with open(self.external_mem, "r") as f:
                mem = json.load(f)
        for app in self.__profile_tree:
            self.__prompts_data[app] = {}
            result = {}
            if app in mem:
                app_mem = mem[app]
            else:
                app_mem = {}
            CONSOLE.log(f"query for {app}", style="red bold")
            with ThreadPoolExecutor() as executor:
                futures_to_data = {}
                for task_hash in self.__profile_tree[app]:
                    name = self.__profile_tree[app][task_hash]["task"]
                    self.__prompts_data[app][task_hash] = []
                    for idx, profile in enumerate(
                        self.__profile_tree[app][task_hash]["profile"]
                    ):
                        try:
                            input_element = re.findall(
                                f"<[^<]+id={profile['label'][0]}.*?>.*?</.*?>",
                                profile["state"],
                            )[0]
                        except:
                            input_element = None
                        if self.use_baseline:
                            prompt = self.__parse_prompts_baseline(
                                interface=profile["state"], 
                                task_name=name,
                                history=profile['history']
                            )
                        else:
                            prompt = self.__parse_prompts_new(
                                task_name=name,
                                history=profile["history"],
                                interface=profile["state"],
                                input=(profile["label"][1] != "null"),
                                input_element=input_element,
                                external_mem={},
                            )
                        self.__prompts_data[app][task_hash].append(prompt)
                        if not self.disable_query:
                            futures_to_data[
                                executor.submit(
                                    queryGPT,
                                    prompt=prompt,
                                    console=CONSOLE,
                                    identifier=f"{task_hash}, step[{idx}]",
                                )
                            ] = (task_hash, idx)
                            self.__tokens += len(enc.encode(prompt))
                if not self.disable_query:
                    for future in as_completed(futures_to_data):
                        task_hash, idx = futures_to_data[future]
                        result[str((task_hash, idx))] = future.result()
                    with open(self.ckpt_folder / f"result_{app}.json", "w") as f:
                        json.dump(result, f)
                with open(self.ckpt_folder / "prompts_data.json", "w") as f:
                    json.dump(self.__prompts_data, f)
            CONSOLE.rule(f"ALL DONE; Tokens: {self.__tokens}")
        return

    def __parse_prompts(self, **kwargs) -> str:
        intro = self.prompt_config["introduction"]
        task = self.prompt_config["task"] + kwargs["task_name"] + "\n"
        history = self.prompt_config["history"] + kwargs["history"] + "\n"
        interface = self.prompt_config["interface"] + kwargs["interface"] + "\n"
        question_prefix = self.prompt_config["question"]
        if not kwargs["input_element"]:
            input_prompt = ""
        else:
            input_prompt = self.prompt_config["input"].replace(
                "[SOE]input_elements[EOE]", kwargs["input_element"]
            )
        element_prompt = self.prompt_config["element"]
        return (
            intro
            + task
            + interface
            + history
            + question_prefix
            + (input_prompt if kwargs["input"] else element_prompt)
        )

    def __parse_prompts_baseline(self, **kwargs) -> str:
        intro = self.prompt_config["introduction"]
        interface = self.prompt_config["interface"] + kwargs["interface"]
        instruction = self.prompt_config["task"] + kwargs["task_name"]
        history = self.prompt_config["history"] + kwargs["history"]
        question = self.prompt_config["question"]
        return intro + interface + instruction + history + question

    def __parse_prompts_new(self, **kwargs) -> str:
        intro = self.prompt_config["introduction"]
        task = self.prompt_config["task"] + kwargs["task_name"]
        history = self.prompt_config["history"] + kwargs["history"]
        interface = self.prompt_config["interface"] + kwargs["interface"]
        question = self.prompt_config["question"]
        task_hash = hashlib.sha256(kwargs["task_name"].encode("utf-8")).hexdigest()
        if kwargs["external_mem"]:
            mem = (
                self.prompt_config["external_mem"]
                + kwargs["external_mem"][task_hash]["path"]
            )
        else:
            mem = ""
        return intro + task + history + interface + mem + question

    def evaluatev1(self):
        result_analysis = self.__profile_tree
        if not self.load_ckpt:
            self.__fetchAnswers()
            if self.disable_query:
                return
        if not self.load_result:
            results_jsons = list(self.ckpt_folder.glob("result_*.json"))
            for results_file in results_jsons:
                app = re.findall("result_(.*?).json", str(results_file))[0]
                if app not in self.__profile_tree:
                    continue
                with open(results_file, "r") as f:
                    results = json.load(f)
                for k, v in results.items():
                    try:
                        task_hash, idx = eval(k)
                    except:
                        task_hash = k.split("-")[0]
                        idx = int(k.split("-")[1])
                    if "correct" not in result_analysis[app][task_hash]:
                        result_analysis[app][task_hash]["correct"] = 0
                    if "end_correct" not in result_analysis[app][task_hash]:
                        result_analysis[app][task_hash]["end_correct"] = False
                    label = result_analysis[app][task_hash]["profile"][idx]["label"]
                    correct = False
                    gt_id = label[0]
                    gt_input = label[1]
                    llm_id = "N/A"
                    llm_action = "N/A"
                    llm_input = "N/A"
                    if v is None:
                        llm_id = -1
                    elif self.use_baseline:
                        try:
                            llm_id = re.findall("ID:\s?(N/A|-?\d+)", v)[0]
                            if llm_id == "N/A":
                                llm_id = -1
                            else:
                                llm_id = int(llm_id)
                        except:
                            llm_id = int(input(v + "\nPlease input id: "))
                        if llm_id == -1:
                            llm_input =  "N/A"
                        else:
                            try:
                                llm_input = re.findall("Input:\s?(.*)", v)[0]
                            except:
                                llm_input = input(v + "\nPlease input text: ")
                                try:
                                    if int(llm_input) == -1:
                                        llm_input = "N/A"
                                except:
                                    pass
                        llm_action = None
                        if gt_input == "null" and llm_id == gt_id:
                            correct = True
                            result_analysis[app][task_hash]["correct"] += int(correct)
                        if gt_input != "null" and llm_id == gt_id:
                            correct = gt_input == llm_input
                            result_analysis[app][task_hash]["correct"] += int(correct)
                    else:
                        try:
                            whether_finished_answer = re.findall(
                                "3\.(.*)4\.", v, flags=re.S
                            )[0]
                            for e in ["Yes.", "Y.", "y.", "yes.", "is already finished"]:
                                if re.match(f"\b{e}\b", whether_finished_answer):
                                    llm_id = -1
                                    llm_action = "N/A"
                                    llm_input = "N/A"
                                    break
                        except:
                            pass
                        try:
                            finished_check = re.findall("4\.(.*)", v, flags=re.S)[0]
                            for e in [
                                "No further interaction is required",
                                "cannot be determined based on",
                                "no further action is needed",
                            ]:
                                if e in finished_check:
                                    llm_id = -1
                                    llm_action = "N/A"
                                    llm_input = "N/A"
                        except:
                            pass
                        if "id should be -1" in v:
                            llm_id = -1
                        if "no further interaction is" in v:
                            llm_id = -1
                        if llm_id != -1:
                            try:
                                llm_id, llm_action, llm_input = re.findall(
                                    "id=(N/A|-?\d+)(?:.|\\n)*(?:-|,)\s?action=(N/A|\w+)(?:.|\\n)*(?:-|,)\s?input text=\"?'?(N/A|\w+)\"?'?",
                                    v,
                                )[0]
                                if llm_id == "N/A":
                                    llm_id = -1
                                else:
                                    llm_id = int(llm_id)
                                if "tapon" in llm_action.lower():
                                    llm_action = "tap"
                                elif "none" in llm_action.lower():
                                    llm_action = "N/A"
                                elif "click" in llm_action.lower():
                                    llm_action = "tap"
                                elif "input" in llm_action.lower():
                                    llm_action = "input"
                                assert llm_action in ["tap", "input", "N/A"]
                            except:
                                try:
                                    llm_id, llm_action = re.findall(
                                        "id=(N/A|-?\d+)(?:.|\\n)*(?:-|,)\s?action=(N/A|\w+)", v.lower(), flags=re.S
                                    )[0]
                                    llm_id = int(llm_id)
                                    if (
                                        "tapon" in llm_action.lower()
                                        or "check" in llm_action.lower()
                                        or "uncheck" in llm_action.lower()
                                    ):
                                        llm_action = "tap"
                                    elif "none" in llm_action.lower():
                                        llm_action = "N/A"
                                    assert llm_action in ["tap", "input", "N/A"]
                                except:
                                    try:
                                        llm_id, llm_action, llm_input = re.findall(
                                            "Action: (N/A|\w+)\\nid=(-?\d+)\\ninput text=(N/A|\w+)", v
                                        )[0]
                                        llm_id = int(llm_id)
                                    except:
                                        llm_id, llm_action, llm_input = eval(
                                            input(
                                                v + "\nPlease input id, action, and text: "
                                            )
                                        )
                                        llm_id = int(llm_id)
                                        llm_action = ["tap", "input", "N/A"][
                                            int(llm_action)
                                        ]
                                        try:
                                            if int(llm_input) == -1:
                                                llm_input = "N/A"
                                        except:
                                            pass
                        if gt_id == -1:
                            correct = llm_id == gt_id
                            if correct:
                                result_analysis[app][task_hash]["end_correct"] = True
                        elif gt_input == "null" and llm_action != "input":
                            correct = llm_id == gt_id
                        elif (
                            gt_input != "null"
                            and llm_action == "input"
                            and llm_id == gt_id
                        ):
                            correct = llm_input == gt_input
                        result_analysis[app][task_hash]["correct"] += int(correct)
                    result_analysis[app][task_hash]["profile"][idx]["answer"] = (
                        llm_id,
                        llm_action,
                        llm_input,
                    )
                    result_analysis[app][task_hash]["profile"][idx]["correct"] = correct
                    result_analysis[app][task_hash]["profile"][idx]["raw"] = v
                    result_analysis[app][task_hash]["profile"][idx][
                        "prompt"
                    ] = self.__prompts_data[app][task_hash][idx]
            with open(self.log_folder / "result_analysis.json", "w") as f:
                json.dump(result_analysis, f)
        else:
            with open(self.log_folder / "result_analysis.json", "r") as f:
                result_analysis = json.load(f)
        self.__showResults(result_analysis)

    def evaluate(self):
        result_analysis = self.__profile_tree
        if not self.load_ckpt:
            self.__fetchAnswers()
            if self.disable_query:
                return
        if not self.load_result:
            results_jsons = list(self.ckpt_folder.glob("result_*.json"))
            for results_file in results_jsons:
                app = re.findall("result_(.*?).json", str(results_file))[0]
                if app not in self.__profile_tree:
                    continue
                with open(results_file, "r") as f:
                    results = json.load(f)
                for k, v in results.items():
                    try:
                        task_hash, idx = eval(k)
                    except:
                        task_hash = k.split("-")[0]
                        idx = int(k.split("-")[1])
                    if "correct" not in result_analysis[app][task_hash]:
                        result_analysis[app][task_hash]["correct"] = 0
                    if "end_correct" not in result_analysis[app][task_hash]:
                        result_analysis[app][task_hash]["end_correct"] = False
                    label = result_analysis[app][task_hash]["profile"][idx]["label"]
                    correct = False
                    gt_id = label[0]
                    gt_input = label[1]
                    llm_id = "N/A"
                    llm_action = "N/A"
                    llm_input = "N/A"
                    if v is None:
                        llm_id = -1
                    elif self.use_baseline:
                        try:
                            llm_id = re.findall("ID:\s?(N/A|-?\d+)", v)[0]
                            if llm_id == "N/A":
                                llm_id = -1
                            else:
                                llm_id = int(llm_id)
                        except:
                            llm_id = int(input(v + "\nPlease input id: "))
                        if llm_id == -1:
                            llm_input =  "N/A"
                        else:
                            try:
                                llm_input = re.findall("Input:\s?(.*)", v)[0]
                            except:
                                llm_input = input(v + "\nPlease input text: ")
                                try:
                                    if int(llm_input) == -1:
                                        llm_input = "N/A"
                                except:
                                    pass
                        llm_action = None
                        if gt_input == "null" and llm_id == gt_id:
                            correct = True
                            result_analysis[app][task_hash]["correct"] += int(correct)
                        if gt_input != "null" and llm_id == gt_id:
                            correct = gt_input == llm_input
                            result_analysis[app][task_hash]["correct"] += int(correct)
                    else:
                        try:
                            if isinstance(v, str):
                                v = json.loads(v)
                        except:
                            print('format error: v')
                            llm_id = -1
                            llm_action = "N/A"
                            llm_input = "N/A"
                        try:
                            if 'Finished' in v.keys():
                                whether_finished_answer = v['Finished'].lower() == 'yes' or v['Finished'].lower() == 'y' or v['Finished'].lower() == 'true' or 'finished' in v['Finished'].lower() 
                            elif 'finished' in v.keys():
                                whether_finished_answer = v['finished'].lower() == 'yes' or v['finished'].lower() == 'y' or v['finished'].lower() == 'true' or 'finished' in v['finished'].lower() 
                            else:
                                whether_finished_answer = False
                            if whether_finished_answer:
                                llm_id = -1
                                llm_action = "N/A"
                                llm_input = "N/A"
                            else:
                                llm_id = 'N/A'

                        except:
                            pass
                        if llm_id != -1:
                            # if 'Next step' in v.keys():
                            #     step_desc = v['Next step']
                            # elif 'next step' in v.keys():
                            #     step_desc = v['next step']
                            
                            # elif 'next_step' in v.keys():
                            #     step_desc = v['next_step']
                            # else:
                            #     print('next step not found')
                            #     llm_id = -1
                            step_desc = v
                            try:
                                llm_id = step_desc['id']
                                llm_action = step_desc['action']
                                llm_input = step_desc['input_text']
                                if llm_id == "N/A":
                                    llm_id = -1
                                else:
                                    llm_id = int(llm_id)
                                if "tap" in llm_action.lower() or "check" in llm_action.lower() or "choose" in llm_action.lower():
                                    llm_action = "tap"
                                elif "none" in llm_action.lower():
                                    llm_action = "N/A"
                                elif "click" in llm_action.lower():
                                    llm_action = "tap"
                                elif "input" in llm_action.lower():
                                    llm_action = "input"
                                assert llm_action in ["tap", "input", "N/A"]
                            except:
                                llm_id = -1
                                # try:
                                #     llm_id, llm_action = re.findall(
                                #         "id=(N/A|-?\d+)(?:.|\\n)*(?:-|,)\s?action=(N/A|\w+)", v.lower(), flags=re.S
                                #     )[0]
                                #     llm_id = int(llm_id)
                                #     if (
                                #         "tapon" in llm_action.lower()
                                #         or "check" in llm_action.lower()
                                #         or "uncheck" in llm_action.lower()
                                #     ):
                                #         llm_action = "tap"
                                #     elif "none" in llm_action.lower():
                                #         llm_action = "N/A"
                                #     assert llm_action in ["tap", "input", "N/A"]
                                # except:
                                #     try:
                                #         llm_id, llm_action, llm_input = re.findall(
                                #             "Action: (N/A|\w+)\\nid=(-?\d+)\\ninput text=(N/A|\w+)", v
                                #         )[0]
                                #         llm_id = int(llm_id)
                                #     except:
                                #         llm_id, llm_action, llm_input = eval(
                                #             input(
                                #                 v + "\nPlease input id, action, and text: "
                                #             )
                                #         )
                                #         llm_id = int(llm_id)
                                #         llm_action = ["tap", "input", "N/A"][
                                #             int(llm_action)
                                #         ]
                                #         try:
                                #             if int(llm_input) == -1:
                                #                 llm_input = "N/A"
                                #         except:
                                #             pass
                        if gt_id == -1:
                            correct = llm_id == gt_id
                            if correct:
                                result_analysis[app][task_hash]["end_correct"] = True
                        elif gt_input == "null" and llm_action != "input":
                            correct = llm_id == gt_id
                        elif (
                            gt_input != "null"
                            and llm_action == "input"
                            and llm_id == gt_id
                        ):
                            correct = llm_input == gt_input
                        result_analysis[app][task_hash]["correct"] += int(correct)
                    result_analysis[app][task_hash]["profile"][idx]["answer"] = (
                        llm_id,
                        llm_action,
                        llm_input,
                    )
                    result_analysis[app][task_hash]["profile"][idx]["correct"] = correct
                    result_analysis[app][task_hash]["profile"][idx]["raw"] = v
                    result_analysis[app][task_hash]["profile"][idx][
                        "prompt"
                    ] = self.__prompts_data[app][task_hash][idx]
            with open(self.log_folder / "result_analysis.json", "w") as f:
                json.dump(result_analysis, f)
        else:
            with open(self.log_folder / "result_analysis.json", "r") as f:
                result_analysis = json.load(f)
        return self.__showResults(result_analysis)

    def __showResults(self, result_analysis):
        results = {}
        ss = 0
        cc = 0
        wc = 0
        cnt = 0
        pwc = 0
        for app in result_analysis:
            if app not in self.__cfg_structure or app not in self.__profile_tree:
                continue
            s = 0
            c = 0
            e = 0
            for task_hash in result_analysis[app]:
                step_num = result_analysis[app][task_hash]["step_num"]
                correct = result_analysis[app][task_hash]["correct"]
                end_correct = result_analysis[app][task_hash]["end_correct"]
                s += step_num
                c += correct
                e += int(end_correct)
                wc += int(step_num == correct)
                pwc += int(step_num == correct + 1) and not end_correct
                cnt += 1
            results[app] = [
                (round(c / s, 3), round(e / len(result_analysis[app]), 3)),
                (f"{c}/{s}", f"{e}/{len(result_analysis[app])}"),
            ]
            ss += s
            cc += c
        results["Acc"] = (round(cc / ss, 3), f"{cc}/{ss}")
        results["Whole_task_Acc"] = (round(wc / cnt, 3), f"{wc}/{cnt}")
        results["Whole_task_Acc_without_end"] = (round((pwc+wc) / cnt, 3), f"{pwc+wc}/{cnt}")
        results["Tokens"] = self.__tokens
        print(results)
        return results

    def to_dict(self) -> dict:
        output = {}
        for k, v in copy.deepcopy(self.__dict__).items():
            if k.startswith(f"_{self.__class__.__name__}"):
                continue
            output[k] = v
        return output

    @classmethod
    def from_dict(cls, config_dict):
        evaluator = cls(**config_dict)
        return evaluator

    def __repr__(self) -> str:
        return str(self.__cfg_structure)
    
    def get_acc_by_types(self):
        tap_num, input_num, end_num = 0, 0, 0
        tap_right_num, input_right_num, end_right_num = 0, 0, 0
        result_analysis = self.__profile_tree
        if not self.load_result:
            results_jsons = list(self.ckpt_folder.glob("result_*.json"))
            for results_file in results_jsons:
                app = re.findall("result_(.*?).json", str(results_file))[0]
                if app not in self.__profile_tree:
                    continue
                with open(results_file, "r") as f:
                    results = json.load(f)
                for k, v in results.items():
                    try:
                        task_hash, idx = eval(k)
                    except:
                        task_hash = k.split("-")[0]
                        idx = int(k.split("-")[1])
                    if "correct" not in result_analysis[app][task_hash]:
                        result_analysis[app][task_hash]["correct"] = 0
                    if "end_correct" not in result_analysis[app][task_hash]:
                        result_analysis[app][task_hash]["end_correct"] = False
                    label = result_analysis[app][task_hash]["profile"][idx]["label"]
                    correct = False
                    gt_id = label[0]
                    gt_input = label[1]
                    llm_id = "N/A"
                    llm_action = "N/A"
                    llm_input = "N/A"
                    if v is None:
                        llm_id = -1
                    elif self.use_baseline:
                        try:
                            llm_id = re.findall("ID:\s?(N/A|-?\d+)", v)[0]
                            if llm_id == "N/A":
                                llm_id = -1
                            else:
                                llm_id = int(llm_id)
                        except:
                            llm_id = int(input(v + "\nPlease input id: "))
                        if llm_id == -1:
                            llm_input =  "N/A"
                        else:
                            try:
                                llm_input = re.findall("Input:\s?(.*)", v)[0]
                            except:
                                llm_input = input(v + "\nPlease input text: ")
                                try:
                                    if int(llm_input) == -1:
                                        llm_input = "N/A"
                                except:
                                    pass
                        llm_action = None
                        if gt_input == "null" and llm_id == gt_id:
                            correct = True
                            result_analysis[app][task_hash]["correct"] += int(correct)
                        if gt_input != "null" and llm_id == gt_id:
                            correct = gt_input == llm_input
                            result_analysis[app][task_hash]["correct"] += int(correct)
                    else:
                        try:
                            if isinstance(v, str):
                                v = json.loads(v)
                        except:
                            print('format error: v')
                            llm_id = -1
                            llm_action = "N/A"
                            llm_input = "N/A"
                        try:
                            if 'Finished' in v.keys():
                                whether_finished_answer = v['Finished'].lower() == 'yes' or v['Finished'].lower() == 'y' or v['Finished'].lower() == 'true' or 'finished' in v['Finished'].lower() 
                            elif 'finished' in v.keys():
                                whether_finished_answer = v['finished'].lower() == 'yes' or v['finished'].lower() == 'y' or v['finished'].lower() == 'true' or 'finished' in v['finished'].lower() 
                            else:
                                whether_finished_answer = False
                            if whether_finished_answer:
                                llm_id = -1
                                llm_action = "N/A"
                                llm_input = "N/A"
                            else:
                                llm_id = 'N/A'

                        except:
                            pass
                        if llm_id != -1:
                            # if 'Next step' in v.keys():
                            #     step_desc = v['Next step']
                            # elif 'next step' in v.keys():
                            #     step_desc = v['next step']
                            
                            # elif 'next_step' in v.keys():
                            #     step_desc = v['next_step']
                            # else:
                            #     print('next step not found')
                            #     llm_id = -1
                            step_desc = v
                            try:
                                llm_id = step_desc['id']
                                llm_action = step_desc['action']
                                llm_input = step_desc['input_text']
                                if llm_id == "N/A":
                                    llm_id = -1
                                else:
                                    llm_id = int(llm_id)
                                if "tap" in llm_action.lower() or "check" in llm_action.lower() or "choose" in llm_action.lower():
                                    llm_action = "tap"
                                elif "none" in llm_action.lower():
                                    llm_action = "N/A"
                                elif "click" in llm_action.lower():
                                    llm_action = "tap"
                                elif "input" in llm_action.lower():
                                    llm_action = "input"
                                assert llm_action in ["tap", "input", "N/A"]
                            except:
                                llm_id = -1
                                # try:
                                #     llm_id, llm_action = re.findall(
                                #         "id=(N/A|-?\d+)(?:.|\\n)*(?:-|,)\s?action=(N/A|\w+)", v.lower(), flags=re.S
                                #     )[0]
                                #     llm_id = int(llm_id)
                                #     if (
                                #         "tapon" in llm_action.lower()
                                #         or "check" in llm_action.lower()
                                #         or "uncheck" in llm_action.lower()
                                #     ):
                                #         llm_action = "tap"
                                #     elif "none" in llm_action.lower():
                                #         llm_action = "N/A"
                                #     assert llm_action in ["tap", "input", "N/A"]
                                # except:
                                #     try:
                                #         llm_id, llm_action, llm_input = re.findall(
                                #             "Action: (N/A|\w+)\\nid=(-?\d+)\\ninput text=(N/A|\w+)", v
                                #         )[0]
                                #         llm_id = int(llm_id)
                                #     except:
                                #         llm_id, llm_action, llm_input = eval(
                                #             input(
                                #                 v + "\nPlease input id, action, and text: "
                                #             )
                                #         )
                                #         llm_id = int(llm_id)
                                #         llm_action = ["tap", "input", "N/A"][
                                #             int(llm_action)
                                #         ]
                                #         try:
                                #             if int(llm_input) == -1:
                                #                 llm_input = "N/A"
                                #         except:
                                #             pass
                        if gt_id == -1:
                            correct = llm_id == gt_id
                            if correct:
                                result_analysis[app][task_hash]["end_correct"] = True
                                end_right_num += 1
                            end_num += 1
                        elif gt_input == "null" and llm_action != "input":
                            correct = llm_id == gt_id
                            tap_num += 1
                            if correct:
                                tap_right_num += 1
                        elif (
                            gt_input != "null"
                            and llm_action == "input"
                            and llm_id == gt_id
                        ):
                            correct = llm_input == gt_input
                            input_num += 1
                            if correct:
                                input_right_num += 1
                        result_analysis[app][task_hash]["correct"] += int(correct)
                    result_analysis[app][task_hash]["profile"][idx]["answer"] = (
                        llm_id,
                        llm_action,
                        llm_input,
                    )
                    result_analysis[app][task_hash]["profile"][idx]["correct"] = correct
                    result_analysis[app][task_hash]["profile"][idx]["raw"] = v
                    result_analysis[app][task_hash]["profile"][idx][
                        "prompt"
                    ] = self.__prompts_data[app][task_hash][idx]
            print(f"tap_acc: {tap_right_num/tap_num}, input_acc: {input_right_num/input_num}, end_acc: {end_right_num/end_num}")
            return {'tap_acc': tap_right_num/tap_num, 'input_acc': input_right_num/input_num, 'end_acc': end_right_num/end_num}
        # else:
        #     with open(self.log_folder / "result_analysis.json", "r") as f:
        #         result_analysis = json.load(f)
        # self.__showResults(result_analysis)
