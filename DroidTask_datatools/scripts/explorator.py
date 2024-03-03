import networkx as nx
from pathlib import Path
import yaml
from pyvis.network import Network
import re
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import queryGPT
from rich.console import Console
from InstructorEmbedding import INSTRUCTOR
# from langchain.embeddings import HuggingFaceInstructEmbeddings
import numpy as np
import hashlib
import pdb

CONSOLE = Console()


class Explorator:
    def __init__(
        self,
        data_folder,
        exploration_backup=None,
        specify_apps=None,
        summary_backup=None,
        states_backup=None,
        element_backup=None,
        output_path=None,
        backup_mode=False
    ):
        self.data_folder = data_folder
        self.specify_apps = []
        self.output_path = "ex_mem.json" if output_path is None else output_path
        self.backup_mode = backup_mode
        exps = Path(self.data_folder).glob("*.yaml")
        for exp in exps:
            app = exp.stem
            if specify_apps and app not in specify_apps:
                continue
            self.specify_apps.append(app)
        if exploration_backup:
            self.specify_apps.extend(list(exploration_backup.keys()))
        self.exploration_data = self.__buildGraph(exploration_backup)
        self.filtered_states = self.__filterStates(states_backup)
        self.app_state_summary = self.__stateSummary(summary_backup)
        self.element_summary = self.__elementSummary(element_backup)

    def __buildGraph(self, exploration_backup):
        exps = Path(self.data_folder).glob("*.yaml")
        result = {}
        leave_apps = []
        if exploration_backup:
            for app in exploration_backup:
                result[app] = exploration_backup[app]
                leave_apps.append(app)
        for exp_file in exps:
            app = exp_file.stem
            if app not in self.specify_apps:
                continue
            if app in leave_apps:
                continue
            with open(exp_file, "r") as f:
                exp_data = yaml.safe_load(f)
            records = exp_data["records"]
            graph = nx.DiGraph()
            root = records[0]["new_state_str"]
            for idx, record in enumerate(records):
                if idx == len(records) - 1:
                    break
                u = record["new_state_str"]
                v = records[idx + 1]["new_state_str"]
                u_elements = re.findall(
                    "<[^<]*>[^<]*</[^>]*>",
                    record["State"].replace("<br>", "@@@").replace("<unknown>", ""),
                )
                v_elements = re.findall(
                    "<[^<]*>[^<]*</[^>]*>",
                    records[idx + 1]["State"]
                    .replace("<br>", "@@@")
                    .replace("<unknown>", ""),
                )
                u_elements = list(map(lambda x: x.replace("@@@", "<br>"), u_elements))
                v_elements = list(map(lambda x: x.replace("@@@", "<br>"), v_elements))
                if not graph.has_node(u):
                    graph.add_node(
                        u,
                        label=str(idx),
                        title=str(record["State"]),
                        elements=u_elements,
                    )
                if not graph.has_node(v):
                    graph.add_node(
                        v,
                        label=str(idx + 1),
                        title=str(records[idx + 1]["State"]),
                        elements=v_elements,
                    )
                event = {"Choice": record["Choice"], "Input": record["Input"]}
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v, title=str([event]), events=[event])
                else:
                    if event not in graph.edges[u, v]["events"]:
                        ori_title = eval(graph.edges[u, v]["title"])
                        ori_title.append(event)
                        graph.edges[u, v]["title"] = str(ori_title)
                        graph.edges[u, v]["events"].append(event)
            result[app] = {"root": root, "graph": graph, "data": exp_data}
        return result

    def __filterStates(self, states_backup):
        if states_backup is not None:
            with open(states_backup, "r") as f:
                return json.load(f)
        result = {}
        for app in self.specify_apps:
            node_filtered_elements = {}
            graph: nx.DiGraph = self.exploration_data[app]["graph"]
            root = self.exploration_data[app]["root"]
            relate_path = nx.shortest_path(graph, source=root)
            exclusive = []
            for ele in graph.nodes[root]["elements"]:
                if (
                    re.match("<p\s?", ele)
                    or re.match("<input\s?", ele)
                    or "go back" in ele
                ):
                    exclusive.append(None)
                    continue
                exclusive.append(re.sub("\sid=\d+", "", ele))
            node_filtered_elements[root] = {
                "path": [],
                "elements": exclusive,
                "debug": graph.nodes[root]["elements"],
                "gpath": [],
            }
            for node in list(graph):
                nonroot_exclusive = copy.deepcopy(exclusive)
                if node == root:
                    continue
                else:
                    path = []
                    elements = []
                    nodes_on_path = relate_path[node]
                    gpath = []
                    for idn, n in enumerate(nodes_on_path):
                        if idn == len(nodes_on_path) - 1:
                            break
                        u = n
                        v = nodes_on_path[idn + 1]
                        ch = graph.edges[u, v]["events"][0]["Choice"]
                        gpath.append((u, ch))
                        for e in graph.nodes[u]["elements"]:
                            if f"id={ch}" in e:
                                path.append(re.sub("\sid=\d+", "", e))
                                break
                        if not idn:
                            continue
                        for ele in graph.nodes[u]["elements"]:
                            filter_ele = re.sub("\sid=\d+", "", ele)
                            if filter_ele not in nonroot_exclusive:
                                nonroot_exclusive.append(filter_ele)
                    for ele in graph.nodes[node]["elements"]:
                        filter_ele = re.sub("\sid=\d+", "", ele)
                        if (
                            (
                                filter_ele not in nonroot_exclusive
                                or "search" in filter_ele.lower()
                            )
                            and not re.match("<p\s?", filter_ele)
                            and not re.match("<input\s?", filter_ele)
                            and "go back" not in filter_ele
                        ):
                            elements.append(filter_ele)
                        else:
                            elements.append(None)
                node_filtered_elements[node] = {
                    "path": path,
                    "elements": elements,
                    "debug": graph.nodes[node]["elements"],
                    "gpath": gpath,
                }
            result[app] = node_filtered_elements
        if self.backup_mode:
            with open("configs/json/node_filtered_elements.json", "w") as f:
                json.dump(result, f)
        return result

    def __stateSummary(self, summary_backup):
        if summary_backup is not None:
            with open(summary_backup, "r") as f:
                return json.load(f)
        exps = Path(self.data_folder).glob("*.yaml")
        final_data = {}
        # for exp_file in exps:
        for app in self.specify_apps:
            # app = exp_file.stem
            # if app not in self.specify_apps:
            #     continue
            records = self.exploration_data[app]["data"]["records"]
            node_states = {}
            result = {}
            for record in records:
                if record["new_state_str"] not in node_states:
                    node_states[record["new_state_str"]] = record["State"]
            with ThreadPoolExecutor() as executor:
                CONSOLE.rule(f"START TO PROCESS {app.upper()}")
                futures_to_data = {}
                for state, screen in node_states.items():
                    prompt = f"You are an agent to help people use their smartphone apps. Below is the HTML interface of the app {app}.\n{screen}\nSummary the function of this interface in 10-15 words. Example: func: manage files."
                    futures_to_data[
                        executor.submit(
                            queryGPT, prompt=prompt, console=CONSOLE, identifier=state
                        )
                    ] = state
                for future in as_completed(futures_to_data):
                    state = futures_to_data[future]
                    result[state] = future.result()
            final_data[app] = result
        if self.backup_mode:
            with open("configs/json/app_state_summary.json", "w") as f:
                json.dump(final_data, f)
        return final_data

    def __elementSummary(self, element_backup):
        if element_backup:
            with open(element_backup, "r") as f:
                return json.load(f)
        result = {}
        # exps = Path(self.data_folder).glob("*.yaml")
        # for exp_file in exps:
        for app in self.specify_apps:
            # app = exp_file.stem
            # if app not in self.specify_apps:
            #     continue
            result[app] = {}
            CONSOLE.rule(f"START TO PROCESS {app.upper()}")
            with ThreadPoolExecutor() as executor:
                futures_to_data = {}
                for node, info in self.filtered_states[app].items():
                    path = ""
                    result[app][node] = [None] * len(info["elements"])
                    if info["path"]:
                        for p in info["path"]:
                            if "checkbox" in p and "checked=True" in p:
                                path += f"Uncheck {p}"
                            elif "checkbox" in p and "checked=False" in p:
                                path += f"Check {p}"
                            else:
                                path += f"Click on {p}"
                    screen_func = self.app_state_summary[app][node]
                    for ide, ele in enumerate(info["elements"]):
                        fresh_path = copy.deepcopy(path)
                        if ele is None:
                            continue
                        if "checkbox" in ele and "checked=True" in ele:
                            fresh_path += f"Uncheck {ele}"
                        elif "checkbox" in ele and "checked=False" in ele:
                            fresh_path += f"Check {ele}"
                        else:
                            fresh_path += f"Click on {ele}"
                        prompt = f"You are an agent to help people use smartphone apps. Given a chain of operations from the homepage of {app}, you need to figure out what task does the user wants to finish. Below is the chain of operations:\n{fresh_path}\nThe function of the current interface: {screen_func}. Predict the task the user is trying to finish in 10-20 words. Example: task: delete a contact Bob."
                        futures_to_data[
                            executor.submit(
                                queryGPT,
                                prompt=prompt,
                                console=CONSOLE,
                                identifier=f"{node}, element[{ide}]",
                            )
                        ] = (node, ide)
                for future in as_completed(futures_to_data):
                    node, ide = futures_to_data[future]
                    result[app][node][ide] = future.result()
            # with open(f"draft/result_{app}.json", "w") as f:
            #     json.dump(result[app], f)
        if self.backup_mode:
            with open("configs/json/element_description.json", "w") as f:
                json.dump(result, f)
        return result

    def getVisibleGraph(self, app):
        graph = self.exploration_data[app]["graph"]
        nt = Network("1500px", notebook=True, directed=True)
        nt.from_nx(graph)
        nt.toggle_physics(True)
        nt.show(f"{app}.html")

    def dataGeneration(self):
        # encoder = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        encoder = INSTRUCTOR('hkunlp/instructor-xl')
        result = {}
        for app in self.specify_apps:
            CONSOLE.log(f"START TO PROCESS [bold red2]{app}[/bold red2]")
            result[app] = {"desc": []}
            task_list = []
            for node, states in self.filtered_states[app].items():
                for ide, ele in enumerate(states["elements"]):
                    if not ele:
                        continue
                    desc = self.element_summary[app][node][ide]
                    path = ""
                    element_chain = copy.deepcopy(states["path"])
                    gpath = copy.deepcopy(states["gpath"])
                    element_chain.extend([ele])
                    gpath.append([node, ide])
                    for p in element_chain:
                        if "checkbox" in p:
                            if "checked=True" in p:
                                path += f"Uncheck {p}\n"
                            elif "checked=False" in p:
                                path += f"Check {p}\n"
                        else:
                            path += f"Click on {p}\n"
                    result[app]["desc"].append(
                        {"path": path, "task": desc, "gpath": gpath}
                    )
                    task_list.append(desc)
            encoded_tasks = encoder.encode(task_list)
            # pdb.set_trace()
            result[app]["data"] = encoded_tasks.tolist()
        if self.backup_mode:
            np.save("configs/external_mem.npy", result)
        return result

    def loadMemory(self, compressed_data=None, tasks_path="utgs"):
        result = {}
        encoder = INSTRUCTOR('hkunlp/instructor-xl')
        # encoder = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        if compressed_data:
            data = np.load(compressed_data, allow_pickle=True).item()
        else:
            data = self.dataGeneration()
        task_cfgs = Path(tasks_path).glob("**/*.yaml")
        for cfg in task_cfgs:
            with open(cfg, "r") as f:
                task = yaml.safe_load(f)["task_name"]
            app = cfg.parent.stem
            if self.specify_apps and app not in self.specify_apps:
                continue
            if app not in result:
                result[app] = {}
            embeds = np.array(data[app]["data"])
            task_embed = np.array(encoder.encode(task))
            cos_sim = np.dot(task_embed, embeds.T) / (
                np.linalg.norm(task_embed) * np.linalg.norm(embeds)
            )
            tgt_idx = int(np.where(cos_sim == np.max(cos_sim))[0][0])
            desc = data[app]["desc"][tgt_idx]["task"]
            task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()
            gpath = data[app]["desc"][tgt_idx]["gpath"]
            result[app][task_hash] = {
                "task": task,
                "similar": desc,
                "path": data[app]["desc"][tgt_idx]["path"],
                "gpath": [g[0] for g in gpath],
                "relevance": np.max(cos_sim),
            }
        with open(self.output_path, "w") as f:
            json.dump(result, f)

    def finetunePrompt(self):
        result = {}
        for app in self.specify_apps:
            CONSOLE.rule(f"START TO PROCESS {app.upper()}")
            app_data = self.filtered_states[app]
            result[app] = {}
            with ThreadPoolExecutor() as executor:
                futures_to_data = {}
                intro = f"You are a smartphone assistant to help users complete tasks by interacting with mobile app {app}.\nGiven a task, the previous UI actions, the content of current UI state, and the correct next one element id to interact with, your job is to answer the question in format.\n"
                task_prefix = "\nTask: "
                history_prefix = "\nPrevious UI actions: \n"
                screen_prefix = "\nCurrent UI state: \n"
                answer_prefix = "\n\nCorrect answer: "
                question = "\n\nYour answer should always use the following format: Based on the correct answer, completing this task on a smartphone usually involves these steps: <bullet list>. Just fill in the blanks."
                for node, info in app_data.items():
                    result[app][node] = [None] * len(info["elements"])
                    ah = f"- Start {app}\n"
                    path = info["path"]
                    for p in path:
                        if "checkbox" in p:
                            if "checked=True" in p:
                                ah += f"- Uncheck {p}\n"
                            elif "checked=False" in p:
                                ah += f"- Check {p}\n"
                        else:
                            ah += f"- Tap on {p}\n"
                    for ide, ele in enumerate(info["elements"]):
                        if not ele:
                            continue
                        target_id = re.findall("id=(\d+)", info["debug"][ide])[0]
                        task_name = (
                            re.sub(
                                "[tT]ask: ", "", self.element_summary[app][node][ide]
                            )
                            + "\n"
                        )
                        prompt = (
                            intro
                            + task_prefix
                            + task_name
                            + history_prefix
                            + ah
                            + screen_prefix
                            + self.exploration_data[app]["graph"].nodes[node]["title"]
                            + answer_prefix
                            + f"\n- Task finished: False\n- Next element id: {target_id}\n- Input needed: False\n- Input text: N/A"
                            + question
                        )
                        futures_to_data[
                            executor.submit(
                                queryGPT,
                                prompt=prompt,
                                console=CONSOLE,
                                identifier=f"{node}, element[{ide}]",
                            )
                        ] = (node, ide)
                for future in as_completed(futures_to_data):
                    node, ide = futures_to_data[future]
                    result[app][node][ide] = future.result()
            with open(f"draft/{app}_q1.json", "w") as f:
                json.dump(result[app], f)
        with open("draft/answer_q1.json", "w") as f:
            json.dump(result, f)
