# DroidTask

Dataset and evaluator of AutoDroid

## Environment Setup
```
conda create -n droidtask python=3.10
pip install -r requirements.txt
```

or:

```
conda env create --name droidtask --file environment.yml
```

## User Task YAMLs

```
user_tasks
├── applauncher
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   ├── task4.yaml
│   └── task5.yaml
├── calendar
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   ├── task4.yaml
│   ├── task5.yaml
│   ├── task6.yaml
│   ├── task7.yaml
│   ├── task8.yaml
│   └── task9.yaml
```
there may be also other subfolders in each app folder such as `events`, etc. They are
generated by autodroid. You can get all information recorded by AutoDroid in these
additional files.

## Explore & Memory Generation
**Notice**: We have provided our generated `.json` file. If you want
to directly use our `data/ex_mem.json` and memory, you can skip this part.
```
utgs
├── applauncher
│   └── utg.yaml
├── calendar
│   └── utg.yaml
```
For the convenience of your direct usage, we keep raw files in the folder. If you
want to create your memory on your own, you need to extract all the `utg.yaml`, and
rename them into `${app}.yaml`, putting them in a `explorations_data` folder.

```python
# run_explorator.py
# example on explorator.py usage, remember to specify API keys in `utils.py`
from .explorator import Explorator

ex = Explorator(
    data_folder="data/exploration_data",
    output_path="data/ex_mem_new.json"
)

ex.loadMemory(tasks_path="data/user_tasks/")
```
in case there may be network problem when you execute code, we provide a `backup_mode`
in the `explorator` code, which will save essential data each step, you can check the 
code and adjust your save path if needed.

## Evaluator
You can evaluate LLM's answers through this code.

**Note: If you set `load_ckpt` to `False`, you will query GPT to generate answers, which costs at least 272,8416 input tokens, and about 150,000 output tokens for all the apps. If you want to use the existing checkpoint, set `load_ckpt` to `False`, and set `INSTANCE_NAME` to the name of the existing checkpoint, for example, `INSTANCE_NAME = "gpt-3.5" or "gpt-4"`.**
```python
# run_evaluator.py
# example on evaluator.py usage, remember to specify API keys in `utils.py`
from .evaluator import Evaluator

ev = Evaluator(
    data_folder="data/user_tasks",
    log_root="logs",
    memory_folder="data/navigation",
    use_memory=True,
    prompt_config="data/prompt.yaml",
    external_mem="data/ex_mem.json",
    instance_name="<set-a-name-by-yourself>",
    specify_app=["calendar", "clock", "contacts", "dialer", "messenger"],
)

ev.evaluate()
```

Some explanations on the parameters:
- `data_folder`: `utgs` folder path.
- `log_root`: log folder.
- `load_ckpt`: bool, whether to use the existing checkpoint.
- `use_memory`: bool, whether to integrate memory.
- `memory_folder`(only if `use_memory`): `navigation` folder path; we provide our data in `data/navigation`, which is similar to `UI_function_table` in the paper; each key represents a `state_str`
of one UI state, and values represent next UI state's functions when operating on these indices of
elements.
- `prompt_config`: configuration file for prompts design. Default to `data/prompt.yaml`
- `use_baseline`: bool, whether to use baseline methods.
- `external_mem`: `.json` file generated by `explorator`. also provided in download link for directly
usage (`ex_mem.json`).``
- `instance_name`: subfolder to store log information for current run below `log_root`.
- `specify_app`: a list, you can choose to just evaluate some of the apps in the `data_folder`, such as ['applauncher', 'calendar'], etc.
- `load_result`: without querying LLMs, just load the result from former logs, when using this
option, set the `instance_name` same as the logs you want to load from.

### Notice

- The output of GPT is random, so the evaluation results may vary. The results in the paper are averaged over 3 runs.