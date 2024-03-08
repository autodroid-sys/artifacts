# example on evaluator.py usage, remember to specify API keys in `utils.py`
from scripts.evaluator import Evaluator
INSTANCE_NAME="gpt-4"
ev = Evaluator(
    data_folder="data/user_tasks",
    log_root="logs",
    load_ckpt=True,
    memory_folder="data/navigation",
    use_memory=True,
    prompt_config="data/prompt.yaml",
    external_mem="data/ex_mem.json",
    instance_name=INSTANCE_NAME,
    specify_app=["applauncher", "camera", "calendar", "clock", "contacts", "dialer", "filemanager", "gallery", "messenger", "musicplayer", "notes", "voicerecorder", "firefox"]
)

all_results = ev.evaluate()    
action_results = ev.get_acc_by_types()
import json
artificial_results = {
    "action accuracy": all_results['Acc'],
    "completion rate": all_results['Whole_task_Acc_without_end'],
    "tap accuracy": action_results['tap_acc'],
    "input accuracy": action_results['input_acc'],
    "end accuracy": action_results['end_acc']
}
with open(f"logs/{INSTANCE_NAME}/Quantitative_results.json", "w") as f:
    json.dump(artificial_results, f, indent=4)
