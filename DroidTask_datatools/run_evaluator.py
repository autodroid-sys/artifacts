# example on evaluator.py usage, remember to specify API keys in `utils.py`
from scripts.evaluator import Evaluator

ev = Evaluator(
    data_folder="data/user_tasks",
    log_root="logs",
    # load_ckpt=True,
    memory_folder="data/navigation",
    use_memory=True,
    prompt_config="data/prompt.yaml",
    external_mem="data/ex_mem.json",
    instance_name="gpt-4",
    specify_app=["applauncher", "camera", "calendar", "clock", "contacts", "dialer", "filemanager", "gallery", "messenger", "musicplayer", "notes", "voicerecorder", "firefox"]
)

ev.evaluate()    
ev.get_acc_by_types()
