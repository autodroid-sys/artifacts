# example on evaluator.py usage, remember to specify API keys in `utils.py`
from .evaluator import Evaluator

ev = Evaluator(
    data_folder="data/user_tasks",
    log_root="logs",
    memory_folder="data/navigation",
    use_memory=True,
    prompt_config="data/prompt.yaml",
    external_mem="data/ex_mem.json",
    instance_name="autodroid-artifact",
    specify_app=["calendar", "clock", "contacts", "dialer", "messenger"],
)

ev.evaluate()    