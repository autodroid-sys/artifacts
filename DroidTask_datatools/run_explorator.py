# example on explorator.py usage, remember to specify API keys in `utils.py`
from scripts.explorator import Explorator

ex = Explorator(
    data_folder="data/raw_utgs",
    output_path="data/ex_mem_new.json",
    backup_mode=True,
    specify_apps=["applauncher", "camera", "calendar", "clock", "contacts", "dialer", "filemanager", "gallery", "messenger", "musicplayer", "notes", "voicerecorder", "firefox"]
)

ex.loadMemory(tasks_path="data/user_tasks/")