import sys, importlib.util, os

# If aifc is missing, load our vendored version
if "aifc" not in sys.modules:
    spec = importlib.util.spec_from_file_location("aifc", os.path.join(os.path.dirname(__file__), "aifc.py"))
    aifc = importlib.util.module_from_spec(spec)
    sys.modules["aifc"] = aifc
    spec.loader.exec_module(aifc)

import speech_recognition as sr
