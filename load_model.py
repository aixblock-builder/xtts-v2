import os
import builtins
from TTS.api import TTS

# Auto accept Coqui ToS
builtins.input = lambda _: "y"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")