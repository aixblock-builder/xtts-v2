import os
import builtins
from TTS.api import TTS

# Auto accept Coqui ToS
builtins.input = lambda _: "y"

assert os.path.isfile("male.wav"), "❌ 'male.wav' not found!"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")