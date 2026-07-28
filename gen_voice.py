#!/usr/bin/env python3
# Kokoro TTS: gen_voice.py <voice> <out.wav>  ; greeting text on stdin. 24kHz wav.
import sys
import numpy as np
import soundfile as sf
from kokoro import KPipeline

voice, out = sys.argv[1], sys.argv[2]
text = sys.stdin.read().strip()
pipe = KPipeline(lang_code='a')
chunks = [audio for _, _, audio in pipe(text, voice=voice)]
audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
sf.write(out, np.asarray(audio, dtype=np.float32), 24000)
print("VOICE_OK", out, round(len(audio) / 24000, 1), "s")
