# subtitles-creator

Tool to create subtitles from .wav or .mp4 files, mainly intended for movies.

Generates subtitles using **local Whisper** and, **diarization with pyannote** to separate speakers.

---

## Features

- Supports .wav and .mp4 files
- Uses **Whisper installed locally** on the system
- Supports **diarization** via a separate environment (diarization-env)
- Designed to automate the creation of subtitles for movies
- Executable from terminal with a single command

---

## Requirements

Before using it, you need to have installed:

1. **Whisper** locally on the system  
2. **diarization-env** with pyannote from Hugging Face  
3. **ffmpeg** installed on the system  
4. Execution permissions on the script

---

## Installation

Give execution permissions to the script:
```
chmod +x run_pipeline.sh
LANGUAGE=en WHISPER_MODEL=large-v3 ./run_pipeline.sh ".mp4 or .wav"
```

---

## Structure

subtitles-creator/<br>
├── run_pipeline.sh<br>
├── README.md<br>
├── filter_with_vad.py (with /diarization-env)<br>
├── whisper_transcribe.py<br>
└── generate_subs_final.py<br>
<br>
optional, translation:<br>
└── deepseek_agent_translator.py (requires a deepseek API key)

---

## NOTE

1) Especially designed for the target language to be neutral Spanish, but it can translate other langs -> other langs.
2) Mostly code with LLMs + 20 years of coding without it :P
3) in run_pipeline.sh the variable LANGUAGE="en" is hardcoded to "en" (source lang), it's recommended to set it using the 2-digit ISO code of the source language ("it", "jp", "fr", etc). 'auto' means auto-detect, but i recommend that only for multilingual movies.
4) deepseek_agent_translator.py needs to hardcode the variables:
```
INPUT_FILE = ""
OUTPUT_FILE = ""

# Source: "auto" or ISO code: it, en, ja, zh, ru, etc.
SOURCE_LANGUAGE = ""

# Target: es, en, it, fr, pt, de, ar, zh, ja, ko, ru, sr, etc.
# Note: Serbian is "sr", not "rs". If you put "rs", it is converted to "sr".
TARGET_LANGUAGE = ""
```

