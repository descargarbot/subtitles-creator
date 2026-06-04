# subtitles-creator

Tool to create subtitles from .wav or .mp4 files, mainly intended for movies.

Generates subtitles using **local Whisper** and, optionally, **diarization with pyannote** to separate speakers.

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
bash
chmod +x run_pipeline.sh
./run_pipeline.sh "file .mp4 or .wav"
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

Especially designed for the target language to be neutral Spanish, but it can translate other langs -> other langs


