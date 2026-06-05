# subtitles-creator

Tool to create subtitles from `.wav` or `.mp4` files, mainly intended for movies.

Generates subtitles using **local Whisper** and **diarization with Pyannote** to separate speakers.  
Optionally, it can translate/adapt the final subtitles with **DeepSeek**, improving idioms, slang and natural dialogue.

---

## Features

- Supports `.wav`, `.mp4`, `.mkv` and most formats readable by `ffmpeg`
- Uses **Whisper installed locally** on the system
- Uses **Pyannote VAD + diarization** from a separate environment: `diarization-env`
- Transcribes only detected speech regions to reduce Whisper hallucinations
- Generates intermediate debug files and a clean final `.srt`
- Optional DeepSeek translation/adaptation script
- Executable from terminal with a single command

---

## Requirements

Before using it, you need to have installed:

1. **ffmpeg**
2. **Whisper** locally on the system
3. **diarization-env** with `pyannote.audio`
4. A Hugging Face token with access to Pyannote models
5. Execution permissions on the script

Example Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token"
```

or:

```bash
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

---

## Installation

Give execution permissions to the script:

```bash
chmod +x run_pipeline.sh
```

Run the pipeline:

```bash
./run_pipeline.sh "movie.mp4"
```

or:

```bash
./run_pipeline.sh "audio.wav"
```

---

## Structure

```text
subtitles-creator/
├── run_pipeline.sh
├── README.md
├── filter_with_vad.py              # runs with /diarization-env
├── whisper_transcribe.py
└── generate_subs_final.py

optional, translation:
└── deepseek_agent_translator.py    # requires a DeepSeek API key
```

---

## Files explanation

### `run_pipeline.sh`

Main entry point of the project.

It runs the full subtitle pipeline:

1. Converts the input video/audio to a temporary mono WAV file at 16 kHz
2. Runs VAD and diarization with Pyannote
3. Runs Whisper only on detected speech regions
4. Generates the final `.srt` subtitle file

It also prints the execution time for each step and lists all generated files at the end.

Important: the script expects the Pyannote environment here:

```text
./diarization-env/bin/python
```

So `filter_with_vad.py` is executed using the Python from `diarization-env`.

Current default values inside the script:

```bash
WHISPER_MODEL="large-v3"
LANGUAGE="en"
```

If the source language is not English, it is recommended to change `LANGUAGE` to the 2-letter ISO code of the source language:

```bash
LANGUAGE="it"
LANGUAGE="fr"
LANGUAGE="ja"
LANGUAGE="zh"
```

`auto` can be used for auto-detection, but it is mainly recommended for multilingual movies.

---

### `filter_with_vad.py`

First stage of the pipeline.

Input:

```text
WAV mono 16 kHz
```

This script runs with `/diarization-env`, because it depends on Pyannote.

It uses Pyannote to perform:

- Voice Activity Detection, VAD
- Speech region extraction
- Speaker diarization

Generated files:

```text
<base>_vad_speech.json
<base>_vad_debug.srt
<base>_pyannote_speakers.json
```

What each output means:

- `_vad_speech.json`: list of detected voice regions
- `_vad_debug.srt`: visual/debug subtitle file showing where voice was detected
- `_pyannote_speakers.json`: speaker turns detected by diarization

The script is designed to work with different Pyannote versions.  
Some versions use `onset/offset`, others use `threshold`, so it tries to instantiate the VAD pipeline in a compatible way.

---

### `whisper_transcribe.py`

Second stage of the pipeline.

Input:

```text
WAV mono 16 kHz
<base>_vad_speech.json
```

It loads the audio and transcribes only the regions where VAD detected speech.

This helps avoid:

- Whisper hallucinations during silence
- Fake subtitles over noise
- Repeated random text
- Credits or unwanted subtitle-like text

Generated files:

```text
<base>_whisper_raw.json
<base>_filtered.json
<base>_filtered.srt
```

What each output means:

- `_whisper_raw.json`: raw Whisper output, including metadata and word timestamps
- `_filtered.json`: cleaned/filtered segments after confidence checks
- `_filtered.srt`: intermediate subtitle file before final formatting

The script applies filters based on:

- average log probability
- no-speech probability
- word confidence
- compression ratio
- repeated text detection
- very long or suspicious segments

---

### `generate_subs_final.py`

Third and final stage of the main pipeline.

Input:

```text
<base>_filtered.json
<base>_pyannote_speakers.json
```

It creates the final subtitle file:

```text
<base>_final.srt
```

This script formats subtitles with scene/Netflix-style rules:

- Maximum 2 lines per subtitle
- Maximum 42 characters per line by default
- Avoids overlapping timestamps
- Controls CPS, characters per second
- Adds dialogue formatting when two speakers are detected
- Merges or splits captions when needed
- Uses diarization to improve speaker/dialogue layout

The final result is the subtitle file intended for real use.

---

### `deepseek_agent_translator.py` optional, but very recomended!

Optional translation/adaptation script.

It takes an existing `.srt` file and uses the **DeepSeek API** to clean, correct and translate subtitles.

This is useful because it does more than literal translation:

- Adapts idioms
- Improves slang and natural dialogue
- Fixes some Whisper transcription errors before translating
- Preserves SRT indexes and timestamps
- Keeps dialogue formatting
- Includes an audit step to detect suspicious untranslated lines

It requires a DeepSeek API key:

```python
API_KEY = ""
```

You also need to configure the input/output files and languages directly inside the script:

```python
INPUT_FILE = ""
OUTPUT_FILE = ""

# Source: "auto" or ISO code: it, en, ja, zh, ru, etc.
SOURCE_LANGUAGE = ""

# Target: es, en, it, fr, pt, de, ar, zh, ja, ko, ru, sr, etc.
# Note: Serbian is "sr", not "rs". If you put "rs", it is converted to "sr".
TARGET_LANGUAGE = ""
```

Example:

```python
INPUT_FILE = "movie_final.srt"
OUTPUT_FILE = "movie_final_es.srt"

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "es"
```

Run it with:

```bash
python3 deepseek_agent_translator.py
```

Optional recommended dependencies for stronger audit:

```bash
pip install wordfreq lingua-language-detector
```

---

## Generated files

After running the main pipeline, you should get files like:

```text
movie_vad_speech.json
movie_vad_debug.srt
movie_pyannote_speakers.json
movie_whisper_raw.json
movie_filtered.json
movie_filtered.srt
movie_final.srt
```

The most important final file is:

```text
movie_final.srt
```

---

## Pipeline summary

The pipeline works in three main phases.

First, `run_pipeline.sh` converts the original video or audio into a normalized WAV file.  
Then `filter_with_vad.py`, running inside `diarization-env`, detects where speech exists and identifies speaker turns with Pyannote.  
After that, `whisper_transcribe.py` sends only those speech regions to Whisper, reducing silence hallucinations and filtering bad segments.  
Finally, `generate_subs_final.py` formats everything into a clean final `.srt`, using diarization data to improve dialogue layout.

Optionally, `deepseek_agent_translator.py` can be used after the final SRT is generated to translate and adapt the subtitles into another language with more natural phrasing.

In short:

```text
video/audio
   ↓
ffmpeg normalization
   ↓
Pyannote VAD + diarization
   ↓
Whisper transcription by speech chunks
   ↓
filtering and cleanup
   ↓
final SRT generation
   ↓
optional DeepSeek translation/adaptation
```

---

## Notes

1. Especially designed for the target language to be neutral Spanish, but it can translate other languages to other languages.
2. `filter_with_vad.py` must run with the Python inside `diarization-env`, because that environment contains Pyannote.
3. In `run_pipeline.sh`, the variable `LANGUAGE="en"` is currently hardcoded to English as source language. It is recommended to set it using the 2-digit ISO code of the source language: `it`, `ja`, `fr`, `zh`, etc.
4. `auto` means auto-detect, but it is recommended mainly for multilingual movies.
5. `deepseek_agent_translator.py` is optional, but important if you want better adaptation of idioms, slang, natural expressions and cultural context.
6. Mostly code with LLMs + 20 years of coding without it :P
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

