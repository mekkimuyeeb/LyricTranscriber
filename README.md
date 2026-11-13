# LyricTranscriber

100% local audio-to-LRC lyric transcription, LLM refinement, and forced word-level alignment tool built on `faster-whisper`, `lyrics-transcriber`, and local Ollama (`qwen3.5:9b`).

## Features

- **100% Offline / Local Transcription**: Uses `faster-whisper` (`vad_filter=False` with music-aware hallucination filtering) without requiring cloud API keys.
- **Hybrid Rule-Based + Local Ollama Refinement**: Combines `lyrics-transcriber` anchor matching with local Ollama (`qwen3.5:9b`) gap resolution and full-text cleanup with automatic retry and graceful fallback.
- **Phrase-Aware Forced Reference Alignment**: Aligns ground-truth reference lyrics (`--lyrics_file`) onto vocal timestamps using Needleman-Wunsch alignment, Unicode homoglyph normalization, breath lead-in clamping, and held-note sustain mastering.
- **Multi-Format Synchronized LRC Output**:
  - **Looper LRC (`--looper-lrc`)**: Two-line `[mm:ss.xxx]` + `[w:offset_ms,duration_ms,char_len;...]` word-sync format.
  - **Enhanced Word LRC (`--word-lrc`)**: A2 extension `[mm:ss.xx]<mm:ss.xx>word ...<mm:ss.xx>` for MusicBee, foobar2000 ESLyric, AIMP, and Symfonium.
  - **Sentence LRC (`--sentence-lrc`)**: Standard `[mm:ss.xx]Line` format for LRCGET, VLC, Navidrome, and standard players.
  - **Progressive Word LRC (`--progressive-lrc`)**: Cumulative word-by-word `[mm:ss.xx]` lines for standard players that do not parse inline `<...>` tags.
  - **MidiCo Karaoke LRC**: Word-per-line `[re:MidiCo]` format.

## Setup

1. Create a Python 3.11+ virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and adjust settings if needed:
   ```powershell
   Copy-Item .env.example .env
   ```

## Quick Start

### Force-Align Reference Lyrics to Audio
```powershell
.\.venv\Scripts\python.exe transcribe.py "path/to/song.mp3" --lyrics_file "path/to/reference_lyrics.txt" --looper-lrc
```

### Transcribe & Refine Without Reference Lyrics (Whisper + Ollama)
```powershell
.\.venv\Scripts\python.exe transcribe.py "path/to/song.mp3"
```

See [USAGE.md](USAGE.md) for the complete CLI flag reference, output file descriptions, and `config.yaml` options.
