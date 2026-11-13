# LyricTranscriber Usage Guide

[`transcribe.py`](file:///c:/Users/Mekanator/.gemini/antigravity/scratch/LyricTranscriber/transcribe.py) transcribes audio tracks locally using `faster-whisper`, refines/corrects lyrics using local Ollama (`qwen3.5:9b`) or a local reference lyrics file (`--lyrics_file`), and generates synchronized `.lrc` and `.txt` files.

---

## 1. Quick Start Commands

Run all commands from the project directory (`c:\Users\Mekanator\.gemini\antigravity\scratch\LyricTranscriber`) using the project virtual environment (`.\.venv\Scripts\python.exe`).

### A. Transcribe & Sync Using Your Reference Lyrics File (Recommended)
When you have corrected lyrics in a `.txt` file, pass `--lyrics_file` to force-align 100% of your lyrics onto the vocal timestamps:

```powershell
cd C:\Users\Mekanator\.gemini\antigravity\scratch\LyricTranscriber

.\.venv\Scripts\python.exe transcribe.py "C:\Users\Mekanator\Music\Louve\SCORPIO VÉNUS\03 BRÛLE.mp3" --lyrics_file "Louve - BRÛLE (Reference Lyrics).txt"
```

> **Note:** When you pass `--lyrics_file`, the script automatically saves a clean normalized copy to `<Artist> - <Title> (Reference Lyrics).txt` so your reference file is never overwritten.

---

### B. Choose Your Primary `.lrc` Output Format (`--lrc-mode`)
You can control which LRC format is written to the main `<Artist> - <Title>.lrc` file (and all four companion files are always generated alongside it):

1. **Looper Mode (`--looper-lrc` or `--lrc-mode looper`)** — *Two-line `[mm:ss.xxx]` + `[w:offset_ms,duration_ms,char_len;...]` format for Looper*:
   ```powershell
   .\.venv\Scripts\python.exe transcribe.py "C:\Users\Mekanator\Music\Louve\SCORPIO VÉNUS\03 BRÛLE.mp3" --lyrics_file "Louve - BRÛLE (Reference Lyrics).txt" --looper-lrc
   ```
   Output format:
   ```text
   [00:14.560]J'ai envie de sentir ta peau contre moi
   [w:0,380,5;380,380,6;760,240,3;1000,300,7;1300,380,3;1680,340,5;2020,260,7;2280,400,3]
   ```

2. **Sentence Mode (`--sentence-lrc` or `--lrc-mode sentence`)** — *Best for LRCGET and standard players*:
   ```powershell
   .\.venv\Scripts\python.exe transcribe.py "C:\Users\Mekanator\Music\Louve\SCORPIO VÉNUS\03 BRÛLE.mp3" --lyrics_file "Louve - BRÛLE (Reference Lyrics).txt" --sentence-lrc
   ```
   Output format:
   ```text
   [00:14.56]J'ai envie de sentir ta peau contre moi
   [00:18.26]J'ai envie de boire mon café avec toi
   ```

3. **Enhanced Word Mode (`--word-lrc` or `--lrc-mode word`)** — *Best for A2 karaoke players (MusicBee, AIMP, foobar2000 ESLyric, Symfonium)*:
   ```powershell
   .\.venv\Scripts\python.exe transcribe.py "C:\Users\Mekanator\Music\Louve\SCORPIO VÉNUS\03 BRÛLE.mp3" --lyrics_file "Louve - BRÛLE (Reference Lyrics).txt" --word-lrc
   ```
   Output format:
   ```text
   [00:14.56]<00:14.56>J'ai <00:14.94>envie <00:15.32>de <00:15.56>sentir <00:15.86>ta <00:16.24>peau <00:16.58>contre <00:16.84>moi<00:17.24>
   ```

4. **Progressive Word Mode (`--progressive-lrc` or `--lrc-mode progressive`)** — *Word-by-word reveal for standard `[mm:ss.xx]` players that do not support `<...>` tags*:
   ```powershell
   .\.venv\Scripts\python.exe transcribe.py "C:\Users\Mekanator\Music\Louve\SCORPIO VÉNUS\03 BRÛLE.mp3" --lyrics_file "Louve - BRÛLE (Reference Lyrics).txt" --progressive-lrc
   ```
   Output format:
   ```text
   [00:14.56]J'ai
   [00:14.94]J'ai envie
   [00:15.32]J'ai envie de
   ```

---

### C. Transcribe Without a Reference File (Auto Whisper + Ollama `qwen3.5:9b` Refinement)
If you don't have a reference `.txt` file yet, omit `--lyrics_file`. The script will transcribe the audio with `faster-whisper` and use your local Ollama model (`qwen3.5:9b` at `http://localhost:11434`) to clean up and correct the lyrics:

```powershell
.\.venv\Scripts\python.exe transcribe.py "C:\Path\To\Song.mp3"
```

---

## 2. Generated Output Files

Every run creates the following files in your output directory:

| File Name | Description |
| :--- | :--- |
| `<Artist> - <Title>.lrc` | **Primary LRC file**, formatted according to `--lrc-mode` (`looper`, `word`, `sentence`, or `progressive`). |
| `<Artist> - <Title> (Looper).lrc` | Looper word-synced format (`[mm:ss.xxx]` + `[w:offset_ms,duration_ms,char_len;...]`). |
| `<Artist> - <Title> (Sentence).lrc` | Clean line-by-line `[mm:ss.xx]` LRC (compatible with LRCGET and all standard players). |
| `<Artist> - <Title> (Enhanced Word).lrc` | Enhanced LRC (A2 `<mm:ss.xx>` inline word tags) for word-level karaoke players. |
| `<Artist> - <Title> (Progressive Word).lrc` | Cumulative word-by-word `[mm:ss.xx]` LRC (no `<...>` tags). |
| `<Artist> - <Title> (Karaoke).lrc` | MidiCo word-level karaoke format (`[re:MidiCo]`, `[mm:ss.mmm]1:/word`). |
| `<Artist> - <Title> (Lyrics Corrected).txt` | Clean plain-text lyrics (one line per synced segment). |
| `<Artist> - <Title> (Reference Lyrics).txt` | Safe normalized backup of the `--lyrics_file` you provided. |
| `<Artist> - <Title> (Lyrics Uncorrected).txt` | Raw uncorrected Whisper transcription text. |
| `<Artist> - <Title> (Lyrics Corrections).json` | Detailed alignment and word-timing metadata JSON. |

---

## 3. CLI Options Reference

| Argument | Description |
| :--- | :--- |
| `audio_file` | **(Required)** Path to the audio file (`.mp3`, `.flac`, `.wav`, `.m4a`). |
| `--lyrics_file`, `--lyrics-file <path>` | Path to a local `.txt` file containing ground-truth/corrected lyrics to force-align. |
| `--lrc-mode {looper,word,sentence,line,progressive}` | Format for the primary `<Artist> - <Title>.lrc` output file. |
| `--looper-lrc` | Shorthand for `--lrc-mode looper` (`[mm:ss.xxx]` + `[w:offset,duration,char_len;...]`). |
| `--sentence-lrc`, `--line-lrc` | Shorthand for `--lrc-mode sentence`. |
| `--word-lrc` | Shorthand for `--lrc-mode word`. |
| `--progressive-lrc` | Shorthand for `--lrc-mode progressive`. |
| `--artist "<Artist>"` | Override artist name (auto-detected from ID3 tags or folder structure if omitted). |
| `--title "<Title>"` | Override song title (auto-detected from ID3 tags or filename if omitted). |
| `--language <code>` | Optional ISO language code for Whisper (e.g., `fr`, `en`). |
| `--whisper-model <size>` | Override local Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). Default: `small`. |
| `--output_dir <dir>` | Directory to write generated files to (defaults to current working directory). |
| `--enable-review` | Launch the interactive browser review UI after alignment. |
| `--config <path>` | Path to YAML configuration file (defaults to [`config.yaml`](file:///c:/Users/Mekanator/.gemini/antigravity/scratch/LyricTranscriber/config.yaml)). |

---

## 4. Configuration (`config.yaml` & `.env`)

You can set permanent defaults in [`config.yaml`](file:///c:/Users/Mekanator/.gemini/antigravity/scratch/LyricTranscriber/config.yaml):

```yaml
refinement:
  enabled: true
  ollama_url: "http://localhost:11434"
  model_name: "qwen3.5:9b"
  timeout_seconds: 60
  max_retries: 3
  fallback_on_error: true

transcription:
  whisper_model: "small"      # tiny, base, small, medium, large-v3
  device: "cpu"               # cpu or cuda
  compute_type: "int8"        # int8, float16, float32

output:
  lrc_mode: "word"            # "word", "sentence", or "progressive"
```

### Force a Fresh Whisper Transcription (Clear Cache)
Whisper raw transcriptions are cached in `C:\Users\Mekanator\lyrics-transcriber-cache` so re-runs with updated lyrics files finish in ~1.5 seconds. If you change `--whisper-model` or `--language` and want to force a fresh audio pass, delete the cached `local_whisper_*.json` file in `C:\Users\Mekanator\lyrics-transcriber-cache`.
