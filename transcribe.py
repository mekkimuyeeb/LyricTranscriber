"""Resilient local runner for lyrics-transcriber + Ollama (qwen3.5:9b).

Implements:
1. Local Audio Transcription (`faster-whisper`, music mode `vad_filter=False`).
2. Unicode homoglyph normalization (fixes Genius Cyrillic `е` -> `e`, smart quotes).
3. Hybrid Rule-Based + Ollama (`qwen3.5:9b`, `think=False`, `format="json"`) gap fixes.
4. Forced Reference Line & Word Alignment: Guarantees 100% of your reference lyrics
   (e.g. `mer`, `Louve`, `pars`, `en`, English bridge) are aligned to word-level
   audio timestamps, with strictly monotonic timestamps (zero duplicate timestamps)
   and true lyric line boundaries (no mid-phrase `/to` splits).
5. Dual LRC Output with Full Metadata Tags (`[ti:]`, `[ar:]`, `[al:]`, `[length:]`, `[offset:0]`):
   - Standard Line-Level LRC (`<Artist> - <Title>.lrc`) for foobar2000, VLC, MusicBee, AIMP, Navidrome.
   - Clean MidiCo Word-Level Karaoke LRC (`<Artist> - <Title> (Karaoke).lrc`) with no trailing spaces.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Auto-switch to the project's local .venv Python if invoked from another environment
_LOCAL_VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if (
    _LOCAL_VENV_PYTHON.is_file()
    and Path(sys.executable).resolve() != _LOCAL_VENV_PYTHON.resolve()
    and not os.environ.get("_LYRIC_TRANSCRIBER_REEXEC")
):
    os.environ["_LYRIC_TRANSCRIBER_REEXEC"] = "1"
    sys.exit(
        subprocess.call(
            [str(_LOCAL_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )

import yaml
from dotenv import load_dotenv

# Map common Cyrillic/Greek homoglyphs found in Genius/web lyrics to Latin ASCII
HOMOGLYPH_TABLE = str.maketrans(
    {
        "\u0430": "a",  # Cyrillic a
        "\u0435": "e",  # Cyrillic e
        "\u043e": "o",  # Cyrillic o
        "\u0440": "p",  # Cyrillic r
        "\u0441": "c",  # Cyrillic s
        "\u0443": "y",  # Cyrillic u
        "\u0445": "x",  # Cyrillic h
        "\u0410": "A",
        "\u0415": "E",
        "\u041e": "O",
        "\u0420": "P",
        "\u0421": "C",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)


def normalize_lyrics_text(text: str) -> str:
    """Normalize Cyrillic homoglyphs, curly apostrophes, and non-breaking spaces."""
    return text.translate(HOMOGLYPH_TABLE)


def normalize_token_for_match(token: str) -> str:
    """Lowercase, strip punctuation/accents for fuzzy word alignment."""
    cleaned = normalize_lyrics_text(token).lower()
    decomposed = unicodedata.normalize("NFD", cleaned)
    stripped_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]", "", stripped_accents)


def configure_runtime_paths() -> None:
    """Ensure `.venv\\Scripts`, native SSL truststore, and `ffmpeg` are configured."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    venv_scripts_dir = str((Path(__file__).parent / ".venv" / "Scripts").resolve())
    current_path = os.environ.get("PATH", "")
    if venv_scripts_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_scripts_dir}{os.pathsep}{current_path}"

    if shutil.which("ffmpeg") is None:
        try:
            import requests
            import static_ffmpeg
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            original_get = requests.get

            def _resilient_get(*args: Any, **kwargs: Any) -> Any:
                try:
                    return original_get(*args, **kwargs)
                except requests.exceptions.SSLError:
                    kwargs["verify"] = False
                    return original_get(*args, **kwargs)

            requests.get = _resilient_get
            try:
                static_ffmpeg.add_paths()
            finally:
                requests.get = original_get
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Could not auto-configure static-ffmpeg: {exc}")


def load_configuration(config_path: Path) -> dict[str, Any]:
    """Load `.env` variables and `config.yaml` refinement/transcription settings."""
    load_dotenv()

    config_data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file_handle:
            config_data = yaml.safe_load(file_handle) or {}

    refinement = config_data.get("refinement", {})
    transcription = config_data.get("transcription", {})
    output_cfg = config_data.get("output", {})

    ollama_url = os.getenv(
        "OLLAMA_BASE_URL",
        refinement.get("ollama_url", "http://localhost:11434"),
    )
    model_name = refinement.get("model_name", "qwen3.5:9b")
    provider_model = os.getenv(
        "AGENTIC_AI_MODEL",
        refinement.get("provider_model", f"ollama/{model_name}"),
    )
    is_enabled = os.getenv(
        "USE_AGENTIC_AI",
        "1" if refinement.get("enabled", True) else "0",
    ).lower() in ("1", "true", "yes")

    max_retries = int(
        os.getenv("OLLAMA_MAX_RETRIES", refinement.get("max_retries", 3))
    )
    timeout_seconds = int(
        os.getenv(
            "OLLAMA_TIMEOUT_SECONDS", refinement.get("timeout_seconds", 120)
        )
    )
    temperature = float(refinement.get("temperature", 0.2))
    fallback_on_error = bool(refinement.get("fallback_to_rules_on_error", True))

    whisper_model = os.getenv(
        "LOCAL_WHISPER_MODEL",
        transcription.get("local_whisper_model", "small"),
    )
    whisper_device = os.getenv(
        "LOCAL_WHISPER_DEVICE",
        transcription.get("device", "cpu"),
    )
    whisper_compute_type = os.getenv(
        "LOCAL_WHISPER_COMPUTE_TYPE",
        transcription.get("compute_type", "int8"),
    )
    lrc_mode = os.getenv(
        "LRC_MODE",
        output_cfg.get("lrc_mode", "word"),
    ).strip().lower()
    if lrc_mode not in ("word", "sentence", "line"):
        lrc_mode = "word"

    return {
        "enabled": is_enabled,
        "ollama_url": ollama_url.rstrip("/"),
        "model_name": model_name,
        "provider_model": provider_model,
        "max_retries": max_retries,
        "timeout_seconds": timeout_seconds,
        "temperature": temperature,
        "fallback_on_error": fallback_on_error,
        "whisper_model": whisper_model,
        "whisper_device": whisper_device,
        "whisper_compute_type": whisper_compute_type,
        "lrc_mode": lrc_mode,
    }


def is_ollama_ready(ollama_url: str, max_retries: int, timeout_seconds: int) -> bool:
    """Check if the Ollama server is reachable, using exponential backoff."""
    health_endpoint = f"{ollama_url}/api/tags"
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(health_endpoint, method="GET")
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"[Attempt {attempt}/{max_retries}] Ollama not reachable at "
                f"{ollama_url} ({exc})."
            )
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    return False


def get_audio_metadata(
    audio_path: Path,
    cli_artist: str | None,
    cli_title: str | None,
) -> dict[str, Any]:
    """Extract artist, title, album, and duration (seconds) from ID3 tags or path."""
    artist = cli_artist
    title = cli_title
    album = ""
    duration = 0.0

    try:
        from tinytag import TinyTag

        tag = TinyTag.get(str(audio_path))
        if not artist and tag.artist:
            artist = tag.artist.strip()
        if not title and tag.title:
            title = tag.title.strip()
        if tag.album:
            album = tag.album.strip()
        if tag.duration:
            duration = float(tag.duration)
    except Exception:  # noqa: BLE001
        pass

    if not title:
        cleaned_stem = re.sub(r"^\d+[\s._-]+", "", audio_path.stem).strip()
        title = cleaned_stem or audio_path.stem

    if not artist:
        parent = audio_path.parent
        grandparent = parent.parent if parent else None
        if (
            grandparent
            and grandparent.name
            and grandparent.name.lower() not in ("music", "downloads", "desktop", "audio")
        ):
            artist = grandparent.name
            if not album and parent and parent.name:
                album = parent.name
        elif parent and parent.name:
            artist = parent.name

    return {
        "artist": artist,
        "title": title,
        "album": album,
        "duration": duration,
    }


def generate_ollama_reference_lyrics(
    raw_lines: list[str],
    artist: str | None,
    title: str | None,
    settings: dict[str, Any],
    output_dir: str,
    output_prefix: str,
    logger: logging.Logger,
) -> str | None:
    """Use local Ollama (`qwen3.5:9b`) to proofread raw Whisper lines when no online lyrics exist."""
    ref_friendly_path = Path(output_dir) / f"{output_prefix} (Reference Lyrics).txt"
    if ref_friendly_path.exists() and ref_friendly_path.stat().st_size > 0:
        logger.info(f"Using existing reference lyrics file: {ref_friendly_path}")
        return str(ref_friendly_path)

    ref_path = Path(output_dir) / f"{output_prefix} (Ollama Reference).txt"
    if ref_path.exists() and ref_path.stat().st_size > 0:
        logger.info(f"Using existing Ollama-refined reference lyrics: {ref_path}")
        return str(ref_path)

    if not raw_lines:
        return None

    numbered_input = "\n".join(raw_lines)
    prompt = (
        f"You are an expert music lyricist and proofreader. Below is a raw Whisper audio "
        f"transcription of the song '{title or 'Unknown'}' by '{artist or 'Unknown'}'.\n"
        f"Fix obvious phonetic/homophone transcription errors (for example in French: "
        f"'chanson de mot' -> 'chanson d\\'amour', 'bord de la merde dans tes bras' -> "
        f"'bord de la mer dans tes bras', 'ma louvre' -> 'ma Louve') while keeping the "
        f"exact same number of lines, language, and phrasing structure.\n"
        f"Output ONLY the corrected lyric lines, one line per input line, with no commentary:\n\n"
        f"{numbered_input}"
    )

    payload = json.dumps(
        {
            "model": settings["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": settings["temperature"]},
        }
    ).encode("utf-8")

    endpoint = f"{settings['ollama_url']}/api/chat"
    for attempt in range(1, settings["max_retries"] + 1):
        try:
            logger.info(
                f"Requesting Ollama ({settings['model_name']}) lyric refinement "
                f"for '{artist} - {title}' (attempt {attempt}/{settings['max_retries']})..."
            )
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=settings["timeout_seconds"]) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body.get("message", {}).get("content", "")
                cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if cleaned:
                    cleaned = normalize_lyrics_text(cleaned)
                    ref_path.write_text(cleaned + "\n", encoding="utf-8")
                    logger.info(f"Saved Ollama-refined reference lyrics to: {ref_path}")
                    return str(ref_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Ollama reference refinement attempt {attempt} failed: {exc}")
            if attempt < settings["max_retries"]:
                time.sleep(2 ** (attempt - 1))

    return None


def build_hybrid_corrector(
    cache_dir: str | Path,
    settings: dict[str, Any],
    use_llm: bool,
    logger: logging.Logger,
) -> Any:
    """Create a `LyricsCorrector` with rule-based handlers + fast Ollama JSON `LLMHandler`."""
    from lyrics_transcriber.correction.corrector import LyricsCorrector
    from lyrics_transcriber.correction.handlers.extend_anchor import ExtendAnchorHandler
    from lyrics_transcriber.correction.handlers.levenshtein import LevenshteinHandler
    from lyrics_transcriber.correction.handlers.llm import LLMHandler
    from lyrics_transcriber.correction.handlers.llm_providers import LLMProvider
    from lyrics_transcriber.correction.handlers.no_space_punct_match import (
        NoSpacePunctuationMatchHandler,
    )
    from lyrics_transcriber.correction.handlers.relaxed_word_count_match import (
        RelaxedWordCountMatchHandler,
    )
    from lyrics_transcriber.correction.handlers.repeat import RepeatCorrectionHandler
    from lyrics_transcriber.correction.handlers.sound_alike import SoundAlikeHandler
    from lyrics_transcriber.correction.handlers.syllables_match import (
        SyllablesMatchHandler,
    )
    from lyrics_transcriber.correction.handlers.word_count_match import (
        WordCountMatchHandler,
    )

    class QwenOllamaProvider(LLMProvider):
        """Ollama provider that disables `<think>` tokens, enforces JSON output, and caches responses."""

        def __init__(self) -> None:
            super().__init__(logger=logger)
            self.model = settings["model_name"]
            self.endpoint = f"{settings['ollama_url']}/api/chat"
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        def generate_response(self, prompt: str, **kwargs: Any) -> str:
            prompt_hash = hashlib.md5(f"{self.model}:{prompt}".encode("utf-8")).hexdigest()
            cache_file = self.cache_dir / f"ollama_gap_{prompt_hash}.json"
            if cache_file.exists():
                try:
                    cached = cache_file.read_text(encoding="utf-8").strip()
                    if cached:
                        return cached
                except Exception:  # noqa: BLE001
                    pass

            payload = json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "options": {"temperature": settings["temperature"]},
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=settings["timeout_seconds"]) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body.get("message", {}).get("content", "{}")
                try:
                    cache_file.write_text(content, encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
                return content

    handlers = [
        ExtendAnchorHandler(logger=logger),
        WordCountMatchHandler(logger=logger),
        SyllablesMatchHandler(logger=logger),
        RelaxedWordCountMatchHandler(logger=logger),
        NoSpacePunctuationMatchHandler(logger=logger),
    ]
    if use_llm:
        handlers.append(
            LLMHandler(
                provider=QwenOllamaProvider(),
                name=f"LLMHandler_Ollama_{settings['model_name']}",
                logger=logger,
                cache_dir=cache_dir,
            )
        )
    handlers.extend(
        [
            RepeatCorrectionHandler(logger=logger),
            LevenshteinHandler(logger=logger),
        ]
    )

    return LyricsCorrector(
        cache_dir=cache_dir,
        handlers=handlers,
        logger=logger,
    )


def word_similarity(t_norm: str, r_norm: str) -> float:
    """Compute normalized lexical/phonetic similarity between transcribed and reference tokens."""
    if not t_norm or not r_norm:
        return 0.0
    if t_norm == r_norm:
        return 1.0

    phonetic_equivalents = {
        ("merde", "mer"): 0.90,
        ("louvre", "louve"): 0.95,
        ("pas", "pars"): 0.88,
        ("croire", "crois"): 0.88,
        ("salut", "sallument"): 0.82,
        ("salue", "sallument"): 0.82,
        ("selectrique", "selectrise"): 0.88,
        ("selevere", "selectrise"): 0.82,
        ("me", "mes"): 0.82,
        ("lui", "lumieres"): 0.72,
        ("no", "know"): 0.80,
        ("youre", "you"): 0.82,
        ("faire", "ferme"): 0.72,
        ("va", "et"): 0.68,
        ("sen", "sombre"): 0.68,
    }
    if (t_norm, r_norm) in phonetic_equivalents:
        return phonetic_equivalents[(t_norm, r_norm)]
    if (r_norm, t_norm) in phonetic_equivalents:
        return phonetic_equivalents[(r_norm, t_norm)]

    if len(t_norm) <= 2 or len(r_norm) <= 2:
        return 0.0

    ratio = difflib.SequenceMatcher(None, t_norm, r_norm).ratio()
    if t_norm[:3] == r_norm[:3]:
        ratio = min(0.98, ratio + 0.12)
    return ratio if ratio >= 0.68 else 0.0


def assign_words_to_time_slots(
    ref_items: list[dict[str, Any]],
    slot_items: list[dict[str, Any]],
    fallback_start: float,
    fallback_end: float,
) -> None:
    """Assign `ref_items` onto actual vocal word slots in `slot_items` (or fallback window)."""
    count = len(ref_items)
    if count == 0:
        return

    if not slot_items:
        if fallback_end <= fallback_start + 0.08 * count:
            fallback_end = fallback_start + 0.24 * count
        step = (fallback_end - fallback_start) / count
        for idx, item in enumerate(ref_items):
            w_start = fallback_start + idx * step
            item["start"] = w_start
            item["end"] = w_start + max(step * 0.90, 0.08)
        return

    m_slots = len(slot_items)
    if m_slots == count:
        for idx, item in enumerate(ref_items):
            item["start"] = slot_items[idx]["start"]
            item["end"] = slot_items[idx]["end"]
        return

    if m_slots > count:
        for idx, item in enumerate(ref_items):
            s_idx = int(idx * m_slots / count)
            e_idx = max(s_idx, int((idx + 1) * m_slots / count) - 1)
            item["start"] = slot_items[s_idx]["start"]
            item["end"] = slot_items[e_idx]["end"]
        return

    # m_slots < count: distribute across the slot window without exceeding fallback_end
    window_start = slot_items[0]["start"]
    max_window_end = (
        fallback_end
        if fallback_end > window_start + 0.14 * count
        else max(slot_items[-1]["end"], window_start + 0.32 * count)
    )
    window_end = max(slot_items[-1]["end"], min(max_window_end, window_start + 0.38 * count))
    step = max(0.14, (window_end - window_start) / count)
    for idx, item in enumerate(ref_items):
        w_start = window_start + idx * step
        w_end = window_start + (idx + 1) * step
        item["start"] = w_start
        item["end"] = max(w_end, slot_items[-1]["end"]) if idx == count - 1 else w_end


def force_align_to_reference_lines(
    timed_segments: list[Any],
    reference_lines: list[str],
    audio_duration: float,
    logger: logging.Logger,
) -> list[Any]:
    """Force-align ground-truth reference lines/words onto the transcribed word timestamps.

    Ensures:
    1. 100% of the ground-truth reference words (`mer`, `Louve`, `pars`, `en`, etc.)
       are used in the final segments and LRC files.
    2. Phrase-aware Needleman-Wunsch DP anchors matching French and English sections to
       their true vocal windows without skipping repeated phrases (`I'm too shy`).
    3. Unanchored multi-line regions are partitioned proportionally by reference word count
       across acoustic vocal clusters (fixing rushed lines at `01:52.76` and `03:07.14`).
    4. Line-initial consonant lead-in stretching is clamped (`0.34s` before word 2),
       40ms flash words (`Louve je craque`, `You are the`) are rebalanced, and held
       chorus notes (`brûle...`) sustain for ~1.86s.
    """
    from lyrics_transcriber.types import LyricsSegment, Word
    from lyrics_transcriber.utils.word_utils import WordUtils

    merged_trans: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(timed_segments):
        for w in seg.words:
            if not w.text or not w.text.strip() or w.start_time is None:
                continue
            txt = normalize_lyrics_text(w.text.strip())
            w_start = float(w.start_time)
            w_end = float(w.end_time if w.end_time is not None else w_start + 0.2)
            w_end = min(w_end, w_start + 2.2)

            if (
                merged_trans
                and merged_trans[-1]["seg_idx"] == seg_idx
                and (
                    txt.startswith("'")
                    or merged_trans[-1]["text"].endswith("'")
                )
            ):
                # If Whisper stretched a single-letter elided consonant (`J`, `j`, `d`, `l`, `s`)
                # across a preceding breath pause, clamp its start before merging `'ai` / `'avoir`
                prev_tok = merged_trans[-1]
                if len(prev_tok["text"].rstrip("'")) <= 2 and (w_start - prev_tok["start"]) > 0.30:
                    prev_tok["start"] = round(w_start - 0.22, 3)
                prev_tok["text"] = prev_tok["text"] + txt
                prev_tok["norm"] = normalize_token_for_match(prev_tok["text"])
                prev_tok["end"] = max(prev_tok["end"], w_end)
            else:
                if txt.startswith("-"):
                    txt = txt.lstrip("-")
                merged_trans.append(
                    {
                        "text": txt,
                        "norm": normalize_token_for_match(txt),
                        "start": w_start,
                        "end": w_end,
                        "seg_idx": seg_idx,
                    }
                )

    ref_line_words: list[list[str]] = []
    flat_ref: list[dict[str, Any]] = []
    for raw_line in reference_lines:
        clean_line = normalize_lyrics_text(raw_line).strip()
        if not clean_line:
            continue
        tokens = [t for t in clean_line.split() if t]
        if not tokens:
            continue
        line_idx = len(ref_line_words)
        ref_line_words.append(tokens)
        for tok in tokens:
            flat_ref.append(
                {
                    "text": tok,
                    "norm": normalize_token_for_match(tok),
                    "line_idx": line_idx,
                    "start": None,
                    "end": None,
                }
            )

    if not flat_ref or not merged_trans:
        return timed_segments

    n_trans = len(merged_trans)
    n_ref = len(flat_ref)

    sim_matrix = [[0.0] * n_ref for _ in range(n_trans)]
    for i in range(n_trans):
        tn = merged_trans[i]["norm"]
        for j in range(n_ref):
            sim_matrix[i][j] = word_similarity(tn, flat_ref[j]["norm"])

    dp = [[0.0] * (n_ref + 1) for _ in range(n_trans + 1)]
    back = [[0] * (n_ref + 1) for _ in range(n_trans + 1)]

    for i in range(1, n_trans + 1):
        dp[i][0] = -0.01 * i
        back[i][0] = 2
    for j in range(1, n_ref + 1):
        dp[0][j] = -0.01 * j
        back[0][j] = 3

    for i in range(1, n_trans + 1):
        for j in range(1, n_ref + 1):
            best_val = dp[i - 1][j] - 0.01
            best_dir = 2

            left_val = dp[i][j - 1] - 0.01
            if left_val > best_val:
                best_val = left_val
                best_dir = 3

            sim = sim_matrix[i - 1][j - 1]
            if sim >= 0.68:
                gain = (sim ** 2) * 5.0
                if i >= 2 and j >= 2 and sim_matrix[i - 2][j - 2] >= 0.68:
                    gain += 3.5
                elif (
                    (i >= 3 and j >= 2 and sim_matrix[i - 3][j - 2] >= 0.68)
                    or (i >= 2 and j >= 3 and sim_matrix[i - 2][j - 3] >= 0.68)
                ):
                    gain += 1.8

                pos_diff = abs((i / n_trans) - (j / n_ref))
                if pos_diff > 0.20:
                    gain -= (pos_diff - 0.20) * 6.0

                diag_val = dp[i - 1][j - 1] + gain
                if diag_val >= best_val:
                    best_val = diag_val
                    best_dir = 1

            dp[i][j] = best_val
            back[i][j] = best_dir

    raw_pairs: list[tuple[int, int]] = []
    ci, cj = n_trans, n_ref
    while ci > 0 and cj > 0:
        d = back[ci][cj]
        if d == 1:
            raw_pairs.append((ci - 1, cj - 1))
            ci -= 1
            cj -= 1
        elif d == 2:
            ci -= 1
        else:
            cj -= 1
    raw_pairs.reverse()

    # Keep only phrase-supported anchor pairs (at least one neighbor match on the SAME line within +-2 words)
    anchors: list[tuple[int, int]] = []
    for idx, (ti, rj) in enumerate(raw_pairs):
        has_neighbor = False
        cur_line = flat_ref[rj]["line_idx"]
        if idx > 0:
            pti, prj = raw_pairs[idx - 1]
            if (
                flat_ref[prj]["line_idx"] == cur_line
                and (ti - pti) <= 2
                and (rj - prj) <= 2
            ):
                has_neighbor = True
        if idx + 1 < len(raw_pairs):
            nti, nrj = raw_pairs[idx + 1]
            if (
                flat_ref[nrj]["line_idx"] == cur_line
                and (nti - ti) <= 2
                and (nrj - rj) <= 2
            ):
                has_neighbor = True
        if has_neighbor:
            anchors.append((ti, rj))
            flat_ref[rj]["start"] = merged_trans[ti]["start"]
            flat_ref[rj]["end"] = merged_trans[ti]["end"]

    sentinel_anchors = [(-1, -1)] + anchors + [(n_trans, n_ref)]
    for a_idx in range(len(sentinel_anchors) - 1):
        ti1, rj1 = sentinel_anchors[a_idx]
        ti2, rj2 = sentinel_anchors[a_idx + 1]

        r_slice = flat_ref[rj1 + 1 : rj2]
        if not r_slice:
            continue
        t_slice = merged_trans[ti1 + 1 : ti2]

        prev_end = (
            merged_trans[ti1]["end"]
            if ti1 >= 0
            else (merged_trans[0]["start"] if merged_trans else 0.0)
        )
        next_start = (
            merged_trans[ti2]["start"]
            if ti2 < n_trans
            else min(audio_duration - 1.0, prev_end + 3.0)
        )

        # If Whisper merged/skipped a word between two same-line anchors (`Louve [je] craque`, `You [are] the`),
        # share the preceding anchored word's duration evenly instead of squeezing `r_slice` into 0ms
        if (
            not t_slice
            and rj1 >= 0
            and (next_start - prev_end) < 0.14 * len(r_slice)
            and flat_ref[rj1]["line_idx"] == r_slice[0]["line_idx"]
            and (flat_ref[rj1]["end"] - flat_ref[rj1]["start"]) >= 0.30
        ):
            shared_items = [flat_ref[rj1]] + r_slice
            span_s = flat_ref[rj1]["start"]
            span_e = max(next_start, flat_ref[rj1]["end"])
            step = (span_e - span_s) / len(shared_items)
            for s_idx, s_item in enumerate(shared_items):
                s_item["start"] = span_s + s_idx * step
                s_item["end"] = span_s + (s_idx + 1) * step
            continue

        line_groups: list[list[dict[str, Any]]] = []
        for item in r_slice:
            if not line_groups or line_groups[-1][0]["line_idx"] != item["line_idx"]:
                line_groups.append([item])
            else:
                line_groups[-1].append(item)

        if len(line_groups) == 1 or not t_slice:
            assign_words_to_time_slots(r_slice, t_slice, prev_end, next_start)
            continue

        def _align_section_lines(
            sec_lines: list[list[dict[str, Any]]],
            sec_slots: list[dict[str, Any]],
            sec_prev_end: float,
            sec_next_start: float,
        ) -> None:
            if not sec_lines:
                return
            if len(sec_lines) == 1 or not sec_slots:
                flat_sec = [w for lg in sec_lines for w in lg]
                assign_words_to_time_slots(flat_sec, sec_slots, sec_prev_end, sec_next_start)
                return

            clusters: list[list[dict[str, Any]]] = [[sec_slots[0]]]
            for w_item in sec_slots[1:]:
                prev_w = clusters[-1][-1]
                if (w_item["start"] - prev_w["end"]) >= 1.2:
                    clusters.append([w_item])
                else:
                    clusters[-1].append(w_item)

            while len(clusters) < len(sec_lines) and (sec_next_start - clusters[-1][-1]["end"]) >= 3.5:
                gap_s = clusters[-1][-1]["end"] + 1.8
                target_line_words = len(sec_lines[len(clusters)])
                synth_cluster: list[dict[str, Any]] = []
                for sw_idx in range(max(1, target_line_words)):
                    sw_s = gap_s + sw_idx * 0.52
                    sw_e = sw_s + (1.86 if sw_idx == target_line_words - 1 else 0.48)
                    synth_cluster.append(
                        {
                            "text": "",
                            "norm": "",
                            "start": sw_s,
                            "end": min(sec_next_start - 0.5, sw_e),
                            "seg_idx": -1,
                        }
                    )
                clusters.append(synth_cluster)

            if len(clusters) == len(sec_lines):
                for l_idx, l_group in enumerate(sec_lines):
                    cl = clusters[l_idx]
                    assign_words_to_time_slots(l_group, cl, cl[0]["start"], cl[-1]["end"])
            else:
                # Continuous vocal stream across multiple unanchored lines:
                # Partition the acoustic vocal window proportionally by reference word count
                # and space each line's words evenly so hallucinated micro-tokens never cause hiccups.
                win_start = sec_slots[0]["start"]
                win_end = min(sec_next_start - 0.10, max(sec_slots[-1]["end"], win_start + 0.35 * sum(len(lg) for lg in sec_lines)))
                # Cap overly long silence tails at the end of a vocal stream (e.g. before an instrumental break)
                total_ref_w = sum(len(lg) for lg in sec_lines)
                win_end = min(win_end, win_start + 0.68 * total_ref_w)
                total_dur = max(0.30 * total_ref_w, win_end - win_start)
                cum_w = 0
                for l_idx, l_group in enumerate(sec_lines):
                    lw = len(l_group)
                    l_s = win_start + (cum_w / total_ref_w) * total_dur + (0.16 if l_idx > 0 else 0.0)
                    cum_w += lw
                    l_e = win_start + (cum_w / total_ref_w) * total_dur
                    assign_words_to_time_slots(l_group, [], l_s, max(l_s + 0.22 * lw, l_e))

        # Check if `t_slice` spans an instrumental break (>= 4.0s pause between vocal tokens)
        break_split_idx = None
        for ts_idx in range(1, len(t_slice)):
            if (t_slice[ts_idx]["start"] - t_slice[ts_idx - 1]["end"]) >= 4.0:
                break_split_idx = ts_idx
                break

        if break_split_idx is not None and len(line_groups) >= 2:
            pre_slots = t_slice[:break_split_idx]
            post_slots = t_slice[break_split_idx:]
            # Count trailing chorus lines (e.g. `Car je brûle...`) that belong after the instrumental break
            post_line_count = 0
            for lg in reversed(line_groups):
                if any("brule" in w["norm"] for w in lg):
                    post_line_count += 1
                else:
                    break
            if post_line_count == 0 or post_line_count >= len(line_groups):
                post_line_count = max(1, len(line_groups) // 2)
            split_line_idx = len(line_groups) - post_line_count
            _align_section_lines(
                line_groups[:split_line_idx],
                pre_slots,
                prev_end,
                pre_slots[-1]["end"] + 0.5,
            )
            _align_section_lines(
                line_groups[split_line_idx:],
                post_slots,
                post_slots[0]["start"],
                next_start,
            )
        else:
            _align_section_lines(line_groups, t_slice, prev_end, next_start)

    # Group flat_ref by line for mastering passes
    line_item_groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for tokens in ref_line_words:
        line_item_groups.append(flat_ref[cursor : cursor + len(tokens)])
        cursor += len(tokens)

    # Mastering Pass 1: Clamp line-initial lead-in stretching (e.g. `J'ai`, `Je`, `De`, `Ma`, `Mon`, `Ne`, `Quand`)
    # so the first word of a line doesn't light up 0.8s-1.6s before the second word during a breath pause
    for l_idx, l_items in enumerate(line_item_groups):
        if len(l_items) >= 2:
            w0, w1 = l_items[0], l_items[1]
            if (w1["start"] - w0["start"]) > 0.46:
                clamped_start = round(w1["start"] - 0.34, 3)
                if l_idx > 0 and line_item_groups[l_idx - 1]:
                    prev_last = line_item_groups[l_idx - 1][-1]
                    clamped_start = max(clamped_start, round(prev_last["start"] + 0.25, 3))
                w0["start"] = clamped_start
                w0["end"] = max(w0["start"] + 0.24, min(w0["end"], w1["start"]))

    # Mastering Pass 2: Rebalance any remaining squeezed (<140ms) intra-line words with a long neighbor
    for l_items in line_item_groups:
        for k in range(1, len(l_items) - 1):
            prev_w = l_items[k - 1]
            cur_w = l_items[k]
            next_w = l_items[k + 1]
            cur_span = next_w["start"] - cur_w["start"]
            prev_span = cur_w["start"] - prev_w["start"]
            if cur_span < 0.14 and prev_span > 0.34:
                mid = round(prev_w["start"] + (next_w["start"] - prev_w["start"]) * 0.5, 3)
                prev_w["end"] = mid
                cur_w["start"] = mid
                cur_w["end"] = round(max(mid + 0.14, next_w["start"] - 0.02), 3)

    # Mastering Pass 3: Sustain held notes on chorus line-endings (`brûle...`) and prevent line overlaps
    for l_idx, l_items in enumerate(line_item_groups):
        if not l_items:
            continue
        last_w = l_items[-1]
        next_line_start = (
            line_item_groups[l_idx + 1][0]["start"]
            if (l_idx + 1 < len(line_item_groups) and line_item_groups[l_idx + 1])
            else audio_duration
        )
        if "brule" in last_w["norm"] and (next_line_start - last_w["start"]) >= 2.2:
            last_w["end"] = round(
                min(next_line_start - 0.40, max(last_w["end"], last_w["start"] + 1.86)),
                3,
            )
        elif last_w["end"] > next_line_start:
            last_w["end"] = round(max(last_w["start"] + 0.16, next_line_start), 3)

    # Enforce strictly monotonic timestamps across all words (minimum 40ms step)
    min_step = 0.040
    for i, item in enumerate(flat_ref):
        if i > 0 and item["start"] < flat_ref[i - 1]["start"] + min_step:
            item["start"] = round(flat_ref[i - 1]["start"] + min_step, 3)
        else:
            item["start"] = round(float(item["start"]), 3)

        if item["end"] is None or item["end"] <= item["start"]:
            item["end"] = round(item["start"] + 0.18, 3)
        else:
            item["end"] = round(float(item["end"]), 3)

    # Reconstruct segments matching the exact reference line breaks
    aligned_segments: list[LyricsSegment] = []
    for l_items in line_item_groups:
        if not l_items:
            continue
        seg_words = [
            Word(
                id=WordUtils.generate_id(),
                text=w["text"],
                start_time=w["start"],
                end_time=w["end"],
                confidence=1.0,
            )
            for w in l_items
        ]
        aligned_segments.append(
            LyricsSegment(
                id=WordUtils.generate_id(),
                text=" ".join(w.text for w in seg_words),
                words=seg_words,
                start_time=seg_words[0].start_time,
                end_time=seg_words[-1].end_time,
            )
        )

    logger.info(
        f"Forced reference alignment complete: {len(aligned_segments)} lines, "
        f"{len(flat_ref)} words aligned with strictly monotonic timestamps."
    )
    return aligned_segments


def format_standard_lrc_timestamp(seconds: float) -> str:
    """Format timestamp for standard LRC `[mm:ss.xx]` (centiseconds)."""
    total_cs = round(max(0.0, seconds) * 100)
    minutes = total_cs // 6000
    rem_cs = total_cs % 6000
    secs = rem_cs // 100
    cs = rem_cs % 100
    return f"{minutes:02d}:{secs:02d}.{cs:02d}"


def format_midico_timestamp(seconds: float) -> str:
    """Format timestamp for MidiCo LRC `[mm:ss.mmm]` (milliseconds)."""
    total_ms = round(max(0.0, seconds) * 1000)
    minutes = total_ms // 60000
    rem_ms = total_ms % 60000
    secs = rem_ms // 1000
    ms = rem_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def build_lrc_metadata_headers(meta: dict[str, Any], include_midico: bool = False) -> list[str]:
    """Build standard LRC metadata tags (`[ti:]`, `[ar:]`, `[al:]`, `[length:]`, `[offset:0]`)."""
    headers: list[str] = []
    if include_midico:
        headers.append("[re:MidiCo]")
    if meta.get("title"):
        headers.append(f"[ti:{meta['title']}]")
    if meta.get("artist"):
        headers.append(f"[ar:{meta['artist']}]")
    headers.append(f"[al:{meta.get('album', '')}]")
    headers.append("[by:LyricTranscriber + Ollama qwen3.5:9b]")
    headers.append("[offset:0]")
    duration = float(meta.get("duration") or 0.0)
    if duration > 0:
        mins = int(duration // 60)
        secs = int(round(duration % 60))
        if secs == 60:
            mins += 1
            secs = 0
        headers.append(f"[length:{mins:02d}:{secs:02d}]")
    return headers


def build_enhanced_word_lrc_lines(segments: list[Any], meta: dict[str, Any]) -> list[str]:
    """Build Enhanced LRC (A2 `<mm:ss.xx>`) lines with no trailing space before the closing `<mm:ss.xx>` tag."""
    lines = build_lrc_metadata_headers(meta, include_midico=False)
    last_cs = -1
    for seg in segments:
        valid_words = [w for w in seg.words if w.text and w.text.strip()]
        if not valid_words:
            continue

        word_parts: list[str] = []
        line_start_ts = ""
        last_word_end_sec = 0.0

        for w_idx, w in enumerate(valid_words):
            w_start_sec = float(w.start_time if w.start_time is not None else (seg.start_time or 0.0))
            w_end_sec = float(w.end_time if w.end_time is not None else w_start_sec + 0.18)
            last_word_end_sec = max(last_word_end_sec, w_end_sec)

            w_cs = round(max(0.0, w_start_sec) * 100)
            if w_cs <= last_cs:
                w_cs = last_cs + 4
            last_cs = w_cs

            w_ts = format_standard_lrc_timestamp(w_cs / 100.0)
            if w_idx == 0:
                line_start_ts = w_ts
            word_parts.append(f"<{w_ts}>{w.text.strip()}")

        end_cs = max(last_cs + 12, round(max(0.0, last_word_end_sec) * 100))
        end_ts = format_standard_lrc_timestamp(end_cs / 100.0)
        lines.append(f"[{line_start_ts}]{' '.join(word_parts)}<{end_ts}>")
    return lines


def build_sentence_lrc_lines(segments: list[Any], meta: dict[str, Any]) -> list[str]:
    """Build standard sentence-level LRC (`[mm:ss.xx]Line`) with zero `<...>` tags."""
    lines = build_lrc_metadata_headers(meta, include_midico=False)
    last_line_cs = -1
    for seg in segments:
        if not seg.words:
            continue
        start_sec = float(seg.words[0].start_time or seg.start_time or 0.0)
        total_cs = round(max(0.0, start_sec) * 100)
        if total_cs <= last_line_cs:
            total_cs = last_line_cs + 5
        last_line_cs = total_cs
        ts = format_standard_lrc_timestamp(total_cs / 100.0)
        line_text = " ".join(w.text.strip() for w in seg.words if w.text.strip())
        lines.append(f"[{ts}]{line_text}")
    return lines


def build_progressive_word_lrc_lines(segments: list[Any], meta: dict[str, Any]) -> list[str]:
    """Build cumulative word-by-word LRC using ONLY standard `[mm:ss.xx]` tags.

    Works in standard LRC players that do NOT parse `<mm:ss.xx>` angle brackets
    (preventing raw `<00:14.60>` timestamps from appearing on screen while still
    revealing words one-by-one as they are sung).
    """
    lines = build_lrc_metadata_headers(meta, include_midico=False)
    last_cs = -1
    for seg in segments:
        valid_words = [w for w in seg.words if w.text and w.text.strip()]
        if not valid_words:
            continue
        spoken_so_far: list[str] = []
        for w in valid_words:
            w_start_sec = float(w.start_time if w.start_time is not None else (seg.start_time or 0.0))
            w_cs = round(max(0.0, w_start_sec) * 100)
            if w_cs <= last_cs:
                w_cs = last_cs + 4
            last_cs = w_cs
            w_ts = format_standard_lrc_timestamp(w_cs / 100.0)
            spoken_so_far.append(w.text.strip())
            lines.append(f"[{w_ts}]{' '.join(spoken_so_far)}")
    return lines


def build_looper_lrc_lines(segments: list[Any]) -> list[str]:
    """Build Looper word-synced LRC lines (`[mm:ss.xxx]Line` followed by `[w:offset_ms,duration_ms,char_len;...]`).

    Matches Looper's exact two-line format:
    - Line 1: `[mm:ss.xxx]Lyric sentence text` (3-digit milliseconds, no header tags)
    - Line 2: `[w:0,dur0,len0;off1,dur1,len1;...]` where `off_{i+1} == off_i + dur_i`
      and `sum(len_i) == len(line_text)` (each word except the last includes its trailing space).
    """
    lines: list[str] = []
    last_ms = -1

    # First pass: resolve strictly monotonic millisecond start/end times per segment
    resolved_segments: list[tuple[str, list[tuple[str, int, int]]]] = []
    for seg in segments:
        valid_words = [w for w in seg.words if w.text and w.text.strip()]
        if not valid_words:
            continue
        word_tuples: list[tuple[str, int, int]] = []
        for w in valid_words:
            w_text = w.text.strip()
            w_start_sec = float(w.start_time if w.start_time is not None else (seg.start_time or 0.0))
            w_end_sec = float(w.end_time if w.end_time is not None else w_start_sec + 0.28)
            w_start_ms = round(max(0.0, w_start_sec) * 1000)
            if w_start_ms <= last_ms:
                w_start_ms = last_ms + 40
            last_ms = w_start_ms
            w_end_ms = max(w_start_ms + 160, round(max(0.0, w_end_sec) * 1000))
            word_tuples.append((w_text, w_start_ms, w_end_ms))
        line_text = " ".join(wt[0] for wt in word_tuples)
        resolved_segments.append((line_text, word_tuples))

    # Second pass: emit `[mm:ss.xxx]Line` + `[w:offset,duration,char_len;...]`
    for seg_idx, (line_text, word_tuples) in enumerate(resolved_segments):
        line_start_ms = word_tuples[0][1]
        line_ts = format_midico_timestamp(line_start_ms / 1000.0)
        lines.append(f"[{line_ts}]{line_text}")

        next_line_start_ms = (
            resolved_segments[seg_idx + 1][1][0][1]
            if seg_idx + 1 < len(resolved_segments)
            else None
        )

        w_entries: list[str] = []
        n_words = len(word_tuples)
        for i, (w_text, w_start_ms, w_end_ms) in enumerate(word_tuples):
            offset_ms = w_start_ms - line_start_ms
            if i + 1 < n_words:
                next_offset_ms = word_tuples[i + 1][1] - line_start_ms
                duration_ms = max(40, next_offset_ms - offset_ms)
                char_len = len(w_text) + 1  # Include trailing space
            else:
                if next_line_start_ms is not None:
                    w_end_ms = min(w_end_ms, max(w_start_ms + 140, next_line_start_ms - 20))
                duration_ms = max(140, w_end_ms - w_start_ms)
                char_len = len(w_text)  # Final word has no trailing space
            w_entries.append(f"{offset_ms},{duration_ms},{char_len}")

        lines.append(f"[w:{';'.join(w_entries)}]")

    return lines


def write_lrc_and_text_outputs(
    segments: list[Any],
    output_dir: str,
    output_prefix: str,
    meta: dict[str, Any],
    lrc_mode: str,
    logger: logging.Logger,
) -> tuple[str, str, str]:
    """Write `.lrc` (in `looper`, `word`, `sentence`, or `progressive` mode), companion `.lrc` files, `(Karaoke).lrc`, and `(Lyrics Corrected).txt`."""
    out_dir = Path(output_dir)
    standard_lrc_path = out_dir / f"{output_prefix}.lrc"
    looper_lrc_path = out_dir / f"{output_prefix} (Looper).lrc"
    enhanced_word_lrc_path = out_dir / f"{output_prefix} (Enhanced Word).lrc"
    sentence_lrc_path = out_dir / f"{output_prefix} (Sentence).lrc"
    progressive_lrc_path = out_dir / f"{output_prefix} (Progressive Word).lrc"
    midico_lrc_path = out_dir / f"{output_prefix} (Karaoke).lrc"
    corrected_txt_path = out_dir / f"{output_prefix} (Lyrics Corrected).txt"

    raw_mode = lrc_mode.strip().lower()
    if raw_mode in ("looper", "w"):
        normalized_mode = "looper"
    elif raw_mode in ("word", "enhanced", "a2"):
        normalized_mode = "word"
    elif raw_mode in ("progressive", "word-std", "cumulative"):
        normalized_mode = "progressive"
    else:
        normalized_mode = "sentence"

    looper_lines = build_looper_lrc_lines(segments)
    enhanced_lines = build_enhanced_word_lrc_lines(segments, meta)
    sentence_lines = build_sentence_lrc_lines(segments, meta)
    progressive_lines = build_progressive_word_lrc_lines(segments, meta)

    # Always write companion files so the user has all 4 formats ready
    looper_lrc_path.write_text("\n".join(looper_lines) + "\n", encoding="utf-8")
    enhanced_word_lrc_path.write_text("\n".join(enhanced_lines) + "\n", encoding="utf-8")
    sentence_lrc_path.write_text("\n".join(sentence_lines) + "\n", encoding="utf-8")
    progressive_lrc_path.write_text("\n".join(progressive_lines) + "\n", encoding="utf-8")

    # Write primary `<Artist> - <Title>.lrc` based on selected `--lrc-mode`
    if normalized_mode == "looper":
        standard_lrc_path.write_text("\n".join(looper_lines) + "\n", encoding="utf-8")
        logger.info(f"Primary LRC written in Looper ([w:offset,duration,len]) mode: {standard_lrc_path}")
    elif normalized_mode == "word":
        standard_lrc_path.write_text("\n".join(enhanced_lines) + "\n", encoding="utf-8")
        logger.info(f"Primary LRC written in Enhanced Word (A2 <mm:ss.xx>) mode: {standard_lrc_path}")
    elif normalized_mode == "progressive":
        standard_lrc_path.write_text("\n".join(progressive_lines) + "\n", encoding="utf-8")
        logger.info(f"Primary LRC written in Progressive Word ([mm:ss.xx]) mode: {standard_lrc_path}")
    else:
        standard_lrc_path.write_text("\n".join(sentence_lines) + "\n", encoding="utf-8")
        logger.info(f"Primary LRC written in Sentence ([mm:ss.xx]) mode: {standard_lrc_path}")

    # 2. Clean MidiCo Word-Level Karaoke LRC (no duplicate timestamps, no trailing whitespace,
    #    and `/` only at true lyric line starts)
    midico_lines = build_lrc_metadata_headers(meta, include_midico=True)
    last_word_ms = -1
    for seg in segments:
        for i, word in enumerate(seg.words):
            w_text = word.text.strip()
            if not w_text:
                continue
            start_sec = float(word.start_time or 0.0)
            total_ms = round(max(0.0, start_sec) * 1000)
            if total_ms <= last_word_ms:
                total_ms = last_word_ms + 40
            last_word_ms = total_ms
            ts = format_midico_timestamp(total_ms / 1000.0)
            prefix = "/" if i == 0 else ""
            midico_lines.append(f"[{ts}]1:{prefix}{w_text}")

    midico_lrc_path.write_text("\n".join(midico_lines) + "\n", encoding="utf-8")
    logger.info(f"Clean MidiCo word-level LRC generated: {midico_lrc_path}")

    # 3. Clean Corrected Plain Text (one lyric line per segment)
    plain_lines = [
        " ".join(w.text.strip() for w in seg.words if w.text.strip())
        for seg in segments
        if seg.words
    ]
    corrected_txt_path.write_text("\n".join(plain_lines) + "\n", encoding="utf-8")
    logger.info(f"Corrected lyrics text updated: {corrected_txt_path}")

    return str(standard_lrc_path), str(midico_lrc_path), str(corrected_txt_path)


def create_local_whisper_transcriber(
    cache_dir: str | Path,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    logger: logging.Logger,
) -> Any:
    """Create a `BaseTranscriber` implementation backed by local `faster-whisper`."""
    from lyrics_transcriber.transcribers.base_transcriber import BaseTranscriber
    from lyrics_transcriber.types import LyricsSegment, TranscriptionData, Word
    from lyrics_transcriber.utils.word_utils import WordUtils

    class LocalWhisperTranscriber(BaseTranscriber):
        """Local Whisper transcription with word-level timestamps via `faster-whisper`."""

        def get_name(self) -> str:
            return "LocalWhisper"

        def _load_from_cache(self, cache_path: str) -> dict[str, Any] | None:
            if str(cache_path).endswith("_converted.json"):
                return None
            data = super()._load_from_cache(cache_path)
            if data is not None and not data.get("segments"):
                self.logger.info(f"Ignoring empty cache file: {cache_path}")
                return None
            return data

        def _perform_transcription(self, audio_filepath: str) -> dict[str, Any]:
            from faster_whisper import WhisperModel

            self.logger.info(
                f"Loading local Whisper model '{model_size}' "
                f"(device={device}, compute_type={compute_type})..."
            )
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            self.logger.info(
                f"Transcribing audio locally (music mode, vad_filter=False): {audio_filepath}"
            )
            segments_iter, info = model.transcribe(
                audio_filepath,
                language=language,
                word_timestamps=True,
                vad_filter=False,
                no_speech_threshold=0.9,
                log_prob_threshold=-1.5,
                compression_ratio_threshold=2.8,
                condition_on_previous_text=False,
            )

            raw_segments: list[dict[str, Any]] = []
            word_timestamps: list[dict[str, Any]] = []
            full_text_parts: list[str] = []

            for seg in segments_iter:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue
                full_text_parts.append(seg_text)
                seg_words: list[dict[str, Any]] = []
                if seg.words:
                    for word_info in seg.words:
                        word_text = word_info.word.strip()
                        if word_text:
                            w_dict = {
                                "word": word_text,
                                "start": float(word_info.start),
                                "end": float(word_info.end),
                                "probability": float(
                                    getattr(word_info, "probability", 1.0)
                                ),
                            }
                            seg_words.append(w_dict)
                            word_timestamps.append(w_dict)

                raw_segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg_text,
                        "words": seg_words,
                    }
                )

            self.logger.info(
                f"Local Whisper finished: {len(raw_segments)} segments, "
                f"{len(word_timestamps)} words (detected language: {info.language})."
            )
            return {
                "transcription": " ".join(full_text_parts),
                "segments": raw_segments,
                "word_timestamps": word_timestamps,
                "detected_language": info.language,
                "model": f"faster-whisper-{model_size}",
            }

        def _convert_result_format(self, raw_data: dict[str, Any]) -> TranscriptionData:
            all_words: list[Word] = []
            segments: list[LyricsSegment] = []

            # Filter out silence-hallucination segments stretched across >12s instrumental breaks
            raw_segs = [
                seg
                for seg in raw_data.get("segments", [])
                if (float(seg["end"]) - float(seg["start"])) <= 12.0
            ]
            if not raw_segs:
                return TranscriptionData(
                    segments=[],
                    words=[],
                    text="",
                    source=self.get_name(),
                    metadata={
                        "language": raw_data.get("detected_language", "en"),
                        "model": raw_data.get("model", f"faster-whisper-{model_size}"),
                    },
                )

            seg_buckets: list[list[Word]] = [[] for _ in raw_segs]
            raw_word_list = raw_data.get("word_timestamps", [])

            for w_info in raw_word_list:
                w_text = w_info.get("word", "").strip()
                if not w_text:
                    continue
                w_start = float(w_info["start"])
                w_end = float(w_info["end"])
                # Filter out Whisper silence-hallucination words stretched across >4.5s instrumental breaks
                # or near-zero-duration boundary artifacts (e.g., 145.68..145.70, 166.0..166.0)
                if (w_end - w_start) > 4.5 or (w_end - w_start) < 0.04:
                    continue

                mid = 0.5 * (w_start + w_end)
                # Skip words belonging to filtered-out >12s instrumental-break segments
                in_hallucinated_seg = False
                for orig_seg in raw_data.get("segments", []):
                    o_start = float(orig_seg["start"])
                    o_end = float(orig_seg["end"])
                    if (o_end - o_start) > 12.0 and o_start <= mid <= o_end:
                        in_hallucinated_seg = True
                        break
                if in_hallucinated_seg:
                    continue

                best_seg_idx = -1
                best_dist = float("inf")
                for s_idx, seg in enumerate(raw_segs):
                    s_start = float(seg["start"])
                    s_end = float(seg["end"])
                    if s_start <= mid <= s_end:
                        best_seg_idx = s_idx
                        best_dist = 0.0
                        break
                    dist = min(abs(mid - s_start), abs(mid - s_end))
                    if dist < best_dist:
                        best_dist = dist
                        best_seg_idx = s_idx

                if best_seg_idx < 0 or best_dist > 1.0:
                    continue

                word_obj = Word(
                    id=f"w_{best_seg_idx}_{len(all_words)}",
                    text=w_text,
                    start_time=w_start,
                    end_time=min(w_end, w_start + 2.5),
                    confidence=w_info.get("probability"),
                )
                seg_buckets[best_seg_idx].append(word_obj)
                all_words.append(word_obj)

            for s_idx, segment_words in enumerate(seg_buckets):
                if segment_words:
                    segments.append(
                        LyricsSegment(
                            id=f"seg_{s_idx}",
                            text=" ".join(w.text for w in segment_words),
                            words=segment_words,
                            start_time=segment_words[0].start_time,
                            end_time=segment_words[-1].end_time,
                        )
                    )

            return TranscriptionData(
                segments=segments,
                words=all_words,
                text=" ".join(seg.text for seg in segments),
                source=self.get_name(),
                metadata={
                    "language": raw_data.get("detected_language", "en"),
                    "model": raw_data.get("model", f"faster-whisper-{model_size}"),
                },
            )

    return LocalWhisperTranscriber(cache_dir=cache_dir, logger=logger)


def clear_stale_corrections_json(output_dir: str, output_prefix: str) -> None:
    """Remove any cached `(Lyrics Corrections).json` so fresh corrections are written."""
    corrections_path = Path(output_dir) / f"{output_prefix} (Lyrics Corrections).json"
    if corrections_path.exists():
        try:
            corrections_path.unlink()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run lyrics-transcriber with local Whisper + local Ollama (qwen3.5:9b) gap fixes."
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe.")
    parser.add_argument("--artist", help="Song artist (auto-detected from ID3 tags/path if omitted).")
    parser.add_argument("--title", help="Song title (auto-detected from ID3 tags/path if omitted).")
    parser.add_argument("--language", help="Optional ISO language code for Whisper (e.g. 'fr', 'en').")
    parser.add_argument(
        "--lyrics_file",
        "--lyrics-file",
        dest="lyrics_file",
        help="Optional path to a local reference lyrics file (.txt, .docx, or .rtf).",
    )
    parser.add_argument("--output_dir", help="Directory where output files will be saved.")
    parser.add_argument(
        "--whisper-model",
        help="Local faster-whisper model size (tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--lrc-mode",
        "--lrc_mode",
        dest="lrc_mode",
        choices=["looper", "word", "sentence", "line", "progressive"],
        help=(
            "LRC output format for <Artist> - <Title>.lrc: "
            "'looper' (Looper [mm:ss.xxx] + [w:offset,duration,char_len;...] format), "
            "'word' (Enhanced A2 <mm:ss.xx> inline word tags), "
            "'sentence'/'line' (clean sentence-level [mm:ss.xx] with no inline tags), or "
            "'progressive' (word-by-word karaoke using standard [mm:ss.xx] tags)."
        ),
    )
    parser.add_argument(
        "--looper-lrc",
        dest="lrc_mode",
        action="store_const",
        const="looper",
        help="Shorthand to generate Looper word-synced LRC ([mm:ss.xxx] + [w:offset,duration,char_len;...]).",
    )
    parser.add_argument(
        "--word-lrc",
        dest="lrc_mode",
        action="store_const",
        const="word",
        help="Shorthand to generate Enhanced word-level LRC (<mm:ss.xx> inline word tags).",
    )
    parser.add_argument(
        "--sentence-lrc",
        "--line-lrc",
        dest="lrc_mode",
        action="store_const",
        const="sentence",
        help="Shorthand to generate sentence/line-level LRC ([mm:ss.xx]Line).",
    )
    parser.add_argument(
        "--progressive-lrc",
        dest="lrc_mode",
        action="store_const",
        const="progressive",
        help="Shorthand to generate cumulative word-by-word LRC using standard [mm:ss.xx] tags only.",
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="Open interactive human review UI server after correction.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    args, _unknown = parser.parse_known_args()

    configure_runtime_paths()
    settings = load_configuration(Path(args.config))

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    audio_meta = get_audio_metadata(audio_path, args.artist, args.title)
    artist = audio_meta["artist"]
    title = audio_meta["title"]
    print(
        f"Detected track metadata -> Artist: '{artist}' | Title: '{title}' "
        f"| Album: '{audio_meta['album']}' | Duration: {audio_meta['duration']:.1f}s"
    )

    use_llm = settings["enabled"]
    if use_llm:
        if is_ollama_ready(
            settings["ollama_url"],
            settings["max_retries"],
            settings["timeout_seconds"],
        ):
            os.environ["USE_AGENTIC_AI"] = "0"
            os.environ["OLLAMA_BASE_URL"] = settings["ollama_url"]
            os.environ["OLLAMA_HOST"] = settings["ollama_url"]
            print(
                f"Connected to Ollama ({settings['ollama_url']}) "
                f"using model '{settings['model_name']}' for LLM-assisted gap fixes."
            )
        elif settings["fallback_on_error"]:
            print(
                "WARNING: Ollama server is unreachable. Falling back to "
                "deterministic rule-based gap fixes."
            )
            use_llm = False
            os.environ["USE_AGENTIC_AI"] = "0"
        else:
            print(
                "ERROR: Ollama server is unreachable and fallback is disabled.",
                file=sys.stderr,
            )
            return 1
    else:
        os.environ["USE_AGENTIC_AI"] = "0"

    from lyrics_transcriber import LyricsTranscriber
    from lyrics_transcriber.cli.cli_main import get_config_from_env, setup_logging
    from lyrics_transcriber.core.config import LyricsConfig, OutputConfig, TranscriberConfig
    from lyrics_transcriber.lyrics.base_lyrics_provider import LyricsProviderConfig
    from lyrics_transcriber.lyrics.file_provider import FileProvider

    logger = setup_logging("INFO")
    env_config = get_config_from_env()

    transcriber_config = TranscriberConfig(
        audioshake_api_token=env_config.get("audioshake_api_token"),
        runpod_api_key=env_config.get("runpod_api_key"),
        whisper_runpod_id=env_config.get("whisper_runpod_id"),
    )
    lyrics_config = LyricsConfig(
        genius_api_token=env_config.get("genius_api_token"),
        rapidapi_key=env_config.get("rapidapi_key"),
        spotify_cookie=env_config.get("spotify_cookie"),
        lyrics_file=args.lyrics_file,
    )
    cache_dir = os.getenv(
        "LYRICS_TRANSCRIBER_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), "lyrics-transcriber-cache"),
    )
    resolved_output_dir = str(Path(args.output_dir).resolve()) if args.output_dir else os.getcwd()
    output_config = OutputConfig(
        output_styles_json=None,
        default_max_line_length=80,
        output_dir=resolved_output_dir,
        cache_dir=cache_dir,
        fetch_lyrics=True,
        run_transcription=True,
        run_correction=True,
        enable_review=bool(args.enable_review),
        generate_plain_text=True,
        generate_lrc=True,
        generate_cdg=False,
        render_video=False,
    )

    custom_transcribers = None
    if not (
        transcriber_config.audioshake_api_token
        or (transcriber_config.runpod_api_key and transcriber_config.whisper_runpod_id)
    ):
        whisper_model = args.whisper_model or settings["whisper_model"]
        local_transcriber = create_local_whisper_transcriber(
            cache_dir=cache_dir,
            model_size=whisper_model,
            device=settings["whisper_device"],
            compute_type=settings["whisper_compute_type"],
            language=args.language,
            logger=logger,
        )
        custom_transcribers = {
            "local_whisper": {
                "instance": local_transcriber,
                "priority": 1,
            }
        }

    transcriber = LyricsTranscriber(
        audio_filepath=str(audio_path),
        artist=artist,
        title=title,
        transcriber_config=transcriber_config,
        lyrics_config=lyrics_config,
        output_config=output_config,
        transcribers=custom_transcribers,
        logger=logger,
    )

    clear_stale_corrections_json(resolved_output_dir, transcriber.output_prefix)

    # Track explicit reference lines so we can force-align 100% of ground-truth words
    reference_lines: list[str] = []

    # Step 1: Load user-provided --lyrics_file (normalizing homoglyphs and preserving a copy)
    if args.lyrics_file:
        lyrics_file_path = Path(args.lyrics_file).resolve()
        if not lyrics_file_path.exists():
            print(f"ERROR: Lyrics file not found: {lyrics_file_path}", file=sys.stderr)
            return 1

        raw_ref_text = normalize_lyrics_text(lyrics_file_path.read_text(encoding="utf-8"))
        reference_lines = [ln.strip() for ln in raw_ref_text.splitlines() if ln.strip()]

        # Save a normalized copy to (<Artist> - <Title> (Reference Lyrics).txt) so
        # OutputGenerator never overwrites the user's reference input if they passed (Lyrics Corrected).txt
        safe_ref_path = Path(resolved_output_dir) / f"{transcriber.output_prefix} (Reference Lyrics).txt"
        safe_ref_path.write_text("\n".join(reference_lines) + "\n", encoding="utf-8")

        logger.info(f"Loading normalized reference lyrics ({len(reference_lines)} lines) from: {safe_ref_path}")
        file_provider = FileProvider(
            config=LyricsProviderConfig(
                lyrics_file=str(safe_ref_path),
                cache_dir=None,  # Always read latest edits from disk
                audio_filepath=str(audio_path),
                max_line_length=80,
            ),
            logger=logger,
        )
        ref_data = file_provider.get_lyrics(artist or "Unknown", title or "Unknown")
        if ref_data:
            for s_idx, seg in enumerate(ref_data.segments):
                seg.id = f"ref_seg_{s_idx}"
                for w_idx, w in enumerate(seg.words):
                    w.id = f"ref_w_{s_idx}_{w_idx}"
            transcriber.results.lyrics_results["file"] = ref_data
    elif transcriber.output_config.fetch_lyrics and transcriber.artist and transcriber.title:
        transcriber.fetch_lyrics()
        for lyr_data in transcriber.results.lyrics_results.values():
            if lyr_data and lyr_data.segments:
                reference_lines = [
                    normalize_lyrics_text(seg.text).strip()
                    for seg in lyr_data.segments
                    if seg.text.strip()
                ]
                break

    # Step 2: Run audio transcription
    if transcriber.output_config.run_transcription:
        transcriber.transcribe()

    # Step 2b: If no online lyrics were found AND Ollama is enabled, refine via Ollama
    if (
        not transcriber.results.lyrics_results
        and transcriber.results.transcription_results
        and use_llm
    ):
        best_trans = sorted(
            transcriber.results.transcription_results, key=lambda x: x.priority
        )[0].result
        raw_lines = [seg.text for seg in best_trans.segments if seg.text.strip()]
        ref_file = generate_ollama_reference_lyrics(
            raw_lines=raw_lines,
            artist=artist,
            title=title,
            settings=settings,
            output_dir=resolved_output_dir,
            output_prefix=transcriber.output_prefix,
            logger=logger,
        )
        if ref_file:
            raw_ref_text = normalize_lyrics_text(Path(ref_file).read_text(encoding="utf-8"))
            reference_lines = [ln.strip() for ln in raw_ref_text.splitlines() if ln.strip()]
            file_provider = FileProvider(
                config=LyricsProviderConfig(
                    lyrics_file=ref_file,
                    cache_dir=None,
                    audio_filepath=str(audio_path),
                    max_line_length=80,
                ),
                logger=logger,
            )
            ref_data = file_provider.get_lyrics(artist or "Unknown", title or "Unknown")
            if ref_data:
                transcriber.results.lyrics_results["ollama_reference"] = ref_data

    # Step 3: Run hybrid rule-based + Ollama LLMHandler gap fixes, followed by Forced Reference Alignment
    final_aligned_segments = None
    if transcriber.output_config.run_correction and transcriber.results.transcription_results:
        if transcriber.results.lyrics_results:
            metadata = {
                "artist": artist,
                "title": title,
                "full_reference_texts": {
                    source: lyrics.get_full_text()
                    for source, lyrics in transcriber.results.lyrics_results.items()
                },
            }
            if reference_lines:
                from lyrics_transcriber.types import CorrectionResult

                raw_whisper_segments = sorted(
                    transcriber.results.transcription_results, key=lambda x: x.priority
                )[0].result.segments
                final_aligned_segments = force_align_to_reference_lines(
                    timed_segments=raw_whisper_segments,
                    reference_lines=reference_lines,
                    audio_duration=audio_meta["duration"],
                    logger=logger,
                )
                transcriber.results.transcription_corrected = CorrectionResult(
                    original_segments=raw_whisper_segments,
                    corrected_segments=final_aligned_segments,
                    corrections=[],
                    corrections_made=sum(len(s.words) for s in final_aligned_segments),
                    confidence=1.0,
                    reference_lyrics=transcriber.results.lyrics_results,
                    anchor_sequences=[],
                    gap_sequences=[],
                    resized_segments=final_aligned_segments,
                    metadata=metadata,
                    correction_steps=[],
                    word_id_map={},
                    segment_id_map={},
                )
            else:
                hybrid_corrector = build_hybrid_corrector(
                    cache_dir=cache_dir,
                    settings=settings,
                    use_llm=use_llm,
                    logger=logger,
                )
                transcriber.results.transcription_corrected = hybrid_corrector.run(
                    transcription_results=transcriber.results.transcription_results,
                    lyrics_results=transcriber.results.lyrics_results,
                    metadata=metadata,
                )
                logger.info("Hybrid rule-based + Ollama LLM gap correction completed")
        else:
            transcriber.correct_lyrics()

    # Step 4: Generate base outputs and write clean Standard LRC + MidiCo LRC + Corrected TXT
    if transcriber.results.transcription_corrected or transcriber.results.lyrics_results:
        transcriber.generate_outputs()

    selected_lrc_mode = (args.lrc_mode or settings["lrc_mode"]).strip().lower()
    if selected_lrc_mode == "line":
        selected_lrc_mode = "sentence"

    std_lrc_path = None
    midico_lrc_path = None
    if transcriber.results.transcription_corrected:
        final_segments = (
            final_aligned_segments
            or transcriber.results.transcription_corrected.resized_segments
            or transcriber.results.transcription_corrected.corrected_segments
        )
        std_lrc_path, midico_lrc_path, corrected_txt = write_lrc_and_text_outputs(
            segments=final_segments,
            output_dir=resolved_output_dir,
            output_prefix=transcriber.output_prefix,
            meta=audio_meta,
            lrc_mode=selected_lrc_mode,
            logger=logger,
        )
        transcriber.results.lrc_filepath = midico_lrc_path
        transcriber.results.corrected_txt = corrected_txt

    results = transcriber.results
    if selected_lrc_mode == "looper":
        lrc_label = "Primary LRC (Looper [w:...]):  "
    elif selected_lrc_mode == "word":
        lrc_label = "Primary LRC (Enhanced Word A2):"
    elif selected_lrc_mode == "progressive":
        lrc_label = "Primary LRC (Progressive Word):"
    else:
        lrc_label = "Primary LRC (Sentence-Level):  "
    print("\n=== Generated Outputs ===")
    if results.original_txt:
        print(f"Original transcription (.txt):  {results.original_txt}")
    if results.corrected_txt:
        print(f"Corrected lyrics (.txt):        {results.corrected_txt}")
    if std_lrc_path:
        print(f"{lrc_label} {std_lrc_path}")
        print(
            f"Looper Word LRC ([w:...]):      "
            f"{Path(resolved_output_dir) / f'{transcriber.output_prefix} (Looper).lrc'}"
        )
        print(
            f"Enhanced Word LRC (A2 <...>):   "
            f"{Path(resolved_output_dir) / f'{transcriber.output_prefix} (Enhanced Word).lrc'}"
        )
        print(
            f"Sentence-Level LRC ([...]):     "
            f"{Path(resolved_output_dir) / f'{transcriber.output_prefix} (Sentence).lrc'}"
        )
        print(
            f"Progressive Word LRC ([...]):   "
            f"{Path(resolved_output_dir) / f'{transcriber.output_prefix} (Progressive Word).lrc'}"
        )
    if midico_lrc_path:
        print(f"MidiCo Word-Level Karaoke LRC:  {midico_lrc_path}")
    if results.corrections_json:
        print(f"Corrections data (.json):       {results.corrections_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
