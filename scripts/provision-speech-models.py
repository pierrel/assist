#!/usr/bin/env python3
"""Provision the immutable local speech models used by manage.voice.speech."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import signal
from urllib.request import urlopen

WHISPER_REVISION = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
WHISPER_BASE_URL = (
    "https://huggingface.co/Systran/faster-whisper-base.en/resolve/"
    f"{WHISPER_REVISION}"
)
WHISPER_FILES = {
    "config.json": (
        "f3bc3821e9fc76a27bae538e11ae5b677dcdd352b4600429ce7951d398569aeb",
        2_227,
    ),
    "model.bin": (
        "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef",
        145_216_508,
    ),
    "tokenizer.json": (
        "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
        2_128_466,
    ),
    "vocabulary.txt": (
        "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
        422_309,
    ),
}
PIPER_VOICE = "en_US-lessac-medium"
PIPER_REVISION = "0d907f158acc877ddeebcbf827659ee13bea8bcd"
PIPER_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/"
    f"{PIPER_REVISION}/en/en_US/lessac/medium/{PIPER_VOICE}.onnx"
)
PIPER_FILES = {
    ".onnx": (
        "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        63_201_294,
    ),
    ".onnx.json": (
        "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
        4_885,
    ),
}


def _download(
    url: str, destination: Path, expected_sha256: str, expected_size: int
) -> None:
    if destination.exists() and destination.stat().st_size == expected_size:
        digest = hashlib.sha256()
        with destination.open("rb") as existing:
            while chunk := existing.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() == expected_sha256:
            return
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest, size = hashlib.sha256(), 0

    def timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"download timed out for {destination.name}")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        with (
            urlopen(url, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(
                min(1024 * 1024, expected_size - size + 1)
            ):
                size += len(chunk)
                if size > expected_size:
                    raise ValueError(
                        f"oversize download for {destination.name}"
                    )
                digest.update(chunk)
                output.write(chunk)
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(f"integrity mismatch for {destination.name}")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    whisper_dir = args.model_dir / "whisper-base.en"
    whisper_dir.mkdir(exist_ok=True)
    for name, (checksum, size) in WHISPER_FILES.items():
        _download(
            f"{WHISPER_BASE_URL}/{name}",
            whisper_dir / name,
            checksum,
            size,
        )
    for suffix, (checksum, size) in PIPER_FILES.items():
        _download(
            PIPER_BASE_URL + (".json" if suffix.endswith(".json") else ""),
            args.model_dir / f"{PIPER_VOICE}{suffix}",
            checksum,
            size,
        )


if __name__ == "__main__":
    main()
