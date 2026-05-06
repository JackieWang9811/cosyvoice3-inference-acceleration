from pathlib import Path
import soundfile as sf

dirs = [
    # "/home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results/wavs/",
    # "/home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results_awq/wavs/",
    # "/home/b24zll/data/wjq_data/CosyVoice/concurrency_testing/test_results_awq_v2/wavs/",
    # "/home/b24zll/data/wjq_data/CosyVoice/speculative_decoding/benchmark_results_1.0/eagle/",
    "/home/b24zll/data/wjq_data/CosyVoice/speculative_decoding/benchmark_results_1.0/baseline/",
]

audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def format_sec(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m}m{s:.2f}s" if m > 0 else f"{s:.4f}s"


for d in dirs:
    p = Path(d)
    print("=" * 100)
    print(f"[DIR] {d}")

    if not p.exists():
        print("  -> directory not found\n")
        continue

    files = sorted(
        [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in audio_exts]
    )

    if not files:
        print("  -> no audio files found\n")
        continue

    durations = []
    per_file = []

    for f in files:
        try:
            info = sf.info(str(f))
            dur = info.frames / info.samplerate
            durations.append(dur)
            per_file.append((str(f), dur, info.samplerate, info.frames))
        except Exception as e:
            print(f"  -> failed to read: {f} | {e}")

    if not durations:
        print("  -> no readable audio files\n")
        continue

    total_sec = sum(durations)
    avg_sec = total_sec / len(durations)
    min_sec = min(durations)
    max_sec = max(durations)

    print(f"num_files : {len(durations)}")
    print(f"avg_sec   : {avg_sec:.4f}  ({format_sec(avg_sec)})")
    print(f"total_sec : {total_sec:.4f}  ({format_sec(total_sec)})")
    print(f"min_sec   : {min_sec:.4f}  ({format_sec(min_sec)})")
    print(f"max_sec   : {max_sec:.4f}  ({format_sec(max_sec)})")
    print()

    print("[Per-file durations]")
    for path, dur, sr, frames in per_file:
        print(f"{dur:10.4f}s | sr={sr:6d} | frames={frames:8d} | {path}")

    print()