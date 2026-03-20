from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time

from reporting.log import log_event


def _stream_prefixed_output(format_name: str, stream) -> None:
    try:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if line:
                print(f"[{format_name}] {line}", flush=True)
            else:
                print(flush=True)
    finally:
        stream.close()


def _build_child_command(
    *,
    entrypoint_path: pathlib.Path,
    language: str,
    format_name: str,
    requested_device: str,
    checkpoint_mode: str,
) -> list[str]:
    return [
        sys.executable,
        str(entrypoint_path),
        "--language",
        language,
        "--format",
        format_name,
        "--device",
        requested_device,
        "--checkpoint-mode",
        checkpoint_mode,
    ]


def run_formats_in_parallel(
    *,
    entrypoint_path: pathlib.Path,
    language: str,
    formats: list[str],
    requested_device: str,
    checkpoint_mode: str,
) -> None:
    started_at = time.perf_counter()
    log_event(
        "formats.parallel.start",
        language=language,
        formats=", ".join(formats),
        requested_device=requested_device,
        checkpoint_mode=checkpoint_mode,
        process_count=len(formats),
    )

    processes: dict[str, subprocess.Popen] = {}
    threads: list[threading.Thread] = []

    try:
        for format_name in formats:
            command = _build_child_command(
                entrypoint_path=entrypoint_path,
                language=language,
                format_name=format_name,
                requested_device=requested_device,
                checkpoint_mode=checkpoint_mode,
            )
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=entrypoint_path.parent.parent.parent,
            )
            processes[format_name] = process

            if process.stdout is None:
                raise SystemExit(
                    f"Failed to capture output for format process: {format_name}"
                )
            thread = threading.Thread(
                target=_stream_prefixed_output,
                args=(format_name, process.stdout),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        failures: list[tuple[str, int]] = []
        active_formats = set(processes)
        while active_formats:
            for format_name in list(active_formats):
                return_code = processes[format_name].poll()
                if return_code is None:
                    continue

                active_formats.remove(format_name)
                if return_code != 0:
                    failures.append((format_name, return_code))
                    for other_format in active_formats:
                        processes[other_format].terminate()
                    active_formats.clear()
                    break
            if active_formats:
                time.sleep(0.2)

        for process in processes.values():
            process.wait()
        for thread in threads:
            thread.join()

        if failures:
            failure_text = ", ".join(
                f"{format_name}={code}" for format_name, code in failures
            )
            raise SystemExit(
                "Parallel format training failed: "
                f"{failure_text}"
            )

        log_event(
            "formats.parallel.complete",
            language=language,
            formats=", ".join(formats),
            duration=f"{time.perf_counter() - started_at:.2f}s",
        )
    finally:
        for process in processes.values():
            if process.stdout is not None:
                process.stdout.close()
