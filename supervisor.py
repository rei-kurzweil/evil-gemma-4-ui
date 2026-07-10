import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request


HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
PORT = int(os.environ.get("INFERENCE_PORT", "5000"))
HEALTH_URL = os.environ.get("INFERENCE_HEALTH_URL", f"http://{HOST}:{PORT}/health")
HEALTH_POLL_SECONDS = float(os.environ.get("INFERENCE_HEALTH_POLL_SECONDS", "2"))
UNHEALTHY_GRACE_SECONDS = float(os.environ.get("INFERENCE_UNHEALTHY_GRACE_SECONDS", "10"))
MAX_RESTART_DELAY_SECONDS = float(os.environ.get("INFERENCE_MAX_RESTART_DELAY_SECONDS", "10"))
RESTART_DELAY_STEP_SECONDS = float(os.environ.get("INFERENCE_RESTART_DELAY_STEP_SECONDS", "1"))
STARTUP_GRACE_SECONDS = float(os.environ.get("INFERENCE_STARTUP_GRACE_SECONDS", "5"))
APP_COMMAND = os.environ.get(
    "INFERENCE_APP_COMMAND",
    os.path.join("venv", "bin", "python") + " app.py",
)


def healthcheck():
    request = urllib.request.Request(HEALTH_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as response:
        return 200 <= response.status < 300


def terminate_process(process):
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    process.kill()
    process.wait(timeout=5)


def main():
    print(f"[supervisor] Starting watchdog for `{APP_COMMAND}`.")
    restart_delay = 0.0
    argv = shlex.split(APP_COMMAND)

    while True:
        if restart_delay > 0:
            print(f"[supervisor] Sleeping {restart_delay:.1f}s before restart.")
            time.sleep(restart_delay)

        started_at = time.monotonic()
        process = subprocess.Popen(argv)
        print(f"[supervisor] Spawned server process pid={process.pid}.")
        unhealthy_since = None

        try:
            while True:
                exit_code = process.poll()
                if exit_code is not None:
                    runtime = time.monotonic() - started_at
                    print(
                        f"[supervisor] Server exited with code {exit_code} after {runtime:.1f}s."
                    )
                    if runtime < STARTUP_GRACE_SECONDS:
                        restart_delay = min(
                            MAX_RESTART_DELAY_SECONDS,
                            restart_delay + RESTART_DELAY_STEP_SECONDS,
                        )
                    else:
                        restart_delay = 0.0
                    break

                try:
                    if healthcheck():
                        unhealthy_since = None
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    now = time.monotonic()
                    if unhealthy_since is None:
                        unhealthy_since = now
                        print(f"[supervisor] Healthcheck failed: {exc}.")
                    elif now - unhealthy_since >= UNHEALTHY_GRACE_SECONDS:
                        print(
                            "[supervisor] Server stayed unhealthy for "
                            f"{UNHEALTHY_GRACE_SECONDS:.1f}s; restarting."
                        )
                        terminate_process(process)
                        restart_delay = 0.0
                        break

                time.sleep(HEALTH_POLL_SECONDS)
        except KeyboardInterrupt:
            print("[supervisor] Stopping watchdog.")
            terminate_process(process)
            return 0


if __name__ == "__main__":
    sys.exit(main())
