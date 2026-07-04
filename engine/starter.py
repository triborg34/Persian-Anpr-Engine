import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)

EXE_PATH = os.environ.get("ANPR_EXE_PATH", "api.exe")
POCKETBASE_PORT = 8090


def get_pids_on_port(port: int) -> list[str]:
    try:
        output = subprocess.check_output(
            f'netstat -ano | findstr :{port}', shell=True,
            stderr=subprocess.DEVNULL).decode()
        pids = set()
        for line in output.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5:
                pids.add(parts[-1])
        return list(pids)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def kill_pid(pid: str) -> None:
    subprocess.run(f'taskkill /PID {pid} /F', shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_api_port() -> int | None:
    import requests
    try:
        resp = requests.get(
            f'http://127.0.0.1:{POCKETBASE_PORT}/api/collections/setting/records',
            timeout=3)
        return resp.json()['items'][0]['port']
    except Exception as e:
        logging.warning(f"Could not read port from PocketBase: {e}")
        return None


def kill_port_processes(port: int) -> None:
    for pid in get_pids_on_port(port):
        logging.info(f"Killing PID {pid} on port {port}")
        kill_pid(pid)


class AppController:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.process = None
        self._setup_ui()
        self._auto_start()

    def _setup_ui(self):
        self.root.title("ANPR Controller")
        self.root.geometry("320x240")
        self.root.resizable(False, False)

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var,
                                       foreground='gray')
        self.status_label.pack(pady=(15, 5))

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Start", width=20,
                   command=self.start).pack(pady=4)
        ttk.Button(btn_frame, text="Stop", width=20,
                   command=self.stop).pack(pady=4)
        ttk.Button(btn_frame, text="Reset", width=20,
                   command=self.reset).pack(pady=4)

    def _set_status(self, text: str, color: str = 'gray'):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    def _auto_start(self):
        self._set_status("Starting...", 'orange')
        threading.Thread(target=self._do_start, daemon=True).start()

    def _do_start(self):
        if self.process and self.process.poll() is None:
            self.root.after(0, self._set_status, "Already running", 'green')
            return
        try:
            self.process = subprocess.Popen(
                EXE_PATH, creationflags=subprocess.CREATE_NO_WINDOW)
            self.root.after(0, self._set_status, "Running", 'green')
            logging.info(f"Started {EXE_PATH} (PID {self.process.pid})")
        except Exception as e:
            self.root.after(0, self._set_status, f"Failed: {e}", 'red')
            logging.error(f"Failed to start: {e}")

    def start(self):
        self._set_status("Starting...", 'orange')
        threading.Thread(target=self._do_start, daemon=True).start()

    def _do_stop(self):
        try:
            api_port = get_api_port()
            if api_port:
                kill_port_processes(api_port)
            kill_port_processes(POCKETBASE_PORT)
        except Exception as e:
            logging.error(f"Error killing ports: {e}")

        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logging.info("App stopped")

        self.root.after(0, self._set_status, "Stopped", 'red')

    def stop(self):
        self._set_status("Stopping...", 'orange')
        threading.Thread(target=self._do_stop, daemon=True).start()

    def reset(self):
        self._set_status("Resetting...", 'orange')
        def _do():
            self._do_stop()
            self._do_start()
        threading.Thread(target=_do, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    AppController(root)
    root.mainloop()
