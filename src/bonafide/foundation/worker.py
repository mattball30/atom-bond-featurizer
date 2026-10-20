"""Persistent worker processes that run foundation model encoders in external environments."""

import atexit
import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bonafide_encoder.py")

# Number of encoder results kept in memory per worker
_MEMO_SIZE = 2048

# Number of characters of the worker's stderr included in error messages
_STDERR_TAIL = 3000


class FoundationModelWorker:
    """A persistent subprocess running a foundation model encoder.

    The model is loaded once when the worker starts. Results are memoized per molecule input, so
    that, e.g., the same SMILES string is only encoded once for all conformers of an ensemble.
    Handled errors are memoized as well; crashes and timeouts are not, and the worker is restarted
    on the next request.

    Parameters
    ----------
    python : str
        The path to the Python interpreter of the foundation model environment.
    encoder : str
        The encoder class specification ("path/to/file.py:ClassName").
    options : Dict[str, Any]
        The options passed to the ``load()`` method of the encoder.
    timeout : float
        The timeout in seconds for loading the model and for each encode request.
    """

    def __init__(self, python: str, encoder: str, options: Dict[str, Any], timeout: float) -> None:
        self.python = python
        self.encoder = encoder
        self.options = options
        self.timeout = timeout

        self._process: Optional[subprocess.Popen[str]] = None
        self._stderr_file: Optional[Any] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._memo: "OrderedDict[str, Tuple[Optional[Dict[str, Any]], Optional[str]]]" = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def encode(self, mol_input: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Encode a molecule.

        Parameters
        ----------
        mol_input : Dict[str, Any]
            The molecule data (see ``MolInput`` in ``bonafide_encoder.py``).

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[str]]
            The model outputs (``None`` if an error occurred) and an error message (``None`` if
            no error occurred).
        """
        key = hashlib.sha256(json.dumps(mol_input, sort_keys=True).encode()).hexdigest()

        with self._lock:
            if key in self._memo:
                self._memo.move_to_end(key)
                return self._memo[key]

            if self._process is None or self._process.poll() is not None:
                err = self._start()
                if err is not None:
                    return None, err

            response, err = self._request({"cmd": "encode", "mol": mol_input})
            if err is not None:
                return None, err
            assert response is not None  # for type checker

            if response["ok"] is True:
                result: Tuple[Optional[Dict[str, Any]], Optional[str]] = (
                    response["outputs"],
                    None,
                )
            else:
                result = (None, f"the encoder raised an error: {response['error']}")

            self._memo[key] = result
            if len(self._memo) > _MEMO_SIZE:
                self._memo.popitem(last=False)
            return result

    def close(self) -> None:
        """Stop the worker process."""
        with self._lock:
            self._stop()

    def _start(self) -> Optional[str]:
        """Start the worker process and load the model. Returns an error message on failure."""
        self._stop()
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            self._process = subprocess.Popen(
                [self.python, "-u", _RUNNER_PATH, "--encoder", self.encoder],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            self._process = None
            return f"could not start the foundation model worker with '{self.python}': {e}"

        # Read the protocol lines in a background thread so that requests can time out
        self._lines = queue.Queue()
        threading.Thread(
            target=self._read_lines, args=(self._process, self._lines), daemon=True
        ).start()

        response, err = self._request({"cmd": "load", "options": self.options})
        if err is not None:
            return err
        assert response is not None  # for type checker
        if response["ok"] is not True:
            error = response["error"]
            self._stop()
            return f"loading the foundation model failed: {error}"
        return None

    @staticmethod
    def _read_lines(process: "subprocess.Popen[str]", lines: "queue.Queue[Optional[str]]") -> None:
        assert process.stdout is not None  # for type checker
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    def _request(self, message: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Send a request and wait for the response. Kills the worker on crash or timeout."""
        assert self._process is not None and self._process.stdin is not None  # for type checker
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
            line = self._lines.get(timeout=self.timeout)
        except queue.Empty:
            stderr = self._stderr_tail()
            self._stop()
            return None, (
                f"the foundation model worker did not respond within {self.timeout} s "
                f"(it will be restarted on the next request); stderr: {stderr}"
            )
        except (BrokenPipeError, OSError):
            line = None

        if line is None:
            stderr = self._stderr_tail()
            self._stop()
            return None, (
                "the foundation model worker terminated unexpectedly (it will be restarted on "
                f"the next request); stderr: {stderr}"
            )
        return json.loads(line), None

    def _stderr_tail(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - _STDERR_TAIL))
            return str(self._stderr_file.read().decode(errors="replace").strip())
        except (OSError, ValueError):
            return ""

    def _stop(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    assert self._process.stdin is not None  # for type checker
                    self._process.stdin.close()
                    self._process.wait(timeout=5)
                except Exception:
                    self._process.kill()
                    self._process.wait()
            self._process = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


_WORKERS: Dict[str, FoundationModelWorker] = {}


def get_worker(
    python: str, encoder: str, options: Dict[str, Any], timeout: float
) -> FoundationModelWorker:
    """Get the worker for a foundation model, creating it if necessary.

    Workers are shared per (interpreter, encoder, options) combination, so changing the options
    with ``set_options()`` starts a new worker with the new options.

    Parameters
    ----------
    python : str
        The path to the Python interpreter of the foundation model environment.
    encoder : str
        The encoder class specification ("path/to/file.py:ClassName").
    options : Dict[str, Any]
        The options passed to the ``load()`` method of the encoder.
    timeout : float
        The timeout in seconds for loading the model and for each encode request.

    Returns
    -------
    FoundationModelWorker
        The worker.
    """
    key = json.dumps([python, encoder, options], sort_keys=True, default=str)
    worker = _WORKERS.get(key)
    if worker is None:
        worker = FoundationModelWorker(
            python=python, encoder=encoder, options=options, timeout=timeout
        )
        _WORKERS[key] = worker
    worker.timeout = timeout
    return worker


def close_all_workers() -> None:
    """Stop all foundation model worker processes."""
    for worker in _WORKERS.values():
        worker.close()
    _WORKERS.clear()


atexit.register(close_all_workers)
