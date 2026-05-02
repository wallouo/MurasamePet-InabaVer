"""Thread-safe memory persistence for MurasamePet-Inaba."""

import json
import threading

_DEFAULT_MEMORY = {
    "name": "",
    "last_topic": "",
    "mood": ""
}
_MEMORY_FILE_PATH = "./memory.json"


class MemoryManager:
    """Provide thread-safe access to the persistent memory JSON file."""

    def __init__(self):
        self._data_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._file_path = _MEMORY_FILE_PATH
        self._last_error = None
        self._memory_data = dict(_DEFAULT_MEMORY)
        self.load()

    def load(self):
        """Load memory data from disk into the in-memory cache."""
        raw_data = None
        with self._file_lock:
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                self._last_error = None
            except FileNotFoundError:
                raw_data = dict(_DEFAULT_MEMORY)
            except (json.JSONDecodeError, OSError) as e:
                self._last_error = f"Failed to load memory: {e}"
                raw_data = dict(_DEFAULT_MEMORY)

        # _sanitize 在鎖外執行，避免潛在 deadlock
        sanitized = self._sanitize(raw_data)
        with self._data_lock:
            self._memory_data = sanitized
        return dict(self._memory_data)

    def get(self):
        """Retrieve a copy of the cached memory data."""
        with self._data_lock:
            return dict(self._memory_data)

    def set(self, *, name=None, last_topic=None, mood=None):
        """Update allowed memory fields and trigger an async save."""
        updates = {}
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("name must be a string.")
            updates["name"] = name
        if last_topic is not None:
            if not isinstance(last_topic, str):
                raise TypeError("last_topic must be a string.")
            updates["last_topic"] = last_topic
        if mood is not None:
            if not isinstance(mood, str):
                raise TypeError("mood must be a string.")
            updates["mood"] = mood

        if not updates:
            return

        with self._data_lock:
            for key, value in updates.items():
                self._memory_data[key] = value
        self.save()

    def save(self):
        """Persist the current memory snapshot to disk asynchronously."""
        with self._data_lock:
            snapshot = dict(self._memory_data)
        threading.Thread(
            target=self._write_to_file,
            args=(snapshot,),
            daemon=True,
        ).start()

    @property
    def last_error(self):
        """Return the last file I/O error message, or None if clean."""
        return self._last_error

    def _write_to_file(self, data):
        """Write memory data to disk (runs in background thread)."""
        try:
            sanitized = self._sanitize(data)
            with self._file_lock:
                with open(self._file_path, "w", encoding="utf-8") as f:
                    json.dump(sanitized, f, ensure_ascii=False, indent=2)
            self._last_error = None
        except OSError as e:
            self._last_error = f"Failed to save memory: {e}"

    def _sanitize(self, data):
        """Return a sanitized copy containing only supported string fields."""
        base = {"name": "", "last_topic": "", "mood": ""}
        if isinstance(data, dict):
            for key in base:
                value = data.get(key, "")
                if isinstance(value, str):
                    base[key] = value
        return base