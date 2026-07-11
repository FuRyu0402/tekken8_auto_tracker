from __future__ import annotations

import os
import time
from pathlib import Path
from types import TracebackType
from typing import BinaryIO


DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0
LOCK_RETRY_SECONDS = 0.05


class CsvLockTimeoutError(TimeoutError):
    pass


def lock_path_for(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.name}.lock")


class CsvFileLock:
    def __init__(self, csv_path: Path, timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS):
        self.csv_path = Path(csv_path)
        self.lock_path = lock_path_for(self.csv_path)
        self.timeout = timeout
        self._file: BinaryIO | None = None

    def __enter__(self) -> CsvFileLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.lock_path.open("a+b")
        if self._file.tell() == 0:
            self._file.write(b"\0")
            self._file.flush()

        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._acquire_once()
                return self
            except OSError as error:
                if time.monotonic() >= deadline:
                    self._file.close()
                    self._file = None
                    raise CsvLockTimeoutError(
                        f"CSVロックを{self.timeout:.1f}秒以内に取得できませんでした: "
                        f"{self.lock_path}"
                    ) from error
                time.sleep(LOCK_RETRY_SECONDS)

    def _acquire_once(self) -> None:
        assert self._file is not None
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None
