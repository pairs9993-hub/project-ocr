"""Latest-frame capture: never build an OCR backlog."""
from collections import deque
import threading
import time
import mss
from .automation import check_cancel


class BufferedCapture:
    def __init__(self, session, preset, fps, duration, stop, max_bytes=128*1024*1024):
        self.session, self.preset = session, preset
        self.interval, self.duration, self.stop = 1/fps, duration, stop
        self.max_bytes = max_bytes
        self.queue = deque()
        self.condition = threading.Condition()
        self.halt = threading.Event()
        self.done = False
        self.error = None
        self.bytes = 0
        self.captured = self.dropped = self.consumed = 0
        self.superseded = 0

    def __enter__(self):
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()
        return self

    def _produce(self):
        try:
            end = time.monotonic()+self.duration
            with mss.mss() as capture:
                while not self.halt.is_set() and not self.stop.is_set() and time.monotonic() < end:
                    started = time.monotonic()
                    image = self.session.frame(self.preset, capture)
                    captured_at = time.monotonic()
                    size = image.width*image.height*len(image.getbands())
                    with self.condition:
                        self.captured += 1
                        while self.queue:
                            _, _, removed_size = self.queue.popleft()
                            self.bytes -= removed_size
                            self.superseded += 1
                        if size <= self.max_bytes:
                            self.queue.append((image, captured_at, size))
                            self.bytes += size
                        else:
                            self.dropped += 1
                        self.condition.notify_all()
                    self.halt.wait(max(0, min(end-time.monotonic(), self.interval-(time.monotonic()-started))))
        except Exception as exc:
            self.error = exc
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def next_frame(self, deadline):
        with self.condition:
            while True:
                check_cancel(self.stop)
                if self.error is not None:
                    raise self.error
                if time.monotonic() >= deadline:
                    return None
                if self.queue:
                    image, captured_at, size = self.queue.popleft()
                    self.bytes -= size
                    self.consumed += 1
                    return image, captured_at
                if self.done:
                    return None
                self.condition.wait(.05)

    def stats(self):
        with self.condition:
            return dict(captured_frames=self.captured, consumed_frames=self.consumed,
                        dropped_frames=self.dropped, superseded_frames=self.superseded,
                        pending_frames=len(self.queue), capture_policy="latest_frame",
                        buffer_limit_bytes=self.max_bytes)

    def __exit__(self, *args):
        self.halt.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Screen capture did not stop within 2 seconds")
