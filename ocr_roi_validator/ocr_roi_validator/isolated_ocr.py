"""Spawned OCR service with cancel/deadline enforcement; never pickle Tk widgets."""
import multiprocessing as mp
import queue
import threading
import time

from .automation import Cancelled


def _worker(connection, config, variables):
    try:
        from .gui import OCRValidatorGUI
        from .ocr_engine import OCREngine
        from .automation_ocr import FrozenValue
        processor = OCRValidatorGUI.__new__(OCRValidatorGUI)
        processor.engine = OCREngine(**config)
        for key, value in variables.items():
            setattr(processor, key, FrozenValue(value))
        while True:
            request = connection.recv()
            if request is None:
                return
            frame, rect, expected = request
            result = processor._run_roi_ocr(frame, rect, expected)
            connection.send((True, result, processor._last_ocr_input))
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send((False, str(exc), None))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class IsolatedOCR:
    def __init__(self, processor):
        self.processor = processor
        self.process = self.connection = None
        self.exchange_thread = None

    def start(self):
        engine = self.processor.engine
        config = {key: getattr(engine, key) for key in (
            "package", "paddle_package", "backend", "use_rapid_default", "use_detection_fallback")}
        variables = {key: value.get() for key, value in vars(self.processor).items() if key.endswith("_var")}
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=_worker, args=(child, config, variables), daemon=True)
        self.process.start()
        child.close()

    def run(self, frame, rect, expected, stop, deadline):
        if self.process is None:
            self.start()
        try:
            if stop.is_set():
                raise Cancelled("OCR cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("OCR observation deadline reached")
            replies = queue.Queue(maxsize=1)
            connection = self.connection

            def exchange():
                try:
                    connection.send((frame, rect, expected))
                    replies.put(connection.recv())
                except (OSError, EOFError, TypeError) as exc:
                    replies.put((False, f"OCR IPC error: {exc}", None))

            # Sending/receiving a large image must not block the watchdog thread.
            self.exchange_thread = threading.Thread(target=exchange, daemon=True)
            self.exchange_thread.start()
            while True:
                if stop.is_set():
                    raise Cancelled("OCR cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("OCR observation deadline reached")
                try:
                    success, result, record = replies.get(timeout=0.02)
                    break
                except queue.Empty:
                    if not self.process.is_alive():
                        raise RuntimeError("OCR worker exited unexpectedly")
            if stop.is_set():
                raise Cancelled("OCR cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("OCR observation deadline reached")
            if not success:
                raise RuntimeError(result)
            self.processor._last_ocr_input = record
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.exchange_thread is not None:
            self.exchange_thread.join(timeout=0.2)
            self.exchange_thread = None