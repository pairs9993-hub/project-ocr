"""Excel/Vesta automation backend. No Tk calls and no modifications to LITE config."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time
import zipfile

from PIL import Image

from .roi_presets import read_excel
from .window_anchor import choose_window, client_windows, foreground, mapped_rect, require_unobstructed


class Cancelled(Exception):
    pass


def check_cancel(stop):
    if stop.is_set():
        raise Cancelled("Automation stopped.")


def number(value, default, label, positive=False):
    result = float(default if value in (None, "") else value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite.")
    return result


def normalize_header(value):
    return re.sub(r"[\s_.-]", "", str(value or "")).lower()


@dataclass
class TestCase:
    sheet: str
    row: int
    title: str
    preset: str
    commands: list[str]
    delay: float = 0
    image: str = ""
    expected: dict[int, str] = field(default_factory=dict)
    expected_from_blocks: bool = False
    tc_number: str = ""
    max_observation: float = 30.0
    verification_mode: str = "content"
    required_cycles: int = 1
    roi_modes: dict[int, str] = field(default_factory=dict)


def verification_options(values):
    maximum = number(values.get("maxobservationsec"), 30, "Max_observation_sec", positive=True)
    if maximum > 30:
        raise ValueError("Max_observation_sec must not exceed 30 seconds.")
    mode = str(values.get("verificationmode") or "content").strip().lower()
    if mode not in ("content", "cycles"):
        raise ValueError("Verification_mode must be content or cycles.")
    cycles = number(values.get("requiredcycles"), 1, "Required_cycles", positive=True)
    if not cycles.is_integer():
        raise ValueError("Required_cycles must be a positive integer.")
    modes = {}
    for key, value in values.items():
        match = re.fullmatch(r"verificationmode(\d+)", key)
        if match and value is not None and str(value).strip():
            value = str(value).strip().lower()
            if value not in ("content", "cycles"):
                raise ValueError(f"{key}: mode must be content or cycles.")
            modes[int(match[1])] = value
    return maximum, mode, int(cycles), modes


def parse_expected_text(value: str) -> dict[int, str]:
    """Parse standalone [ROI<ID>] markers; preserve text whitespace inside blocks.

    Blank lines surrounding a block are separators, not expected text. Marker
    lines are reserved; malformed markers must not become another ROI's text.
    """
    if not isinstance(value, str):
        raise ValueError("Expected_Text must be text using [ROI1] blocks.")
    expected = {}
    current_id = None
    lines = []

    def finish():
        content = list(lines)
        while content and not content[0].strip():
            content.pop(0)
        while content and not content[-1].strip():
            content.pop()
        if not content:
            raise ValueError(f"Expected_Text: [ROI{current_id}] has empty expected text.")
        expected[current_id] = "\n".join(content)

    for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        marker = re.fullmatch(r"\[ROI(\d+)\]", line.strip(), re.IGNORECASE)
        if marker:
            if current_id is not None:
                finish()
            current_id = int(marker[1])
            if current_id <= 0:
                raise ValueError("Expected_Text: ROI IDs must be positive integers.")
            if current_id in expected:
                raise ValueError(f"Expected_Text: duplicate [ROI{current_id}].")
            lines = []
        else:
            if re.match(r"\[\s*ROI", line.strip(), re.IGNORECASE):
                raise ValueError(f"Expected_Text: invalid marker {line.strip()!r}; use [ROI1] on its own line.")
            if current_id is None:
                if line.strip():
                    raise ValueError("Expected_Text must begin with [ROI1] (or another saved ROI ID).")
            else:
                lines.append(line)
    if current_id is None:
        raise ValueError("Expected_Text contains no [ROI<ID>] blocks.")
    finish()
    return expected


def command_action(command: str, tables: dict) -> tuple[str, str | float]:
    """LITE raw-shell / :TBL / :API System.sleep semantics, fail closed otherwise."""
    command = command.strip()
    if not command:
        raise ValueError("Empty Vesta command.")
    if not command.startswith(":"):
        # LITE needs reboot lifecycle handling; do not silently continue after it.
        if command.split()[0].lower() == "reboot":
            raise ValueError("reboot requires LITE device lifecycle support; split the TC run instead.")
        return "shell", command
    prefix, _, arg = command.partition(" ")
    arg = arg.strip()
    if prefix == ":TBL":
        params = json.loads(arg) if arg.startswith("{") else tables.get(arg)
        if not isinstance(params, dict):
            raise ValueError(f"Unknown :TBL entry: {arg}")
        packet, label = str(params.get("Packet", "")), params.get("Label")
        if not packet.startswith("@"):
            raise ValueError(f":TBL packet must start with @: {arg}")
        if label and str(label).lower().startswith("uart"):
            return "shell", f"sim_uart write -l {label} -d {packet.lstrip('@')}"
        if label is None:
            return "shell", f"sim_irrc write -d {packet.lstrip('@')}"
        raise ValueError(f"Unsupported :TBL label: {label}")
    if prefix == ":API":
        match = re.fullmatch(r'System\.sleep\s*\{\s*["\']?time["\']?\s*:\s*([\d.eE+-]+)\s*\}', arg)
        if match:
            return "sleep", number(match[1], 0, "Sleep")
    raise ValueError(f"Unsupported LITE command: {command}. Supported: shell, :TBL, :API System.sleep.")


def load_cases(path: Path, roi_column="ROI") -> tuple[list[TestCase], dict]:
    sheets = read_excel(path)
    tables = {}
    for name, rows in sheets.items():
        if name.lower().startswith("tbl_"):
            for row in rows:
                if len(row) >= 4 and row[0] is not None and str(row[3] or "").startswith("@"):
                    key = str(row[0]).strip()
                    entry = dict(zip(("Data", "Type", "Label", "Packet"), row[:4]))
                    if entry["Label"] is None:
                        entry.pop("Label")
                    if key in tables and tables[key] != entry:
                        raise ValueError(f"Ambiguous :TBL key: {key}")
                    tables[key] = entry
    candidates = {n: rows for n, rows in sheets.items() if n.lower().startswith("tc_")}
    if not candidates:
        candidates = {n: rows for n, rows in sheets.items() if not n.lower().startswith("tbl_")}
    cases = []
    for name, rows in candidates.items():
        header_index = next((i for i, row in enumerate(rows)
                             if "commands" in [normalize_header(v) for v in row]
                             and normalize_header(roi_column) in [normalize_header(v) for v in row]), None)
        if header_index is None:
            continue
        headers = [normalize_header(v) for v in rows[header_index]]
        nonempty = [h for h in headers if h]
        if len(set(nonempty)) != len(nonempty):
            raise ValueError(f"{name}: duplicate Excel headers.")
        for row_number, row in enumerate(rows[header_index+1:], header_index+2):
            if not any(v is not None and str(v).strip() for v in row):
                continue
            values = dict(zip(headers, row))
            if str(values.get("execution", "")).strip().upper() in ("N", "NO", "FALSE", "0"):
                continue
            def text(key):
                value = values.get(key)
                return "" if value is None else str(value).strip()
            preset = text(normalize_header(roi_column))
            if not preset:
                raise ValueError(f"{name}!{row_number}: missing ROI preset name.")
            commands = [line.strip() for line in text("commands").splitlines() if line.strip()]
            for command in commands:
                try:
                    command_action(command, tables)
                except ValueError as exc:
                    raise ValueError(f"{name}!{row_number}: {exc}") from exc
            expected = {}
            for key, value in values.items():
                match = re.fullmatch(r"expected(?:roi)?(\d+)", key)
                if match and value is not None and str(value).strip():
                    roi_id = int(match[1])
                    if roi_id <= 0 or roi_id in expected:
                        raise ValueError(f"{name}!{row_number}: invalid/duplicate Expected ROI ID {roi_id}.")
                    expected[roi_id] = str(value)
            block_value = values.get("expectedtext")
            expected_from_blocks = block_value is not None and bool(str(block_value).strip())
            if expected_from_blocks:
                if expected:
                    raise ValueError(f"{name}!{row_number}: use either Expected_Text or Expected_<ID> columns, not both in one row.")
                try:
                    expected = parse_expected_text(block_value)
                except ValueError as exc:
                    raise ValueError(f"{name}!{row_number}: {exc}") from exc
            try:
                maximum, mode, cycles, modes = verification_options(values)
            except ValueError as exc:
                raise ValueError(f"{name}!{row_number}: {exc}") from exc
            cases.append(TestCase(name, row_number, text("title") or text("no") or f"Row {row_number}",
                         preset, commands, number(values.get("capturewaitingtime"), 0, "Capture_waiting_time"),
                         text("image"), expected, expected_from_blocks, text("no"), maximum, mode, cycles, modes))
    if not cases:
        raise ValueError(f"No enabled test rows found. Add Commands and {roi_column} headers to a TC sheet.")
    return cases, tables


def safe_extract(archive: Path, destination: Path, stop) -> Path:
    """Reject traversal, symlinks, drive/ADS names and oversized ZIPs before writing."""
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > 100000 or sum(m.file_size for m in members) > 8 * 1024**3:
            raise ValueError("Vesta ZIP exceeds the 8 GiB / 100000 entry limit.")
        seen = set()
        for member in members:
            name = member.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith("/") or any(p in ("..", ".") or ":" in p
                    or p.endswith((" ", ".")) for p in parts)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe ZIP member: {name}")
            if name.casefold().rstrip("/") in seen:
                raise ValueError(f"Duplicate ZIP member: {name}")
            seen.add(name.casefold().rstrip("/"))
        for member in members:
            check_cancel(stop)
            target = destination.joinpath(*PurePosixPath(member.filename.replace("\\", "/")).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    while chunk := source.read(1024*1024):
                        check_cancel(stop)
                        output.write(chunk)
    executables = [p for p in destination.rglob("*") if p.name.lower() == "vesta.exe"]
    if len(executables) != 1:
        raise ValueError("ZIP must contain exactly one vesta.exe.")
    return executables[0]


def product_config_names(names) -> list[str]:
    """Vesta ships JSON content in .cfg files as well as .json files."""
    result = set()
    for name in names:
        path = PurePosixPath(str(name).replace("\\", "/"))
        if (path.parent.name == "sim_config" and path.suffix.lower() in (".cfg", ".json")
                and path.stem.lower() not in ("target", "global")):
            result.add(path.stem)
    return sorted(result)


def product_config(executable: Path, product: str) -> tuple[str, str]:
    config_dir = executable.parent / "sim_config"
    available = product_config_names(config_dir.glob("*"))
    configs = [p for p in config_dir.glob("*") if p.stem in available and p.suffix.lower() in (".json", ".cfg")]
    if product:
        configs = [p for p in configs if p.stem == product]
    if len(configs) != 1:
        raise ValueError("Enter a Product matching a sim_config .cfg/.json filename. Available: " + ", ".join(available))
    config = json.loads(configs[0].read_text(encoding="utf-8-sig"))
    port = config["Port List"]["Socket"]["0"]["Client"]
    host, sep, raw_port = str(port).rpartition(":")
    if not sep or host not in ("127.0.0.1", "localhost") or not 1 <= int(raw_port) <= 65535:
        raise ValueError("Vesta debug Client must be a localhost TCP endpoint (127.0.0.1:port).")
    return configs[0].stem, str(port)


class VestaSession:
    def __init__(self, lite_root: Path, output: Path):
        self.sdk = lite_root / "src" / "sdk" / "ldb.py"
        self.output = output
        self.process = None
        self.log = None
        self.port = ""

    @property
    def pid(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("The launched Vesta process is not running.")
        return self.process.pid

    def launch(self, archive: Path, product: str, stop):
        import psutil  # Check cleanup dependency before launching any executable.
        if not self.sdk.is_file():
            raise ValueError(f"LITE SDK not found: {self.sdk}")
        executable = safe_extract(archive, self.output / "vesta", stop)
        product, self.port = product_config(executable, product)
        host, port = self.port.rsplit(":", 1)
        # Never send commands to a pre-existing simulator sharing this endpoint.
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex((host, int(port))) == 0:
                raise ValueError(f"{self.port} is already in use. Close the old Vesta before starting.")
        self.log = (self.output / "vesta.log").open("wb")
        self.process = subprocess.Popen([str(executable), "-t", product], cwd=executable.parent,
                                        stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            check_cancel(stop)
            self.pid
            with socket.socket() as probe:
                probe.settimeout(0.3)
                if probe.connect_ex((host, int(port))) == 0 and client_windows(self.pid):
                    return
            stop.wait(0.2)
        raise TimeoutError("Vesta did not open its window/debug port in 60 seconds. See vesta.log.")

    def shell(self, command, stop, timeout=30):
        self.pid
        worker = Path(__file__).with_name("lite_ldb_worker.py")
        with (self.output / "commands.log").open("ab") as log:
            log.write((f"\n>>> {command}\n").encode("utf-8"))
            log.flush()
            process = subprocess.Popen([sys.executable, str(worker), str(self.sdk), self.port],
                                       stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                process.stdin.write(command.encode("utf-8"))
                process.stdin.close()
                deadline = time.monotonic()+timeout
                while process.poll() is None:
                    check_cancel(stop)
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"LDB command timed out: {command}")
                    stop.wait(0.05)
                if process.returncode:
                    raise RuntimeError(f"LDB command failed: {command}; see commands.log")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()

    def execute(self, case, tables, stop):
        previous_was_shell = False
        for command in case.commands:
            check_cancel(stop)
            action, value = command_action(command, tables)
            if action == "sleep":
                if stop.wait(value):
                    raise Cancelled()
                previous_was_shell = False
            else:
                # Explicit sleep commands replace the default inter-command wait.
                if previous_was_shell and stop.wait(1.0):
                    raise Cancelled()
                check_cancel(stop)
                self.shell(value, stop)
                previous_was_shell = True
        if stop.wait(case.delay):
            raise Cancelled()

    def focus(self, preset):
        foreground(choose_window(client_windows(self.pid), preset["capture_anchor"]))

    def frame(self, preset, capture_session):
        from .capture import grab_screen_rect_with
        window = choose_window(client_windows(self.pid), preset["capture_anchor"])
        rect = mapped_rect(preset["capture_anchor"], window.rect, preset["image_size"])
        require_unobstructed(self.pid, rect)
        frame = grab_screen_rect_with(capture_session, rect)
        size = tuple(preset["image_size"])
        return frame if frame.size == size else frame.resize(size, Image.Resampling.LANCZOS)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                # Only descendants of the process launched by this session.
                import psutil
                try:
                    children = psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in reversed(children):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                _, alive = psutil.wait_procs(children, timeout=2)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            self.process = None
        if self.log:
            self.log.close()
            self.log = None


def reference_path(case: TestCase, directory: Path) -> Path | None:
    if not case.image:
        return None
    root = directory.resolve()
    relative = Path(case.image)
    direct = (root / relative).resolve()
    if not direct.is_relative_to(root):
        raise ValueError("Image must be relative to the reference folder.")
    if direct.is_file():
        return direct
    matches = [p for p in root.rglob("*") if p.is_file() and p.name.casefold() == relative.name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Reference image missing/ambiguous: {case.image}")
    return matches[0]