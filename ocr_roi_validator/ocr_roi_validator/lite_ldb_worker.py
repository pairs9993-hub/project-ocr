"""Short-lived process importing LITE's actual LDB SDK, without its GUI/config."""
import importlib.util
from pathlib import Path
import sys


def main():
    sdk_path, port = sys.argv[1:3]
    command = sys.stdin.buffer.read().decode("utf-8")
    spec = importlib.util.spec_from_file_location("lite_ldb_sdk", Path(sdk_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Non-buffered mode returns a transport/protocol success boolean (buffered
    # mode cannot distinguish an empty successful reply from connection failure).
    result = module.process_shell_command(port, command, buffering=False)
    return 0 if result is True else 1


if __name__ == "__main__":
    raise SystemExit(main())