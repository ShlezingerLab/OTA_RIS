import os
import subprocess
import sys


def _run(script_relpath: str, script_args: list[str]) -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(root, script_relpath)
    cmd = [sys.executable, script_path, *script_args]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: playground.py {train,test} [args...]\n\n"
            "Single entrypoint to run training or testing with your chosen args.\n\n"
            "Examples:\n"
            "  python playground.py train --epochs 5 --batchsize 64\n"
            "  python playground.py test --checkpoint MY_code/models_dict/minn_model.pth\n"
            "  python playground.py train -h   (shows training.py help)\n"
            "  python playground.py test -h    (shows test.py help)\n"
        )
        return 0

    command = argv[0]
    forwarded = argv[1:]

    if command == "train":
        return _run(os.path.join("MY_code", "training.py"), forwarded)
    if command == "test":
        return _run(os.path.join("MY_code", "test.py"), forwarded)

    print(f"[ERROR] Unknown command: {command}\n")
    print("Run: python playground.py -h")
    return 2


if __name__ == "__main__":
    # ===== IDE-friendly config =====
    # If you run this file from your IDE (no CLI args), edit these two lines:
    IDE_COMMAND = "train"  # "train" or "test"
    IDE_ARGS = ["-h"]  # default: show help so you don't accidentally start a long run
    # Examples:
    # IDE_ARGS = ["--epochs", "5", "--batchsize", "64", "--save_path", "MY_code/models_dict/my_run.pth"]

    # If you *do* provide CLI args in your IDE run configuration, they win.
    argv = sys.argv[1:]
    if not argv:
        argv = [IDE_COMMAND, *IDE_ARGS]

    raise SystemExit(main(argv))
