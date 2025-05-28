import os
import shutil


def remove_pycache(start_dir: str = ".") -> None:
    for root, dirs, _ in os.walk(start_dir):
        if "__pycache__" in dirs:
            target = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(target)
                print(f"Removed {target}")
            except OSError as exc:
                print(f"Failed to remove {target}: {exc}")


if __name__ == "__main__":
    remove_pycache()
