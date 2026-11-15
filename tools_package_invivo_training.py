# -*- coding: utf-8 -*-
"""打包服务器五参数训练的代码和小标签，不包含大缓存或任何训练执行。"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    files = sorted((ROOT / "bmode_dl").glob("*.py")) + [ROOT / name for name in (
        "tools_train_six_param.py", "tools_evaluate_six_param.py",
        "tests/verify_five_param_training.py", "docs/five_parameter_training_20261115.md",
        "data/labels_console_invivo_carotid_target.jsonl")]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path.name)
    destination = ROOT / "data/server_packages/invivo_five_v1.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("invivo_five_package_manifest.json", json.dumps(manifest, indent=2) + "\n")
    report = ("Server training package: data/server_packages/invivo_five_v1.zip\n"
              "Files: %d plus a SHA256 manifest\nBytes: %d\n"
              "The large cache is excluded; transfer cache.npz and index.json separately.\n"
              "No training was executed.\n" % (len(files), destination.stat().st_size))
    with io.open(ROOT / "tools_package_invivo_training.txt", "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
