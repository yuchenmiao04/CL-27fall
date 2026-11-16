#!/usr/bin/env python3
"""Run local correctness checks and write portable, data-free receipts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from cl27.protocol import validate_protocol

    output = ROOT / "validation"
    output.mkdir(exist_ok=True)
    xml_path = output / "unit-tests.xml"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests", "--junitxml=" + str(xml_path)],
        cwd=ROOT,
        check=False,
    )
    tree = ET.parse(xml_path)
    for node in tree.iter():
        if "hostname" in node.attrib:
            node.set("hostname", "local-validation")
        for key, value in list(node.attrib.items()):
            node.set(key, value.replace(str(ROOT), "<repository>"))
    tree.write(xml_path, encoding="unicode", xml_declaration=True)
    suites = list(tree.getroot().iter("testsuite"))
    totals = {
        key: sum(int(s.get(key, "0")) for s in suites)
        for key in ("tests", "errors", "failures", "skipped")
    }
    totals["seconds"] = sum(float(s.get("time", "0")) for s in suites)
    pin = json.loads((ROOT / "UPSTREAM_PIN.json").read_text())
    changed = [
        entry["path"]
        for entry in pin["files"]
        if not (ROOT / entry["path"]).is_file()
        or digest(ROOT / entry["path"]) != entry["sha256"]
    ]
    expected = ["README.md", "trainer.py", "utils/factory.py"]
    if sorted(changed) != expected:
        raise RuntimeError("Unexpected upstream source changes: " + repr(changed))
    if digest(ROOT / "UPSTREAM_README.md") != pin["files"][0]["sha256"]:
        raise RuntimeError("Original upstream README identity differs")
    configs = []
    for path in sorted((ROOT / "exps/pilot_figure2").glob("*.json")):
        config = json.loads(path.read_text())
        config["seed"] = config["seed"][0]
        validate_protocol(config)
        configs.append(str(path.relative_to(ROOT)))
    implementation = {}
    for pattern in (
        "cl27/**/*.py",
        "models/shared_basis.py",
        "models/history_transport.py",
        "models/current_adapted.py",
        "scripts/*.py",
        "tests/*.py",
        "exps/pilot_figure2/*.json",
        "exps/smoke/*.json",
        "requirements*.txt",
    ):
        for path in sorted(ROOT.glob(pattern)):
            implementation[str(path.relative_to(ROOT))] = digest(path)
    summary = {
        "status": "passed" if result.returncode == 0 else "failed",
        "scope": "numerical, native lifecycle and synthetic CLI correctness; no full benchmark",
        "tests": totals,
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "architecture": platform.machine(),
            **{
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "torchvision",
                    "timm",
                    "numpy",
                    "scipy",
                    "Pillow",
                    "pytest",
                )
            },
        },
        "upstream_commit": pin["commit"],
        "upstream_expected_changed_files": changed,
        "upstream_preserved_file_count": len(pin["files"]) - len(changed),
        "formal_config_checks": configs,
        "implementation_sha256": implementation,
        "full_experiment_started": False,
        "benchmark_performance_claim": False,
    }
    (output / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if result.returncode:
        return result.returncode
    print(
        json.dumps(
            {
                "status": "passed",
                "tests": totals["tests"],
                "protocol_configs": len(configs),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
