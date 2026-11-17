#!/usr/bin/env python3
"""Compare two trusted local CL27 checkpoints without starting CUDA.

All loaded state is compared, including unknown metadata. The sole exclusion is
the key ``wall_seconds`` in each mapping directly inside root ``stages``.
Tensor/array values and floating-point scalars are compared as raw bits, not
with numeric equality or tolerances. Object arrays and dataclasses are walked
recursively. File SHA256 and timing-normalized semantic SHA256 are distinct.

Only use this on trusted local files: torch.load(weights_only=False) can execute
pickle code. CPU map_location avoids GPU allocation; storage aliasing and the
serialization's device/storage encoding are not learner-state comparison keys.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch


REQUIRED_FIELDS = {
    "version", "random_stream_policy", "config_fingerprint", "data_fingerprint",
    "learner", "rng", "stages",
}
IGNORED_POLICY = "Only root stages[index].wall_seconds is excluded."


def type_name(value):
    cls = type(value)
    return cls.__module__ + "." + cls.__qualname__


def _frame(tag, payload):
    return tag + struct.pack("!Q", len(payload)) + payload


def scalar_payload(value):
    """Canonical bytes for supported Python scalars, preserving signed zero."""
    if value is None:
        return b""
    if type(value) is bool:
        return bytes([value])
    if type(value) is int:
        return str(value).encode("ascii")
    if type(value) is float:
        return struct.pack("!d", value)
    if type(value) is complex:
        return struct.pack("!dd", value.real, value.imag)
    if type(value) is str:
        return value.encode("utf-8", errors="surrogatepass")
    if type(value) in {bytes, bytearray}:
        return bytes(value)
    raise TypeError("Unsupported scalar type: " + type_name(value))


def key_token(value):
    """Collision-free ordering/matching tokens for ordinary checkpoint keys."""
    tag = type_name(value).encode()
    if type(value) is tuple:
        payload = b"".join(_frame(b"k", key_token(x)) for x in value)
    elif type(value) is frozenset:
        payload = b"".join(_frame(b"k", x) for x in sorted(map(key_token, value)))
    elif isinstance(value, np.generic) and not value.dtype.hasobject:
        payload = _frame(b"d", repr(value.dtype.descr).encode()) + value.tobytes()
    else:
        payload = scalar_payload(value)
    return _frame(b"t", tag) + _frame(b"v", payload)


def dtype_description(dtype):
    return {
        "str": dtype.str,
        "descr": dtype.descr,
        "aligned": bool(dtype.isalignedstruct),
        "metadata": dict(dtype.metadata) if dtype.metadata is not None else None,
    }


def stage_mapping(path):
    return len(path) == 2 and path[0] == "stages" and type(path[1]) is int


def mapping_items(value, path):
    return {
        key_token(key): (key, item)
        for key, item in value.items()
        if not (stage_mapping(path) and type(key) is str and key == "wall_seconds")
    }


def dataclass_items(value):
    result = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    # Dynamic attributes are state too; do not silently restrict to declared fields.
    result.update(getattr(value, "__dict__", {}))
    return result


def tensor_parts(value):
    if value.layout != torch.strided or value.is_quantized:
        raise TypeError("Only dense, unquantized checkpoint tensors are supported")
    metadata = {
        "dtype": str(value.dtype), "shape": tuple(value.shape),
        "layout": str(value.layout), "stride": tuple(value.stride()),
        "requires_grad": bool(value.requires_grad), "names": tuple(value.names),
        "conjugate": bool(value.is_conj()), "negative": bool(value.is_neg()),
    }
    raw = value.detach().cpu().resolve_conj().resolve_neg().contiguous()
    # A uint8 view also supports bfloat16 and zero-dimensional tensors.
    payload = raw.reshape(-1).view(torch.uint8).numpy().tobytes()
    return metadata, payload


def hash_state(value, path=()):
    """Typed Merkle SHA256, insensitive only to the explicit stage-time field."""
    digest = hashlib.sha256()
    digest.update(_frame(b"t", type_name(value).encode()))
    if isinstance(value, torch.Tensor):
        metadata, payload = tensor_parts(value)
        digest.update(_frame(b"m", hash_state(metadata)))
        digest.update(_frame(b"v", payload))
    elif isinstance(value, np.ndarray):
        digest.update(_frame(b"m", hash_state({
            "dtype": dtype_description(value.dtype), "shape": tuple(value.shape),
        })))
        if value.dtype.hasobject:
            for index in np.ndindex(value.shape):
                digest.update(_frame(b"i", hash_state(value[index], path + (index,))))
        else:
            digest.update(_frame(b"v", value.tobytes(order="C")))
    elif isinstance(value, np.generic):
        digest.update(_frame(b"m", hash_state(dtype_description(value.dtype))))
        if value.dtype.hasobject:
            # Structured object scalars must expose values, not pointer bytes.
            if value.dtype.names:
                digest.update(_frame(b"v", hash_state({
                    name: value[name] for name in value.dtype.names
                }, path)))
            else:
                digest.update(_frame(b"v", hash_state(value.item(), path)))
        else:
            digest.update(_frame(b"v", value.tobytes()))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        digest.update(_frame(b"v", hash_state(dataclass_items(value), path)))
    elif isinstance(value, Mapping):
        items = mapping_items(value, path)
        for token in sorted(items):
            key, child = items[token]
            digest.update(_frame(b"k", token))
            digest.update(_frame(b"v", hash_state(child, path + (key,))))
    elif type(value) in {list, tuple}:
        for index, child in enumerate(value):
            digest.update(_frame(b"v", hash_state(child, path + (index,))))
    elif type(value) in {set, frozenset}:
        for token in sorted(map(key_token, value)):
            digest.update(_frame(b"v", token))
    else:
        digest.update(_frame(b"v", scalar_payload(value)))
    return digest.digest()


def path_string(path):
    result = "$"
    for part in path:
        if isinstance(part, tuple):
            result += "[" + ",".join(map(str, part)) + "]"
        elif type(part) is int:
            result += "[" + str(part) + "]"
        elif type(part) is str and part.isidentifier():
            result += "." + part
        else:
            result += "[" + repr(part) + "]"
    return result


class Comparison:
    def __init__(self):
        self.differences = []
        self.compared_nodes = 0
        self.ignored_paths = set()

    def different(self, path, reason, **details):
        self.differences.append({"path": path_string(path), "reason": reason, **details})

    def payload(self, left, right, path):
        if left == right:
            return
        extra = {
            "reference_payload_sha256": hashlib.sha256(left).hexdigest(),
            "candidate_payload_sha256": hashlib.sha256(right).hexdigest(),
            "reference_payload_bytes": len(left), "candidate_payload_bytes": len(right),
        }
        if len(left) == len(right):
            mismatch = np.frombuffer(left, dtype=np.uint8) != np.frombuffer(right, dtype=np.uint8)
            extra.update(different_bytes=int(np.count_nonzero(mismatch)),
                         first_different_byte=int(np.flatnonzero(mismatch)[0]))
        if len(left) <= 16 and len(right) <= 16:
            extra.update(reference_bits_hex=left.hex(), candidate_bits_hex=right.hex())
        self.different(path, "raw_bits", **extra)

    def mapping(self, left, right, path):
        if stage_mapping(path) and ("wall_seconds" in left or "wall_seconds" in right):
            self.ignored_paths.add(path_string(path + ("wall_seconds",)))
        a, b = mapping_items(left, path), mapping_items(right, path)
        for token in sorted(set(a) | set(b)):
            if token not in a:
                self.different(path + (b[token][0],), "missing_in_reference")
            elif token not in b:
                self.different(path + (a[token][0],), "missing_in_candidate")
            else:
                key, child = a[token]
                self.walk(child, b[token][1], path + (key,))

    def walk(self, left, right, path=()):
        self.compared_nodes += 1
        if type(left) is not type(right):
            self.different(path, "type", reference_type=type_name(left), candidate_type=type_name(right))
        elif isinstance(left, torch.Tensor):
            a, raw_a = tensor_parts(left)
            b, raw_b = tensor_parts(right)
            self.walk(a, b, path + ("<tensor_metadata>",))
            self.payload(raw_a, raw_b, path)
        elif isinstance(left, np.ndarray):
            self.walk(dtype_description(left.dtype), dtype_description(right.dtype), path + ("<dtype>",))
            if left.shape != right.shape:
                self.different(path, "shape", reference_shape=list(left.shape), candidate_shape=list(right.shape))
            elif left.dtype.hasobject != right.dtype.hasobject:
                self.different(path, "object_storage_kind")
            elif left.dtype.hasobject:
                for index in np.ndindex(left.shape):
                    self.walk(left[index], right[index], path + (index,))
            else:
                self.payload(left.tobytes(order="C"), right.tobytes(order="C"), path)
        elif isinstance(left, np.generic):
            self.walk(dtype_description(left.dtype), dtype_description(right.dtype), path + ("<dtype>",))
            if left.dtype.hasobject != right.dtype.hasobject:
                self.different(path, "object_storage_kind")
            elif left.dtype.hasobject:
                a = {name: left[name] for name in left.dtype.names} if left.dtype.names else left.item()
                b = {name: right[name] for name in right.dtype.names} if right.dtype.names else right.item()
                self.walk(a, b, path + ("<object_value>",))
            else:
                self.payload(left.tobytes(), right.tobytes(), path)
        elif dataclasses.is_dataclass(left) and not isinstance(left, type):
            self.mapping(dataclass_items(left), dataclass_items(right), path)
        elif isinstance(left, Mapping):
            self.mapping(left, right, path)
        elif type(left) in {list, tuple}:
            if len(left) != len(right):
                self.different(path, "length", reference_length=len(left), candidate_length=len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                self.walk(a, b, path + (index,))
        elif type(left) in {set, frozenset}:
            if set(map(key_token, left)) != set(map(key_token, right)):
                self.different(path, "set_members")
        else:
            self.payload(scalar_payload(left), scalar_payload(right), path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trusted_checkpoint(path):
    before = file_sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if before != file_sha256(path):
        raise ValueError("Checkpoint changed during read: " + str(path))
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    missing = REQUIRED_FIELDS - set(state)
    if missing:
        raise ValueError("Checkpoint is missing required fields: " + ", ".join(sorted(missing)))
    if not isinstance(state["learner"], Mapping) or "core" not in state["learner"]:
        raise ValueError("Checkpoint requires learner.core")
    if not isinstance(state["rng"], Mapping) or {"python", "numpy", "torch", "cuda"} - set(state["rng"]):
        raise ValueError("Checkpoint requires the complete global RNG mapping")
    if type(state["stages"]) is not list:
        raise ValueError("Checkpoint stages must be a list")
    return state, before


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    options = parser.parse_args(argv)
    report_path = options.report.expanduser().resolve()
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    inputs = [options.reference.expanduser().resolve(), options.candidate.expanduser().resolve()]
    if report_path in inputs or temporary.resolve() in inputs:
        parser.error("Report must not overwrite an input checkpoint")
    report = {
        "schema_version": 1, "passed": False,
        "reference": str(inputs[0]), "candidate": str(inputs[1]),
        "ignored_policy": IGNORED_POLICY,
        "difference_count_policy": "Mismatching comparison sites; not a count of unequal scalar elements.",
    }
    code = 2
    try:
        reference, reference_sha = load_trusted_checkpoint(inputs[0])
        candidate, candidate_sha = load_trusted_checkpoint(inputs[1])
        report.update(reference_file_sha256=reference_sha, candidate_file_sha256=candidate_sha)
        a, b = hash_state(reference).hex(), hash_state(candidate).hex()
        comparison = Comparison()
        comparison.walk(reference, candidate)
        if not comparison.differences and a != b:
            raise RuntimeError("Comparator/digest disagreement; refusing to pass")
        report.update(
            reference_state_sha256=a, candidate_state_sha256=b,
            compared_nodes=comparison.compared_nodes,
            ignored_paths=sorted(comparison.ignored_paths),
            ignored_path_count=len(comparison.ignored_paths),
            difference_count=len(comparison.differences), differences=comparison.differences,
            passed=not comparison.differences and a == b,
        )
        code = 0 if report["passed"] else 1
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(report_path)
    print(json.dumps({
        "passed": report["passed"], "difference_count": report.get("difference_count"),
        "report": str(report_path), **({"error": report["error"]} if "error" in report else {}),
    }, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
