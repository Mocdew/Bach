"""Minimal runner so the suite works without pytest installed.

`python -m pytest tests/ -q` is the normal path. This exists for sandboxes with
no network. It supports exactly what the suite uses: module-scoped fixtures
injected by parameter name.
"""

from __future__ import annotations

import inspect
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tests.test_recoup as mod  # noqa: E402

FIXTURES = ("sim", "fitted")


def main() -> int:
    cache: dict = {}

    def build(name: str):
        if name not in cache:
            fn = getattr(mod, name)
            cache[name] = fn(*[build(p) for p in inspect.signature(fn).parameters])
        return cache[name]

    for name in FIXTURES:
        build(name)

    tests = [(n, f) for n, f in vars(mod).items()
             if n.startswith("test_") and callable(f)]
    tests.sort(key=lambda kv: kv[0])

    passed, failed = 0, []
    t0 = time.time()
    for name, fn in tests:
        args = [cache[p] for p in inspect.signature(fn).parameters]
        try:
            fn(*args)
            passed += 1
            print(".", end="", flush=True)
        except Exception:
            failed.append((name, traceback.format_exc()))
            print("F", end="", flush=True)

    print(f"\n\n{passed} passed, {len(failed)} failed in {time.time() - t0:.1f}s")
    for name, tb in failed:
        print(f"\n--- {name} " + "-" * (70 - len(name)))
        print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
