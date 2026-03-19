#!/usr/bin/env python3
"""
check_valid.py
--------------
Validates a .root output file. Called by simulate.py (idempotency guard)
and by resubmit.py (to find incomplete tasks).

Usage:
    python check_valid.py /path/to/output_process_3.root
    echo $?   # 0 = valid, 1 = invalid/missing
"""

import os
import sys

import uproot


def is_valid_root_file(path: str) -> bool:
    """
    Returns True if the ROOT file exists and passes basic integrity checks.

    Replace the body of this function with a real check using your ROOT
    installation. Two common approaches:

    Option A — uproot (pure Python, no ROOT install needed):
        import uproot
        try:
            with uproot.open(path) as f:
                # Optionally check that expected trees/keys exist:
                # assert "events" in f
                return True
        except Exception:
            return False

    Option B — PyROOT (requires ROOT installation):
        import ROOT
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie() or f.TestBit(ROOT.TFile.kRecovered):
            return False
        f.Close()
        return True
    """
    if not os.path.exists(path):
        return False

    if os.path.getsize(path) == 0:
        return False

    # --- Replace this block with Option A or B above ---
    # Minimal check: file exists and is non-empty (placeholder only)
    # try:
        # with open(path, "rb") as f:
        #     header = f.read(4)
    try:
        with uproot.open(path) as f:
            # check that expected trees/keys exist:
            assert "events" in f
            return True
    except Exception:
        return False
        # return len(header) > 0  # placeholder: just check non-empty
    # except OSError:
    #     return False
    # --- End placeholder ---


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_root_file>")
        sys.exit(2)

    path = sys.argv[1]
    valid = is_valid_root_file(path)
    if valid:
        print(f"OK: {path}")
        sys.exit(0)
    else:
        print(f"INVALID or MISSING: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()