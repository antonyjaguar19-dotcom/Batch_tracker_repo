"""Entry point: python -m sidecar [--port N] [--portfile P] [--token T] [--parent PID]"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0,
                    help="0 = let the OS pick and write it to the portfile")
    ap.add_argument("--portfile", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--parent", type=int, default=0,
                    help="PID to follow; the sidecar exits when it does")
    a = ap.parse_args()
    server.serve(port=a.port, portfile=a.portfile or None,
                 token=a.token or None, parent_pid=a.parent)


if __name__ == "__main__":
    main()
