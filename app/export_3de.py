# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, List, Tuple

def write_tracks_txt(out_path: str, tracks: Dict[str, List[Tuple[int, float, float]]], end_frame: int) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Safe sorting: tries to sort numerically, falls back to string comparison if text is present
    def safe_sort(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, str(s))

    names = sorted(tracks.keys(), key=safe_sort)
    
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(names)}\n")
        for name in names:
            # 3D Equalizer supports string names natively
            f.write(f"{name}\n")
            # Color ID
            f.write("0\n")
            # FIX: Write the actual number of tracked coordinates, NOT the total video length!
            f.write(f"{len(tracks[name])}\n")
            
            # Write the coordinates
            for fr, x, y in tracks[name]:
                f.write(f"{int(fr)} {float(x):.12f} {float(y):.12f}\n")