#
# 3DE4.script.name:    Import from SynthEyes...
# 3DE4.script.version:    v1.4
# 3DE4.script.gui:    Main Window::3DE4::File::Import
# 3DE4.script.comment:    Converts SynthEyes text tracks to 3DE4 points.
#

import tde4
import os

def import_syntheyes_tracks():
    cam = tde4.getCurrentCamera()
    pg = tde4.getCurrentPGroup()
    
    if not cam or not pg:
        tde4.postQuestionRequester("Error", "Please make sure a camera and point group are selected.", "OK")
        return

    req = tde4.createCustomRequester()
    tde4.addFileWidget(req, "file_path", "SynthEyes Text File", "*.txt")
    
    ret = tde4.postCustomRequester(req, "Import SynthEyes Tracks", 600, 0, "Import", "Cancel")
    if ret == 1:
        file_path = tde4.getWidgetValue(req, "file_path")
        if file_path and os.path.exists(file_path):
            w = tde4.getCameraImageWidth(cam)
            h = tde4.getCameraImageHeight(cam)
            
            tracks = {}
            current_tracker = None
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) == 1:
                            current_tracker = parts[0]
                            tracks[current_tracker] = []
                        elif len(parts) >= 3 and current_tracker:
                            frame = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            tracks[current_tracker].append((frame, x, y))
            except Exception as e:
                tde4.postQuestionRequester("Error", "Failed to parse file: " + str(e), "OK")
                return
            
            for trk_name, pts in tracks.items():
                if not pts:
                    continue

                point = tde4.createPoint(pg)
                tde4.setPointName(pg, point, trk_name)

                pts_sorted = sorted(pts, key=lambda p: p[0])

                # First and last frame in 3DE's 1-based numbering
                start_frame = pts_sorted[0][0] + 1
                end_frame   = pts_sorted[-1][0] + 1

                # Build the tracking curve
                tracking_curve = []
                for pt in pts_sorted:
                    x_norm = pt[1] / w
                    y_norm = pt[2] / h
                    tracking_curve.append([x_norm, y_norm])

                # Write all frames in one call
                tde4.setPointPosition2DBlock(pg, point, cam, start_frame, tracking_curve)

                # --- THE FIX: mark the last frame as the end of this track ---
                # This tells 3DE "the track stops here" so it doesn't
                # bleed/interpolate beyond the tracker's actual range
                tde4.setPointStatus2D(pg, point, cam, end_frame, "POINT_KEYFRAME_END")

            tde4.postQuestionRequester("Success", "Imported {} tracks.".format(len(tracks)), "OK")

import_syntheyes_tracks()