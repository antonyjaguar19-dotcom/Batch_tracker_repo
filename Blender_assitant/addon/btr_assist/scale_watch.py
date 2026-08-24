"""Watching the pattern box while it tracks: when did it stop being the artist's feature?

With the motion model on `LocScale` Blender solves a SIZE for the pattern box on every
frame, not only a position. That size is a measurement like any other, and it is the one
that says something the position cannot: a box that swells has stopped locking onto the
feature and started locking onto whatever surrounds it. A drifting position looks like
tracking. A box that doubles over eight frames is the tracker telling you it is no longer
sure what it is matching -- and the frames it has already written are the ones nobody looks
at, because the track is still alive and still moving plausibly.

So this is a leash on scale, in the same spirit as the position leash in `track_core`: it
does not decide anything about the plate, it decides WHEN TO STOP AND LOOK. What the swell
actually means is a pixel question, and it is answered where the pixels are
(`sidecar/patmatch.drift_report`).

Two thresholds, because they catch two different failures:

  * `rate` -- fractional size change from ONE frame to the next. A real feature approaching
    camera grows smoothly; a box that jumps 30 % in a frame has jumped onto something else.
  * `ratio` -- cumulative size against the box the artist set. A slow, steady creep never
    trips `rate` and is the more common way a tracker rots: 4 % a frame for fifteen frames
    is a doubling and every single step looked reasonable.

`onset` is the third number and the one the repair needs. The frame that trips a threshold
is not the frame the trouble started: it is where the accumulated evidence crossed a line.
The onset is the first frame after which the box never returned to the artist's size band,
and that is the last frame worth trusting -- everything after it was measured by a box that
was already growing.

No bpy import here on purpose: this is arithmetic over a sequence of numbers, so it is
testable without Blender (`tests/test_scale_watch.py`) rather than only inside a running
tracking session.
"""


class ScaleWatch:
    """One track's pattern-box size, frame by frame.

    `size` is a single linear number in plate pixels -- the geometric mean of the box's
    width and height. Using the mean rather than either side keeps a box that stretches on
    one axis only (an edge point sliding) from reading as twice the growth it is, and keeps
    the two thresholds comparable to each other.
    """

    def __init__(self, size, rate=0.12, ratio=1.6, onset=1.06):
        self.base = float(size)
        self.prev = float(size)
        self.prev_frame = None
        self.rate = float(rate)
        self.ratio = float(ratio)
        self.onset_band = float(onset)
        self.onset_frame = None
        self.peak = 1.0            # worst cumulative departure seen, for the report

    # ---------------------------------------------------------------- api

    def rebase(self, size, frame=None):
        """Accept the current size as the new normal.

        Called after a repair: whatever the box is now IS the reference, otherwise the next
        frame trips the same cumulative threshold again and the loop repairs forever. Also
        called when a swell is judged to be a real change of scale -- the feature genuinely
        got bigger, and refusing to update the baseline would make every later frame a
        violation of a size that no longer exists.
        """
        self.base = self.prev = float(size)
        self.prev_frame = frame
        self.onset_frame = None
        self.peak = 1.0

    def feed(self, frame, size):
        """One frame's box size. Returns None, or the flag that stops this track.

        A flag is not a death. It is "this track has to be looked at before it writes
        another frame", and the caller keeps the marker it already has.
        """
        size = float(size)
        if size <= 0.0 or self.base <= 0.0:
            return None
        cum = size / self.base
        dev = cum if cum >= 1.0 else 1.0 / cum
        if dev > self.peak:
            self.peak = dev

        # Onset tracking runs BEFORE the thresholds, so a flag can report the frame the
        # departure started rather than the frame it was noticed.
        if dev <= self.onset_band:
            self.onset_frame = None
        elif self.onset_frame is None:
            self.onset_frame = int(frame)

        step = size / self.prev if self.prev > 0.0 else 1.0
        step_dev = step if step >= 1.0 else 1.0 / step
        self.prev, self.prev_frame = size, int(frame)

        if self.rate > 0.0 and (step_dev - 1.0) > self.rate:
            return self._flag(frame, size, cum, step, "rate")
        if self.ratio > 1.0 and dev > self.ratio:
            return self._flag(frame, size, cum, step, "ratio")
        return None

    # ---------------------------------------------------------------- internals

    def _flag(self, frame, size, cum, step, reason):
        return {
            "frame": int(frame),
            "reason": reason,
            "size": size,
            "base": self.base,
            "ratio": cum,
            "step": step,
            # The onset can legitimately be this frame (a single jump), and is never later
            # than it.
            "onset": int(self.onset_frame if self.onset_frame is not None else frame),
            "text": ("pattern box %s %.0f%% in one frame"
                     % ("grew" if step >= 1.0 else "shrank", abs(step - 1.0) * 100.0)
                     if reason == "rate" else
                     "pattern box is %.2fx the size you set" % cum),
        }


def size_of(pattern_px):
    """(w, h) in plate pixels -> the single number the watch compares."""
    w, h = float(pattern_px[0]), float(pattern_px[1])
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return (w * h) ** 0.5
