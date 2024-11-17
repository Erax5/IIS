from typing import Optional
import numpy as np
import cv2
import opencv_jupyter_ui as jcv2


def cv2_write_text(img: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Helper function for writing text with a black border, in a predefined font."""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 6)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


def cv2_snap() -> Optional[np.ndarray]:
    """
    Utility for taking snapshots with your webcam.

    * If run from the CLI, it opens a normal OpenCV window. Use SPACE to take a snapshot and ESC to finish.
    * If run from a notebook, it uses `opencv-jupyter-ui` to embed a view in the notebook.
    * On completion, returns the snapshot as an RGB `ndarray`.
    * If no snapshot was taken, returns `None`.
    """

    cam = cv2.VideoCapture(0)
    snap = None

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                raise("OpenCV found an error reading the next frame.")

            (h, w, c) = frame.shape
            display = np.zeros_like(frame, shape=(h, w*2, c))
            display[:, :w, :] = frame

            if snap is not None:
                display[:, w:, :] = snap

            cv2_write_text(display, "Press SPACE to take a pic", ((5 * w) // 4, (8 * h) // 10))
            cv2_write_text(display, "Press ESC to exit", ((5 * w) // 4, (9 * h) // 10))

            jcv2.imshow("snap", display)

            key = jcv2.waitKey(1) & 0xFF
            if key == 27: # ESC to exit
                break
            elif key == 32: # SPACE to take snap
                snap = frame
    finally:
        cam.release()
        jcv2.destroyAllWindows()

    # Need to convert to RGB format for compatibility with everyone else.
    if snap is not None:
        snap = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)

    return snap


def cv2_vid() -> tuple[float, np.ndarray]:
    """
    Utility for taking videos with your webcam.

    * If run from the CLI, it opens a normal OpenCV window. Use SPACE to start/stop recording, and ESC to finish.
    * If run from a notebook, it uses `opencv-jupyter-ui` to embed a view in the notebook.
    * On completion, returns a tuple `(fps, video)`, with `video` in RGB.
    """

    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)

    video = []
    playback_idx = 0

    last_recording = False
    recording = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                raise("OpenCV found an error reading the next frame.")


            (h, w, c) = frame.shape
            display = np.zeros_like(frame, shape=(h, w*2, c))
            display[:, :w, :] = frame

            if len(video) > 0:
                playback_idx = (playback_idx + 1) % len(video)
                display[:, w:, :] = video[playback_idx]

            if recording:
                if not last_recording:
                    video = []
                cv2_write_text(img=display, text="RECORDING. Press SPACE to stop.", position=((11 * w) // 10, (8 * h) // 10), color=(127, 127, 255))
                video.append(frame)
            else:
                cv2_write_text(img=display, text="STOPPED. Press SPACE to start.", position=((11 * w) // 10, (8 * h) // 10), color=(127, 127, 127))

            last_recording = recording

            cv2_write_text(img=display, text="Press ESC to exit", position=((11 * w) // 10, (9 * h) // 10))

            jcv2.imshow("vid", display)

            key = jcv2.waitKey(1) & 0xFF
            if key == 27: # ESC to exit
                break
            elif key == 32: # SPACE to switch from recording to not.
                recording = not recording
    finally:
        cam.release()
        jcv2.destroyAllWindows()

    # Need to convert to RGB format for compatibility with everyone else.
    if len(video) == 0:
        converted = np.empty(shape=())
    else:
        converted = np.stack([ cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video ], axis=0)

    return fps, converted
