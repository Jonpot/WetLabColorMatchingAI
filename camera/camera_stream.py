import cv2
import time
import threading

class CameraStream:
    """Background camera capture."""
    def __init__(self, cam_index: int = 0,
                 res: tuple[int, int] | None = (1600, 1200),
                 warm: int = 10,
                 display_feed: bool = False) -> None:
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
        if res:
            w, h = res
            self.cap.set(3, w)
            self.cap.set(4, h)
            time.sleep(0.2)
        for _ in range(warm):
            self.cap.read()
            time.sleep(0.04)
        self.frame = None
        self.running = True
        self.display_feed = display_feed
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ret, frm = self.cap.read()
            if ret:
                self.frame = frm
                if self.display_feed:
                    cv2.imshow(f"Camera {self.cam_index}", frm)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self) -> None:
        self.running = False
        self.thread.join()
        self.cap.release()

_streams: dict[int, CameraStream] = {}


def get_stream(cam_index: int = 0,
               res: tuple[int, int] | None = (1600, 1200),
               warm: int = 10,
               display_feed: bool = False) -> CameraStream:
    """Return a running CameraStream for the given index."""
    stream = _streams.get(cam_index)
    if stream is None:
        stream = CameraStream(cam_index, res=res, warm=warm, display_feed=display_feed)
        _streams[cam_index] = stream
    return stream
