import cv2
import time
import threading

class CameraStream:
    """Background camera capture."""
    def __init__(self, cam_index: int = 0,
                 res: tuple[int, int] | None = (1920, 1080),
                 warm: int = 10,
                 display_feed: bool = False) -> None:
        display_feed = True
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        while not self.cap.isOpened():
            time.sleep(0.2)
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            print(f"Waiting for camera {cam_index} to open...")
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
        # Sometimes, this camera breaks and returns all black.
        # If 90% of the pixels are pure black, we assume it's broken.
        # Kill the thread and restart it.
        while self.frame is None or (self.frame == 0).mean() > 0.9:
            print(f"Camera {self.cam_index} appears to be broken, restarting...")
            self.stop()
            self.__init__(self.cam_index)
            time.sleep(1)

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


if __name__ == "__main__":
    # Example usage
    stream = get_stream(cam_index=0, res=(640, 480), warm=5, display_feed=True)
    try:
        while True:
            frame = stream.read()
            if frame is not None:
                cv2.imshow("Camera Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        stream.stop()
        cv2.destroyAllWindows()