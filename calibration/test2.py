import cv2
import threading
import time
from queue import Queue, Empty, Full


class IPStreamReader:
    def __init__(self, stream_url, name="Camera"):
        self.stream_url = stream_url
        self.name = name
        self.frame_queue = Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(
            target=self._update,
            name=self.name,
            daemon=True
        )

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        cap = cv2.VideoCapture(self.stream_url)

        # May not work on all backends, but worth trying
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            grabbed, frame = cap.read()

            if not grabbed:
                print(f"[{self.name}] Failed to grab frame. Reconnecting...")
                time.sleep(1)

                cap.release()
                cap = cv2.VideoCapture(self.stream_url)
                continue

            try:
                # Remove old frame if queue already full
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()

                self.frame_queue.put_nowait(frame)

            except (Empty, Full):
                pass

        cap.release()

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.stopped = True
        self.thread.join()


def main():
    # Replace with your phone IP Webcam URLs
    URL_B = "http://192.168.31.182:8080/video" 
    URL_A = "http://192.168.31.214:8080/video"
    print("Initializing streams...")

    stream_A = IPStreamReader(
        URL_A,
        name="Phone_A"
    ).start()

    stream_B = IPStreamReader(
        URL_B,
        name="Phone_B"
    ).start()

    # Allow streams to warm up
    time.sleep(2)

    prev_time = time.perf_counter()

    while True:
        frame_A = stream_A.get_frame()
        frame_B = stream_B.get_frame()

        if frame_A is not None:
            cv2.imshow("Phone A Stream", frame_A)

        if frame_B is not None:
            cv2.imshow("Phone B Stream", frame_B)

        # FPS calculation
        current_time = time.perf_counter()
        dt = current_time - prev_time

        if dt > 0:
            fps = 1.0 / dt
            print(f"Main Loop Speed: {fps:.2f} FPS", end="\r")

        prev_time = current_time

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\nShutting down cleanly...")

    stream_A.stop()
    stream_B.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()