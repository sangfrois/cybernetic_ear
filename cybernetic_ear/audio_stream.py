import pyaudio
import numpy as np
import threading
import time

class AudioStream:
    """
    Manages a real-time audio stream from the microphone using PyAudio.
    """
    def __init__(self, rate=22050, chunk_size=1024, channels=1):
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paFloat32

        self._p = pyaudio.PyAudio()
        self._stream = None
        self._running = False
        self._thread = None
        self.callbacks = []

    @staticmethod
    def list_devices():
        """Prints available PyAudio input devices."""
        p = pyaudio.PyAudio()
        print("--- Available Audio Input Devices ---")
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print(f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
        print("------------------------------------")
        p.terminate()

    def add_callback(self, callback):
        """Adds a callback function to be called with each new audio chunk."""
        self.callbacks.append(callback)

    def _read_stream(self):
        """
        The main loop that reads from the audio stream and calls callbacks.
        """
        print("[AudioStream Thread] Read loop started.")
        while self._running:
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                # print(f"[AudioStream Thread] Read {len(chunk)} frames.") # Uncomment for extreme verbosity
                for callback in self.callbacks:
                    callback(chunk)
            except IOError as e:
                print(f"[AudioStream Thread] IOError: {e}")
                self._running = False
            except Exception as e:
                print(f"[AudioStream Thread] Unexpected error: {e}")
                self._running = False
        print("[AudioStream Thread] Read loop finished.")


    def start(self):
        """Starts the audio stream and the reading thread."""
        if self._running:
            print("[AudioStream] Stream is already running.")
            return

        print("[AudioStream] Starting stream...")
        try:
            self._stream = self._p.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk_size)
            print("[AudioStream] PyAudio stream opened successfully.")
        except Exception as e:
            print(f"[AudioStream] Failed to open stream: {e}")
            self._p.terminate()
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._read_stream)
        self._thread.daemon = True
        self._thread.start()
        print("[AudioStream] Stream started.")

    def stop(self):
        """Stops the audio stream and joins the reading thread."""
        if not self._running:
            # print("[AudioStream] Stream is not running.")
            return

        print("[AudioStream] Stopping stream...")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            print("[AudioStream] PyAudio stream closed.")
        
        self._p.terminate()
        print("[AudioStream] PyAudio terminated.")

if __name__ == '__main__':
    # Example usage and test
    AudioStream.list_devices()

    def print_chunk_info(chunk):
        print(f"Callback received chunk of size: {len(chunk)}")

    audio_stream = AudioStream()
    audio_stream.add_callback(print_chunk_info)
    audio_stream.start()
    
    try:
        time.sleep(5)
    finally:
        audio_stream.stop()
