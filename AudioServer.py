import socket
import pyaudio
import struct
import numpy as np
import threading
from collections import deque
import time
import torch
import queue
from faster_whisper import WhisperModel

UDP_IP    = "0.0.0.0"
UDP_PORT  = 5005
CHUNK     = 512
RATE      = 16000

# VAD config ───────────────────────────────────────────────
VAD_THRESHOLD          = 0.5   # silero speech probability cutoff
SILENCE_FRAMES_TO_FLUSH = 16   # ~500ms of silence triggers a flush
MIN_SPEECH_FRAMES      = 4     # ignore bursts shorter than ~120ms

# ─── Jitter buffer ─────────────────────────────────────────────────
# Holds incoming packets, playback thread drains it steadily
JITTER_BUFFER_SIZE = 20       # packets to pre-fill before playing
MAX_BUFFER_SIZE    = 100       # drop old packets if buffer grows too large
audio_buffer       = deque()
buffer_lock        = threading.Lock()
buffer_ready       = threading.Event()

# ───────────────────────────────────────────────────────────────────

print("[Init] Loading silero-vad...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    verbose=False,
)
vad_model.eval()
print("[Init] Loading faster-whisper (tiny, cuda)...")
whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")
print("[Init] Models ready.\n")

# ─── NEW: Queue between VAD thread and Whisper thread ─────────────
whisper_queue = queue.Queue()


p = pyaudio.PyAudio()
stream = p.open(
    format            = pyaudio.paInt16,
    channels          = 1,
    rate              = RATE,
    output            = True,
    frames_per_buffer = CHUNK
)



# ─── Playback thread — drains buffer at steady rate ────────────────
def playback_thread():
    print("[Player] Waiting for buffer to fill...")
    buffer_ready.wait()     # wait until jitter buffer has enough packets
    print("[Player] Playing!\n")

    while True:
        if len(audio_buffer) > 0:
            audio = audio_buffer[0]
        else:
            # Buffer underrun — play silence to keep stream alive
            audio = bytes(CHUNK * 2)

        stream.write(audio, exception_on_underflow=False)

player = threading.Thread(target=playback_thread, daemon=True)
player.start()

# ─── Network thread — receives UDP packets ─────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"=== UDP Audio Server ===")
print(f"Listening on port {UDP_PORT}\n")

try:
    while True:
        data, addr = sock.recvfrom(65535)
        if len(data) < 4:
            continue

        seq   = struct.unpack_from('<I', data, 0)[0]
        audio = data[4:]

        with buffer_lock:
            # Drop oldest packet if buffer is overflowing
            if len(audio_buffer) >= MAX_BUFFER_SIZE:
                audio_buffer.popleft()
                print("[WARN] Buffer overflow, dropping old packet")

            audio_buffer.append(bytes(audio))

            # Signal playback to start once pre-fill is reached
            if not buffer_ready.is_set() and len(audio_buffer) >= JITTER_BUFFER_SIZE:
                buffer_ready.set()

        # ── Print level every 100 packets ──────────────────────────
        if seq % 100 == 0:
            samples = np.frombuffer(audio, dtype=np.int16)
            rms     = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            bar     = min(int(rms / 500), 30)
            buf_len = len(audio_buffer)
            print(f"#{seq:05d} | RMS:{rms:7.1f} | {'█' * bar:<30} | buf={buf_len}")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    sock.close()