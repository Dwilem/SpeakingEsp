import socket
import struct
import threading
import numpy as np
import cv2
import sounddevice as sd
import signal
import sys
import time
from collections import deque

AUDIO_PORT = 5005
VIDEO_PORT = 5006
SAMPLE_RATE = 16000

stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\n[main] stopping...")
    stop_event.set()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ── Audio diagnostics ─────────────────────────────────────────────────────────

def audio_thread():
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    stream.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", AUDIO_PORT))
    sock.settimeout(1.0)
    print(f"[audio] listening on :{AUDIO_PORT}")

    packets_received = 0
    packets_dropped  = 0
    last_report      = time.time()
    last_packet_time = None

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
            now = time.time()

            # First packet ever
            if last_packet_time is None:
                print(f"[audio] first packet from {addr}, size={len(data)} bytes")

            # Gap detection — if more than 200ms passed between packets something stalled
            elif (now - last_packet_time) > 0.2:
                gap_ms = (now - last_packet_time) * 1000
                print(f"[audio] GAP detected: {gap_ms:.0f}ms since last packet")

            last_packet_time = now
            packets_received += 1

            # Validate packet size — should always be 960 bytes (480 samples × 2)
            expected = 480 * 2
            if len(data) != expected:
                print(f"[audio] WRONG SIZE: got {len(data)} bytes, expected {expected}")
                packets_dropped += 1
                continue

            samples = np.frombuffer(data, dtype=np.int16)

            # Check if audio is just silence (possible mic wiring issue)
            peak = np.max(np.abs(samples))
            if peak < 10:
                print(f"[audio] WARNING: near-silence, peak={peak} (mic disconnected?)")

            # Write to speaker — catch underrun/overrun
            try:
                stream.write(samples)
            except sd.PortAudioError as e:
                print(f"[audio] sounddevice error: {e}")
                packets_dropped += 1

            # Report stats every 5 seconds
            if now - last_report >= 5.0:
                print(f"[audio] stats: received={packets_received} "
                      f"dropped={packets_dropped} "
                      f"loss={packets_dropped/max(1,packets_received)*100:.1f}%")
                last_report = now

        except socket.timeout:
            if last_packet_time and (time.time() - last_packet_time) > 2.0:
                print(f"[audio] no packets for 2+ seconds — ESP32 stopped sending?")
            continue
        except Exception as e:
            print(f"[audio] unexpected error: {e}")

    stream.stop()
    sock.close()
    print(f"[audio] stopped — total received={packets_received} dropped={packets_dropped}")

# ── Video diagnostics ─────────────────────────────────────────────────────────

def video_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", VIDEO_PORT))
    sock.settimeout(1.0)
    print(f"[video] listening on :{VIDEO_PORT}")

    frames = {}
    frame_count      = 0
    chunks_received  = 0
    frames_dropped   = 0
    last_report      = time.time()
    last_frame_time  = None
    decode_times     = deque(maxlen=30)  # rolling average of decode time

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            if last_frame_time and (time.time() - last_frame_time) > 2.0:
                print(f"[video] no packets for 2+ seconds — ESP32 stopped sending?")
                last_frame_time = None  # only warn once
            continue
        except Exception as e:
            print(f"[video] socket error: {e}")
            continue

        if len(data) < 6:
            print(f"[video] runt packet: {len(data)} bytes, skipping")
            continue

        frame_id, chunk_idx, total_chunks = struct.unpack_from("<HHH", data, 0)
        payload = data[6:]
        chunks_received += 1

        # Warn if chunk index is out of range
        if chunk_idx >= total_chunks:
            print(f"[video] bad chunk: idx={chunk_idx} >= total={total_chunks}")
            continue

        frames.setdefault(frame_id, {
            "total":    total_chunks,
            "chunks":   {},
            "arrived":  time.time()
        })
        frames[frame_id]["chunks"][chunk_idx] = payload

        # Check if frame is complete
        entry = frames[frame_id]
        if len(entry["chunks"]) == entry["total"]:
            t0   = time.time()
            jpeg = b"".join(entry["chunks"][i] for i in range(entry["total"]))
            del frames[frame_id]

            # Decode
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            decode_ms = (time.time() - t0) * 1000
            decode_times.append(decode_ms)

            if img is None:
                print(f"[video] frame {frame_id} DECODE FAILED "
                      f"({len(jpeg)} bytes — corrupted JPEG?)")
                frames_dropped += 1
            else:
                frame_count += 1
                last_frame_time = time.time()
                cv2.imshow("ESP32 Camera", img)
                cv2.waitKey(1)

        # Evict stale frames and report them
        if len(frames) > 30:
            oldest_id = min(frames)
            oldest    = frames[oldest_id]
            missing   = oldest["total"] - len(oldest["chunks"])
            age_ms    = (time.time() - oldest["arrived"]) * 1000
            print(f"[video] dropping stale frame {oldest_id}: "
                  f"got {len(oldest['chunks'])}/{oldest['total']} chunks, "
                  f"missing={missing}, age={age_ms:.0f}ms")
            frames_dropped += 1
            del frames[oldest_id]

        # Stats every 5 seconds
        now = time.time()
        if now - last_report >= 5.0:
            avg_decode = sum(decode_times) / max(1, len(decode_times))
            print(f"[video] stats: frames_shown={frame_count} "
                  f"dropped={frames_dropped} "
                  f"chunks={chunks_received} "
                  f"avg_decode={avg_decode:.1f}ms "
                  f"incomplete_in_buffer={len(frames)}")
            last_report = now

    sock.close()
    cv2.destroyAllWindows()
    print(f"[video] stopped — shown={frame_count} dropped={frames_dropped}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=audio_thread, daemon=True).start()
    print("[main] press Ctrl+C to stop")
    video_thread()