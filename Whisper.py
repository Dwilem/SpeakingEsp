import threading
import queue
import AudioServer
import torch

def vad_thread():
    """
    Drains audio_buffer independently of the playback thread.
    Runs silero-vad on each 512-sample frame (1:1 with ESP32 packets).
    Accumulates speech frames; flushes to whisper_queue on silence.
    """
    speech_frames  = []
    silence_count  = 0
    in_speech      = False

    print("[VAD] Ready.\n")

    while True:
        # Wait for a packet without holding the playback lock long
        with buffer_lock:
            if len(audio_buffer) > 0:
                frame_bytes = audio_buffer.popleft()
            else:
                # Buffer underrun — play silence to keep stream alive
                frame_bytes = None

        if frame_bytes is None:
            time.sleep(0.005)
            continue

        # Convert to float32 tensor for silero
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor  = torch.from_numpy(samples)

        with torch.no_grad():
            prob = vad_model(tensor, RATE).item()

        is_speech = prob >= VAD_THRESHOLD

        if is_speech:
            silence_count = 0
            in_speech     = True
            speech_frames.append(frame_bytes)
        else:
            if in_speech:
                silence_count += 1
                speech_frames.append(frame_bytes)  # keep trailing silence natural

                if silence_count >= SILENCE_FRAMES_TO_FLUSH:
                    if len(speech_frames) >= MIN_SPEECH_FRAMES:
                        segment = b"".join(speech_frames)
                        whisper_queue.put(segment)
                        print(f"[VAD] Flushed {len(speech_frames)} frames → Whisper queue")
                    speech_frames  = []
                    silence_count  = 0
                    in_speech      = False

        time.sleep(0.001)  # yield to other threads

# ─── NEW: Whisper thread ───────────────────────────────────────────
def whisper_thread():
    """
    Pops PCM blobs from whisper_queue, transcribes with faster-whisper,
    prints results with timestamps.
    """
    print("[Whisper] Ready.\n")
    while True:
        segment_bytes = whisper_queue.get()  # blocks until work arrives

        # Convert raw PCM → float32 array normalised to [-1, 1]
        audio_np = (
            np.frombuffer(segment_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        segments, info = whisper_model.transcribe(
            audio_np,
            language        = "en",
            beam_size       = 5,
            vad_filter      = False,   # we already gated with silero
            condition_on_previous_text = True,
        )

        for seg in segments:
            text = seg.text.strip()
            if text:
                print(f"[{seg.start:5.2f}s → {seg.end:5.2f}s]  {text}")

        whisper_queue.task_done()


threading.Thread(target=vad_thread,      daemon=True).start()
threading.Thread(target=whisper_thread,  daemon=True).start()