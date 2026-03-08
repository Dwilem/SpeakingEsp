from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import login
from PIL import Image
from dotenv import load_dotenv
import torch
import io
import os

load_dotenv() 

login(os.getenv("login"))
# ── Load model once at startup (slow, ~10-30s) ───────────────────────────────
# Do this ONCE when your server starts, not on every request

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"  # good balance of speed/quality

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,   # use float16 to save VRAM
    device_map="auto",            # automatically use GPU if available, else CPU
    quantization_config=quantization_config,            # 4-bit quantization — cuts VRAM usage in half
                                  # requires: pip install bitsandbytes
)


def history_to_prompt(history: list[dict], current_prompt: str) -> str:
    """
    Convert conversation history into LLaVA's expected prompt format.
    LLaVA uses a specific [INST] template for conversation turns.
    """
    conversation = ""

    for msg in history:
        if msg["role"] == "user":
            conversation += f"[INST] {msg['content']} [/INST]"
        elif msg["role"] == "assistant":
            conversation += f" {msg['content']} "

    # Add current message — <image> token tells the model where the image goes
    conversation += f"[INST] <image>\n{current_prompt} [/INST]"

    return conversation


def chat_with_vision(
    prompt: str,
    image_bytes: bytes,       # raw JPEG/PNG bytes, e.g. directly from UDP socket
    history: list[dict],
    max_new_tokens: int = 300,
) -> str:
    """
    Send a message with an image and conversation history to a local HuggingFace model.

    Args:
        prompt:          The user's current text message
        image_bytes:     Raw image bytes (from file, webcam, or UDP)
        history:         List of previous messages
        max_new_tokens:  How long the response can be

    Returns:
        The assistant's response as a string
    """

    # Convert raw bytes to PIL Image (what the model expects)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Build the full prompt string with history
    full_prompt = history_to_prompt(history, prompt)

    # Tokenize — processor handles both text and image together
    inputs = processor(
        text=full_prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,    # higher = more creative, lower = more focused
            top_p=0.9,
        )

    # Decode only the NEW tokens (skip the input prompt tokens)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    response = processor.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    history = [
        {"role": "user", "content": "Hello, can you see what I'm showing you?"},
        {"role": "assistant", "content": "Yes! I can see images you share. What would you like to know?"},
    ]

    # Load image as raw bytes (mirrors how you'd receive it from UDP)
    with open("test.jpg", "rb") as f:
        image_bytes = f.read()

    response = chat_with_vision(
        prompt="What objects can you see? Describe the scene.",
        image_bytes=image_bytes,
        history=history,
    )

    print(f"Assistant: {response}")

    # Update history after each turn
    history.append({"role": "user", "content": "What objects can you see?"})
    history.append({"role": "assistant", "content": response})