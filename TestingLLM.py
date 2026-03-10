from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from huggingface_hub import login
from PIL import Image
from dotenv import load_dotenv
import torch
import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Usage: python ask_local.py image.jpg "What is in this image?"
image_path = "./test.jpg"

#MODEL = "llava-hf/llava-1.5-7b-hf"
MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # swap for any vision model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

processor = AutoProcessor.from_pretrained(MODEL)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=quantization_config
).to(device)


def chat_with_vision( history=[], max_new_tokens=128) -> str:
    #print("\nHistory:", history)
    prompt = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(text=prompt, images=image_inputs, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3, top_p=0.9)

    new_tokens = output[0][input_len:]
    response = processor.decode(new_tokens, skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":

    load_dotenv(".env")  # Load environment variables from .env file
    login(os.getenv("login"))

    history = [
        {
            "role": "system",
            "content": """You are a friendly, casual conversational partner. 
                        You talk like a real person having a chat, not an AI assistant.
                        Rules:
                        - NEVER start responses with image descriptions
                        - Only describe the image if the user explicitly asks "what do you see" or "describe the image"
                        - React to what the user says, ask follow up questions
                        - Be curious, friendly, and engaging
                        - Keep responses short, 1-3 sentences max
                        - If user tells you something about themselves or the image, acknowledge it and respond naturally
                        """
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, can you see what I'm showing you? It is an image of a me, Augustas"},
                ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Yes! I can see images you share. What would you like to know?"}],
        },
    ]

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        history.append( {
            "role": "user",
            "content": [{"type": "text", "text": f"Respond naturally to this, do not describe the image: {question}"},
                        {"type": "image", "image": image_path}
                        ],
        })

        response = chat_with_vision(
            history=history,
        )

        history.pop()  # Remove the last user input to avoid repeating the image in the next turn

        print(f"Assistant: {response}")

        # Update history after each turn
        history.append( {
            "role": "user",
            "content": [{"type": "text", "text": f"Respond naturally to this, do not describe the image: {question}"}],
        })
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        })
    
