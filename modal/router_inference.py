"""Modal inference server with learned router and multi-LoRA support.

Serves a Qwen3-VL model with multiple LoRA adapters and a trained routing head
that automatically selects the appropriate adapter for each request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

# Modal configuration matches training stack
CUDA_KEYRING_URL = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"

app = modal.App("moe-router-inference")

flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("wget", "gnupg", "build-essential")
    .run_commands(
        f"wget {CUDA_KEYRING_URL}",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "rm cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-6",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        flash_attn_wheel,
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.27.0",
        "peft>=0.11.0",
        "qwen-vl-utils",
        "huggingface-hub>=0.20.0",
        "Pillow>=10.0.0",
        "numpy<2",
    )
)

# Defaults
DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_ROUTER_PATH = os.environ.get("ROUTER_PATH", "/output/router-head.pt")
DEFAULT_LORA_CAL = os.environ.get("LORA_CALENDAR", "/checkpoints/calendar-tasks")
DEFAULT_LORA_CLAIM = os.environ.get("LORA_CLAIM", "/checkpoints/claim-window")
DEFAULT_LORA_PROVIDER = os.environ.get("LORA_PROVIDER", "/checkpoints/provider-select")

# Volumes
VOLUMES = {
    "/checkpoints": modal.Volume.from_name("moe-lora-checkpoints", create_if_missing=False),
    "/output": modal.Volume.from_name("routing-output", create_if_missing=False),
}


def get_hidden_size(model) -> int:
    """Extract hidden_size from a Qwen3-VL model."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model, "model") and hasattr(model.model, "config"):
        if hasattr(model.model.config, "hidden_size"):
            return model.model.config.hidden_size
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        return model.config.text_config.hidden_size
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.shape[1]
    raise AttributeError(f"Cannot find hidden_size in model config")


def build_router_head(hidden_size: int, num_labels: int):
    """Build router head architecture."""
    import torch
    from torch import nn

    class RouterHead(nn.Module):
        def __init__(self, hidden_size: int, num_labels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_labels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return RouterHead(hidden_size, num_labels)


@app.cls(
    image=IMAGE,
    gpu="A10G",
    volumes=VOLUMES,
    container_idle_timeout=300,
)
class RouterInference:
    """Inference class with router and multi-LoRA support."""

    @modal.enter()
    def load_models(self):
        """Load base model, LoRAs, and router head on container startup."""
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_id = DEFAULT_BASE_MODEL
        self.router_path = Path(DEFAULT_ROUTER_PATH)

        print(f"Loading base model: {self.base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)

        # Load base model for routing
        self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.encoder.eval()

        # Load router head
        print(f"Loading router head: {self.router_path}")
        checkpoint = torch.load(self.router_path, map_location=self.device, weights_only=False)
        self.adapters = checkpoint["adapters"]
        hidden_size = get_hidden_size(self.encoder)
        self.router_head = build_router_head(hidden_size, num_labels=len(self.adapters))
        self.router_head.load_state_dict(checkpoint["state_dict"])
        self.router_head.to(device=self.device, dtype=torch.bfloat16)
        self.router_head.eval()
        print(f"Router loaded with adapters: {self.adapters}")

        # Load base model with LoRAs for generation
        lora_paths = {
            "calendar": DEFAULT_LORA_CAL,
            "claim-window": DEFAULT_LORA_CLAIM,
            "provider-select": DEFAULT_LORA_PROVIDER,
        }
        # Filter to only adapters that exist in router
        lora_paths = {k: v for k, v in lora_paths.items() if k in self.adapters}

        print(f"Loading LoRAs: {list(lora_paths.keys())}")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Load first adapter
        first_name, first_path = next(iter(lora_paths.items()))
        self.model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)
        # Load remaining adapters
        for name, path in list(lora_paths.items())[1:]:
            self.model.load_adapter(path, adapter_name=name)
        self.model.eval()
        print("All models loaded successfully")

        # For vision processing
        self.process_vision_info = process_vision_info

    def _route(self, messages: list[dict[str, Any]]) -> str:
        """Use router to select the best adapter for the given messages."""
        import torch

        # Extract text for routing (similar to training data extraction)
        text = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        break
                if text:
                    break

        if not text:
            # Fallback to first adapter if no text found
            return self.adapters[0]

        # Tokenize and get router prediction
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.encoder(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, 0, :]  # First token of last layer
            logits = self.router_head(hidden)
            pred_idx = int(logits.argmax(dim=-1).item())

        return self.adapters[pred_idx]

    @modal.method()
    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        """Generate response using router-selected LoRA.

        Args:
            messages: List of message dicts with role and content (Qwen3-VL format)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dict with keys:
                - response: Generated text
                - adapter: Name of the adapter used
                - router_confidence: Confidence score for the routing decision
        """
        import torch

        # Route to appropriate adapter
        selected_adapter = self._route(messages)
        self.model.set_adapter(selected_adapter)

        # Prepare inputs
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info if present
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if image_inputs is not None:
            inputs["pixel_values"] = image_inputs.to(self.device)
        if video_inputs is not None:
            inputs["pixel_values_videos"] = video_inputs.to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        # Decode response (remove input prompt)
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )

        return {
            "response": response,
            "adapter": selected_adapter,
            "model": self.base_model_id,
        }


@app.function(
    image=IMAGE,
    volumes=VOLUMES,
)
@modal.web_endpoint(method="POST")
def inference_endpoint(request: dict) -> dict:
    """Web endpoint for inference requests.

    Example request:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Click on December 3rd"}
                ]
            }
        ],
        "max_new_tokens": 128,
        "temperature": 0.7
    }
    """
    messages = request.get("messages", [])
    if not messages:
        return {"error": "No messages provided"}

    inference = RouterInference()
    result = inference.generate.remote(
        messages=messages,
        max_new_tokens=request.get("max_new_tokens", 512),
        temperature=request.get("temperature", 0.7),
        top_p=request.get("top_p", 0.95),
    )

    return result


@app.local_entrypoint()
def test_inference(
    prompt: str = "Click on December 3rd in the calendar",
):
    """Local test: send a sample request."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inference = RouterInference()
    result = inference.generate.remote(messages=messages, max_new_tokens=128)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print(f"Adapter: {result['adapter']}")
    print(f"Response: {result['response']}")
    print(f"{'='*60}\n")
