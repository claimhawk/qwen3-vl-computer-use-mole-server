#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

# Version: 2025-12-02-v6 - Separate prompts: ROUTER for routing, COMPUTER_USE for task LoRAs
"""Stacked inference: Routing LoRA → Task LoRA or Chandra.

Uses the trained routing LoRA to classify which adapter/model to use:
- calendar, claim-window, provider-select → Qwen + task LoRA
- chandra → Chandra OCR model → Qwen formats result as tool_call

Usage:
    modal run modal/stacked_inference.py --test
    modal run modal/stacked_inference.py --sample-idx 0
"""

import json
from pathlib import Path
from typing import Any

import modal

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================
# Volume names and paths are loaded from config/adapters.yaml via the SDK.
# Users can customize these by editing the YAML file.
# Fallbacks are provided for Modal remote execution where SDK may not be available.

try:
    from sdk.modal_compat import (
        get_volume_name,
        get_router_inference_path,
        get_base_vlm,
        get_ocr_model,
    )
    INFERENCE_VOLUME_NAME = get_volume_name("inference")
    INFERENCE_LORAS_PATH = "/inference/loras"  # Parent path, not per-expert
    DEFAULT_ROUTING_CHECKPOINT = get_router_inference_path()
    BASE_MODEL = get_base_vlm()
    CHANDRA_MODEL = get_ocr_model()
except ImportError:
    # Fallback for Modal remote execution
    INFERENCE_VOLUME_NAME = "moe-inference"
    INFERENCE_LORAS_PATH = "/inference/loras"
    DEFAULT_ROUTING_CHECKPOINT = "/inference/routing/adapter"
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
    CHANDRA_MODEL = "datalab-to/chandra"

app = modal.App("stacked-lora-inference")

# Volumes - single inference volume for all deployed assets
inference_volume = modal.Volume.from_name(INFERENCE_VOLUME_NAME, create_if_missing=False)

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.26.0",
        "peft>=0.14.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
        "chandra-ocr",
    )
)

# Server-side system prompt for computer use tasks
COMPUTER_USE_SYSTEM_PROMPT = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
  "name": "computer_use",
  "description": "Perform computer actions",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "scroll", "hscroll", "wait", "terminate", "answer", "ocr"]
      },
      "coordinate": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "X and Y coordinates in 1000x1000 normalized space"
      },
      "text": {
        "type": "string",
        "description": "Text content for ocr action"
      }
    },
    "required": ["action"]
  }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Action descriptions

* `key`: Press keys in order, release in reverse.
* `type`: Type a string of text.
* `mouse_move`: Move the cursor to (x, y).
* `left_click`: Left click at (x, y).
* `left_click_drag`: Click and drag from current to (x, y).
* `right_click`: Right click at (x, y).
* `middle_click`: Middle click at (x, y).
* `double_click`: Double-click at (x, y).
* `triple_click`: Triple-click at (x, y).
* `scroll`: Scroll the mouse wheel.
* `hscroll`: Horizontal scroll.
* `wait`: Wait N seconds.
* `terminate`: End the task with a status.
* `answer`: Answer a question.
* `ocr`: Return extracted text from the image.

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) One or more <tool_call>...</tool_call> blocks, one per line, each containing only the JSON.

Rules:
- Output exactly in the order: Action, <tool_call>(s).
- Be brief: one sentence for Action.
- Multiple tool calls can be output, one per line.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""

# Router system prompt - for routing LoRA (classifies which adapter to use)
# Adapter list is loaded from centralized config/adapters.yaml
def _build_router_system_prompt() -> str:
    """Build router system prompt with adapter list from central registry."""
    try:
        from sdk.adapters import AdapterRegistry
        registry = AdapterRegistry()
        adapters_section = registry.router_system_prompt()
    except ImportError:
        # Fallback for Modal remote execution where sdk may not be available
        # This should match config/adapters.yaml
        adapters_section = """Valid adapters:
- calendar: Calendar/scheduling screens
- claim-window: Insurance claim forms and windows
- provider-select: Provider/doctor selection screens
- chandra: OCR text extraction tasks
- desktop: Desktop/home screen interactions
- appointment: Appointment booking screens
- login-window: Login/authentication screens
- chart-screen: Patient chart/medical record screens"""

    return f"""You are a routing classifier for a computer use agent. Given a screenshot and user instruction, output ONLY the name of the appropriate adapter to handle this task.

{adapters_section}

Output ONLY the adapter name, nothing else."""


ROUTER_SYSTEM_PROMPT = _build_router_system_prompt()


def discover_deployed_loras() -> dict[str, str]:
    """Auto-discover deployed LoRA adapters from the inference volume.

    Scans /inference/loras/*/adapter for deployed adapters (via deploy.py).
    Each adapter directory contains adapter_config.json and deploy_metadata.json.

    Returns:
        Dict mapping adapter name (screen_type) to adapter path
    """
    loras_path = Path(INFERENCE_LORAS_PATH)
    if not loras_path.exists():
        print(f"WARNING: Inference loras path {INFERENCE_LORAS_PATH} does not exist")
        return {}

    discovered = {}

    for screen_type_dir in loras_path.iterdir():
        if not screen_type_dir.is_dir() or screen_type_dir.name.startswith("."):
            continue

        adapter_dir = screen_type_dir / "adapter"
        adapter_config = adapter_dir / "adapter_config.json"

        if adapter_config.exists():
            discovered[screen_type_dir.name] = str(adapter_dir)

            # Try to read deploy metadata for logging
            metadata_path = adapter_dir / "deploy_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                acc = metadata.get("eval_accuracy", "N/A")
                dataset = metadata.get("dataset_name", "N/A")
                print(f"  Discovered LoRA: {screen_type_dir.name} (acc: {acc:.1%}, dataset: {dataset})")
            else:
                print(f"  Discovered LoRA: {screen_type_dir.name} -> {adapter_dir}")

    return discovered


def format_ocr_tool_call(text: str) -> str:
    """Format OCR text as a tool_call response.

    Args:
        text: Raw text extracted by Chandra

    Returns:
        Formatted tool_call string for the ocr action
    """
    tool_call = {
        "name": "computer_use",
        "arguments": {
            "action": "ocr",
            "text": text,
        },
    }
    return f"Action: Return the extracted text.\n<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"


def run_chandra_ocr(image_path: str, device: str = "cuda") -> str:
    """Run Chandra OCR model on an image.

    Args:
        image_path: Path to the image file
        device: Device to run on

    Returns:
        Extracted text from the image
    """
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
    from PIL import Image

    print(f"Loading Chandra model: {CHANDRA_MODEL}")
    model = AutoModelForVision2Seq.from_pretrained(
        CHANDRA_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.processor = AutoProcessor.from_pretrained(CHANDRA_MODEL, trust_remote_code=True)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Create batch input
    batch = [
        BatchInputItem(
            image=image,
            prompt_type="ocr_layout"
        )
    ]

    # Run inference
    result = generate_hf(batch, model)[0]
    extracted_text = parse_markdown(result.raw)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return extracted_text


# =============================================================================
# WARM MODEL SERVER - keeps all models loaded for fast inference
# =============================================================================


@app.cls(
    image=image,
    gpu="H100",
    timeout=600,
    container_idle_timeout=300,  # Keep warm for 5 minutes
    volumes={
        "/inference": inference_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class MoEInferenceServer:
    """Mixture of Experts inference server with all models loaded and warm.

    Loads on startup:
    - Base Qwen3-VL-8B model
    - Routing LoRA adapter
    - All task LoRA adapters (calendar, claim-window, provider-select)
    - Chandra OCR model

    Inference is fast (~2-5 seconds) because models stay loaded.
    """

    @modal.enter()
    def load_models(self):
        """Load all models on container startup."""
        import time
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from transformers import AutoModelForVision2Seq
        from peft import PeftModel

        inference_volume.reload()

        self.device = torch.device("cuda")
        timings = {}
        total_start = time.time()

        print(f"\n{'='*60}")
        print("MoE Inference Server - Loading Models")
        print(f"{'='*60}")

        # Load processor
        t0 = time.time()
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        timings["processor"] = time.time() - t0
        print(f"  -> {timings['processor']:.2f}s")

        # Load base model
        t0 = time.time()
        print(f"Loading base model: {BASE_MODEL}...")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_model.eval()
        timings["base_model"] = time.time() - t0
        print(f"  -> {timings['base_model']:.2f}s")

        # Load routing LoRA
        t0 = time.time()
        routing_path = Path(DEFAULT_ROUTING_CHECKPOINT)
        print(f"Loading routing LoRA: {routing_path}...")
        self.router_model = PeftModel.from_pretrained(self.base_model, routing_path)
        self.router_model.eval()
        timings["routing_lora"] = time.time() - t0
        print(f"  -> {timings['routing_lora']:.2f}s")

        # Load task LoRAs onto a single base model using PEFT multi-adapter
        # This saves ~48GB of VRAM compared to loading separate base models
        self.task_model = None
        # Auto-discover deployed LoRAs from inference volume
        print("Discovering deployed LoRAs...")
        deployed_loras = discover_deployed_loras()

        if not deployed_loras:
            print("WARNING: No deployed LoRAs found in /inference/loras")

        self.loaded_adapters = []
        first_adapter = True
        for adapter_name, lora_path in deployed_loras.items():
            t0 = time.time()
            print(f"Loading task LoRA: {adapter_name} from {lora_path}...")
            if first_adapter:
                # First adapter - create PeftModel from base
                task_base = Qwen3VLForConditionalGeneration.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.task_model = PeftModel.from_pretrained(
                    task_base, lora_path, adapter_name=adapter_name
                )
                first_adapter = False
            else:
                # Subsequent adapters - load onto existing model
                self.task_model.load_adapter(lora_path, adapter_name=adapter_name)
            self.loaded_adapters.append(adapter_name)
            self.task_model.eval()
            timings[f"task_lora_{adapter_name}"] = time.time() - t0
            print(f"  -> {timings[f'task_lora_{adapter_name}']:.2f}s")

        # Load Chandra OCR model
        t0 = time.time()
        print(f"Loading Chandra OCR model: {CHANDRA_MODEL}...")
        self.chandra_model = AutoModelForVision2Seq.from_pretrained(
            CHANDRA_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
        self.chandra_processor = AutoProcessor.from_pretrained(CHANDRA_MODEL, trust_remote_code=True)
        self.chandra_model.eval()
        timings["chandra_ocr"] = time.time() - t0
        print(f"  -> {timings['chandra_ocr']:.2f}s")

        total_time = time.time() - total_start

        print(f"\n{'='*60}")
        print(f"MoE INFERENCE SERVER READY")
        print(f"{'='*60}")
        print(f"\nTIMING BREAKDOWN:")
        for name, t in timings.items():
            print(f"  {name:25s}: {t:6.2f}s")
        print(f"  {'TOTAL':25s}: {total_time:6.2f}s")

        print(f"\n{'='*60}")
        print("LOADED BASE MODELS:")
        print(f"{'='*60}")
        print(f"  [1] Qwen VL (routing + tasks): {BASE_MODEL}")
        print(f"  [2] Chandra OCR:               {CHANDRA_MODEL}")

        print(f"\n{'='*60}")
        print("ROUTING LORA:")
        print(f"{'='*60}")
        routing_exists = Path(DEFAULT_ROUTING_CHECKPOINT).exists()
        print(f"  Path:   {DEFAULT_ROUTING_CHECKPOINT}")
        print(f"  Status: {'LOADED' if routing_exists else 'NOT FOUND'}")

        print(f"\n{'='*60}")
        print(f"TASK LORAS ({len(self.loaded_adapters)} loaded):")
        print(f"{'='*60}")
        for i, adapter in enumerate(self.loaded_adapters, 1):
            print(f"  [{i}] {adapter}")
        if not self.loaded_adapters:
            print("  (none)")

        print(f"\n{'='*60}")
        print(f"Server ready for inference!")
        print(f"{'='*60}\n")

    def _run_chandra_ocr(self, image_path: str) -> str:
        """Run Chandra OCR using pre-loaded model."""
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
        result = generate_hf(batch, self.chandra_model)[0]
        return parse_markdown(result.raw)

    @modal.method()
    def infer(self, image_b64: str, prompt: str) -> dict[str, Any]:
        """Fast inference using pre-loaded MoE models.

        Args:
            image_b64: Base64 encoded image (data URL format)
            prompt: The task prompt

        Returns:
            Dict with task_output, routed_adapter, and metadata
        """
        import torch
        import base64
        import tempfile
        from qwen_vl_utils import process_vision_info

        print(f"\n{'='*60}")
        print("MoE Inference")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:100]}...")

        # Decode base64 image to temp file
        if image_b64.startswith("data:"):
            b64_data = image_b64.split(",", 1)[1]
        else:
            b64_data = image_b64
        image_bytes = base64.b64decode(b64_data)
        temp_image_path = Path(tempfile.mktemp(suffix=".jpg"))
        temp_image_path.write_bytes(image_bytes)

        try:
            # Step 1: Route using routing LoRA
            # NOTE: Routing model is trained WITH the router system prompt
            routing_messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                routing_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(routing_messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            print("Running routing inference...")
            with torch.no_grad():
                outputs = self.router_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            routed_adapter = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip().lower()

            print(f"Routed to: {routed_adapter}")

            # Step 2: Run appropriate task model
            if routed_adapter == "chandra":
                # OCR path
                print("Running Chandra OCR...")
                extracted_text = self._run_chandra_ocr(str(temp_image_path))
                task_output = format_ocr_tool_call(extracted_text)
            elif routed_adapter in self.loaded_adapters:
                # Task LoRA path - switch to the appropriate adapter
                print(f"Running {routed_adapter} task LoRA...")
                self.task_model.set_adapter(routed_adapter)

                # Build task messages with COMPUTER USE prompt (different from routing)
                task_messages = [
                    {"role": "system", "content": COMPUTER_USE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_image_path}"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                task_text = self.processor.apply_chat_template(
                    task_messages, tokenize=False, add_generation_prompt=True
                )
                task_image_inputs, task_video_inputs = process_vision_info(task_messages)

                task_inputs = self.processor(
                    text=[task_text],
                    images=task_image_inputs,
                    videos=task_video_inputs,
                    return_tensors="pt",
                    padding=True,
                )
                task_inputs = {k: v.to(self.device) for k, v in task_inputs.items()}
                task_input_len = task_inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.task_model.generate(
                        **task_inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )

                task_output = self.processor.decode(
                    outputs[0][task_input_len:], skip_special_tokens=True
                ).strip()
            else:
                # Unknown adapter - return error (no fallback)
                error_msg = f"Unknown adapter '{routed_adapter}'. Valid adapters: {self.loaded_adapters + ['chandra']}"
                print(f"ERROR: {error_msg}")
                return {
                    "task_output": None,
                    "routed_adapter": routed_adapter,
                    "model": BASE_MODEL,
                    "error": error_msg,
                }

            print(f"Output: {task_output[:200]}...")
            print(f"{'='*60}\n")

            return {
                "task_output": task_output,
                "routed_adapter": routed_adapter,
                "model": BASE_MODEL,
            }

        finally:
            temp_image_path.unlink(missing_ok=True)

    @modal.method()
    def direct_infer(self, image_b64: str, prompt: str) -> dict[str, Any]:
        """Direct inference using base model only (no routing, no LoRA).

        Args:
            image_b64: Base64 encoded image (data URL format)
            prompt: The task prompt

        Returns:
            Dict with task_output and metadata
        """
        import torch
        import base64
        import tempfile
        from qwen_vl_utils import process_vision_info

        print(f"\n{'='*60}")
        print("Direct Inference (base model only)")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:100]}...")

        # Decode base64 image to temp file
        if image_b64.startswith("data:"):
            b64_data = image_b64.split(",", 1)[1]
        else:
            b64_data = image_b64
        image_bytes = base64.b64decode(b64_data)
        temp_image_path = Path(tempfile.mktemp(suffix=".jpg"))
        temp_image_path.write_bytes(image_bytes)

        try:
            messages = [
                {"role": "system", "content": COMPUTER_USE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            print("Running inference...")
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            task_output = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()

            print(f"Output: {task_output}")
            print(f"{'='*60}\n")

            return {
                "task_output": task_output,
                "routed_adapter": "base",
                "model": BASE_MODEL,
            }

        finally:
            temp_image_path.unlink(missing_ok=True)


@app.local_entrypoint()
def main():
    """Test the MoE inference server."""
    print("MoE Inference Server")
    print("Use MoEInferenceServer.infer() or MoEInferenceServer.direct_infer() for inference.")
    print("\nTo deploy: modal deploy modal/stacked_inference.py")

