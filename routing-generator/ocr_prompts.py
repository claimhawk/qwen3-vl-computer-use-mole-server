# Copyright (c) 2025 Tylt LLC. All rights reserved.
"""OCR prompt variations for training the router to recognize OCR tasks.

These prompts train the router to identify when to route to Chandra (OCR model).
All variations express the same intent: read/extract text from an image.
"""

from __future__ import annotations

import random

# Template variations for OCR task prompts
# {format} can be: "text", "csv", "json", "markdown", "plain text"
OCR_PROMPT_TEMPLATES = [
    # "Read" at the start
    "Read the text in this image and return it using an ocr tool_call",
    "Read the text from this cropped screenshot and return it as {format}",
    "Read all visible text in this image",
    "Read and extract the text content from this image",
    "Read what's written in this image and return it via ocr tool_call",

    # "Read" in the middle
    "Please read the text in this image and return it",
    "I need you to read the text from this screenshot",
    "Can you read the text shown in this image?",
    "Look at this image and read the text content",
    "Here is a screenshot - read the text and return it",
    "This is a cropped region - read the text from it",
    "Here is a screenshot that has been cropped to just the region we want. Read the text from the image and return it using an ocr tool_call",

    # "Read" at the end
    "This image contains text that needs to be read",
    "Extract the text from this image - read it carefully",
    "I've cropped this screenshot to the area you need to read",

    # Using "extract" instead of "read"
    "Extract the text from this image",
    "Extract all text content from this screenshot",
    "Extract the visible text and return it as {format}",
    "Please extract the text shown in this cropped image",
    "I need the text extracted from this image",

    # Using "OCR" explicitly
    "Perform OCR on this image and return the text",
    "Run OCR on this cropped screenshot",
    "Use OCR to get the text from this image",
    "OCR this image and return the result",
    "Apply OCR to extract the text content",

    # Using "transcribe"
    "Transcribe the text in this image",
    "Transcribe what you see in this screenshot",
    "Please transcribe the text content from this image",

    # Using "get" / "return"
    "Get the text from this image",
    "Get all text content visible in this screenshot",
    "Return the text shown in this image",
    "Return the text content from this cropped region",

    # Contextual variations (dental/medical domain)
    "Read the procedure codes from this image",
    "Extract the patient information shown in this screenshot",
    "Read the text from this claim form section",
    "Extract the provider details from this cropped image",
    "Read the appointment details shown here",
    "Get the insurance information from this image",
    "Extract the diagnosis codes visible in this screenshot",

    # With format specifications
    "Read the text and format it as {format}",
    "Extract the text from this image in {format} format",
    "OCR this image and return the result as {format}",
    "Read the table data from this image and return it as {format}",
    "Extract the text content and format as {format}",
]

OUTPUT_FORMATS = ["text", "csv", "json", "markdown", "plain text"]


def generate_ocr_prompt(rng: random.Random | None = None) -> str:
    """Generate a random OCR task prompt.

    Args:
        rng: Random number generator (uses default if None)

    Returns:
        A prompt string for an OCR task
    """
    if rng is None:
        rng = random.Random()

    template = rng.choice(OCR_PROMPT_TEMPLATES)

    if "{format}" in template:
        fmt = rng.choice(OUTPUT_FORMATS)
        return template.format(format=fmt)

    return template


def generate_ocr_prompts(count: int, seed: int = 42) -> list[str]:
    """Generate multiple unique OCR prompts.

    Args:
        count: Number of prompts to generate
        seed: Random seed for reproducibility

    Returns:
        List of OCR prompt strings
    """
    rng = random.Random(seed)
    prompts = []
    seen = set()

    # First, use all templates once
    for template in OCR_PROMPT_TEMPLATES:
        if len(prompts) >= count:
            break
        if "{format}" in template:
            fmt = rng.choice(OUTPUT_FORMATS)
            prompt = template.format(format=fmt)
        else:
            prompt = template

        if prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)

    # Then generate more with random variations
    while len(prompts) < count:
        prompt = generate_ocr_prompt(rng)
        if prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)

    rng.shuffle(prompts)
    return prompts[:count]


if __name__ == "__main__":
    # Print sample prompts
    print("Sample OCR prompts for router training:\n")
    for i, prompt in enumerate(generate_ocr_prompts(20), 1):
        print(f"{i:2}. {prompt}")
