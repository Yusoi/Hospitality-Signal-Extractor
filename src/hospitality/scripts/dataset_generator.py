import asyncio
from collections import Counter
from enum import StrEnum
import json
import random
from typing import Optional
from uuid import uuid7

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from loguru import logger
import orjson
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy


class Roles(StrEnum):
    USER = "user"
    BOT = "bot"


class ValidationLabel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ABANDONED = "abandoned"


class MessageFormat(BaseModel):
    role: Roles
    text: str
    booking: Optional[str] = None


class OutputFormat(BaseModel):
    messages: list[MessageFormat]
    true_label: ValidationLabel


async def main():

    model = init_chat_model("gpt-5.4-nano", temperature=1.1, top_p=0.95)
    agent = create_agent(
        model,
        response_format=ProviderStrategy(OutputFormat),
    )

    PROMPT_TEMPLATE = """
    Generate a realistic hotel chatbot conversation in JSON format.
    Target booking intent level: {target_label}

    Definitions:
    - low: Broad discovery questions (e.g., "what's nearby?", "any weekend deals?", "where should I have dinner?", "what are some nearby attractions?")
    - medium: Comparing concrete options (dates, prices, room types, cancellation terms, spa pricing, facilities)
    - high: Booking-ready commands or details (e.g., "book 2 nights Oct 12-14"). 
    - abandoned: Reached payment/confirmation or asked to hold a room but didn't finish. There must be information on the bot's side that a payment request has been sent, or a booking is waiting for user confirmation. Booking confirmation is only made if there is a confirmation message in the following format: {{"role": "bot", "text": "Booking confirmed", "booking": "confirmed"}} (The text component of the confirmation message may not be exactly the same as the one provided).

    Vary:
    - hotel type (budget/luxury/business/resort/boutique)
    - language (English or Portuguese)
    - conversation length (3-15 turns) 
    - guest persona (family, business, solo, couple, friends, etc.)
    
    Conversation details:
    - Hotel Type: {hotel_type}
    - Persona: {persona}
    - Location: {location}
    - Language: {language}
    - User tone: {tone}
    - Typos: {typos}

    Output format: {{"messages": [{{"role": "user"/"bot", "text": "..."}}], "true_label": "{target_label}"}}
    """

    def generate_prompt(label: ValidationLabel):

        # Include diversity in prompt
        typos = random.choices(
            [
                "include typos in the user prompts",
                "do not include typos in any prompts",
            ],
            [0.3, 0.7],
        )

        hotel_types = [
            "budget hostel",
            "boutique hotel",
            "luxury resort",
            "business hotel",
            "airport hotel",
            "beachfront inn",
        ]
        personas = [
            "solo business traveler",
            "family with young kids",
            "couple on a getaway",
            "group of friends",
            "elderly couple",
            "digital nomad",
        ]
        locations = [
            "Lisbon",
            "Porto",
            "Rio de Janeiro",
            "Madrid",
            "New York",
            "Bangkok",
        ]
        languages = ["English", "Portuguese", "Spanish"]
        tones = [
            "polite and formal",
            "casual and brief",
            "chatty and detailed",
            "impatient",
            "hesitant and unsure",
        ]

        return PROMPT_TEMPLATE.format(
            target_label=label.value,
            hotel_type=random.choice(hotel_types),
            persona=random.choice(personas),
            location=random.choice(locations),
            language=random.choice(languages),
            tone=random.choice(tones),
            typos=random.choice(typos),
        )

    labels = list(ValidationLabel)
    synthetic_data = []

    sem = asyncio.Semaphore(8)

    async def call_llm(prompt: str) -> OutputFormat:
        async with sem:
            try:
                logger.info("Calling agent ...")
                result = await agent.ainvoke({
                    "messages": [{"role": "user", "content": prompt}]
                })
                logger.info("Called agent Successfully")
                return result["structured_response"]
            except Exception as e:
                logger.warning(f"Skipping failed generation: {e}")
                return None

    tasks = [
        call_llm(generate_prompt(label))
        for label in labels
        for _ in range(
            125
        )  # 125 per class = 500 total (Guarantees balanced class labels)
    ]

    synthetic_data = await asyncio.gather(*tasks)

    # Logging of missing operations
    labels_per_request = [label.value for label in labels for _ in range(125)]
    failed = Counter(
        lbl for lbl, r in zip(labels_per_request, synthetic_data) if r is None
    )
    if failed:
        logger.warning(f"Failed generations per label: {dict(failed)}")

    synthetic_data = [
        sd.model_dump() | {"session_id": str(uuid7())}
        for sd in synthetic_data
        if sd is not None
    ]

    with open("data/test/1_base_data.jsonl", "w") as f:
        for conv in synthetic_data:
            f.write(json.dumps(conv) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
