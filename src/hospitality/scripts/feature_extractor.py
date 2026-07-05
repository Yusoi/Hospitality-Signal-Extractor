import asyncio
from collections import Counter
from enum import StrEnum
import json
from typing import TypedDict
from uuid import uuid7

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from loguru import logger
import orjson
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy

from pydantic import BaseModel, Field
from enum import StrEnum


class PriceSensitivity(StrEnum):
    NONE = "none"
    MENTIONED = "mentioned"
    EXPLICIT_BUDGET = "explicit_budget"


class ExtractedFeatures(BaseModel):
    mentions_specific_dates: bool = Field(
        description="Guest mentioned concrete check-in/check-out dates"
    )
    mentions_room_type: bool = Field(
        description="Guest specified a room type or category"
    )
    num_distinct_questions: int = Field(
        description="Count of distinct discovery/comparison questions asked"
    )
    urgency_signals: bool = Field(
        description="Language suggesting time pressure or immediate need"
    )
    comparison_count: int = Field(
        description="Number of options/prices/dates being compared"
    )
    guest_persona_hint: str = Field(
        description="Inferred persona: family/business/solo/couple/unclear"
    )
    attractions_mentioned: bool = Field(
        description="Guest mentioned nearby attractions"
    )
    specific_location_mentioned: bool = Field(
        description="Guest mentions a specific location, city or spot"
    )
    reached_payment_step: bool = Field(
        description="Guest reached a hold, confirmation, or payment/checkout step (e.g. was asked for name, email, billing details, or to confirm a hold), regardless of whether payment was actually completed"
    )
    language: str = Field(description="Language the guest is speaking")
    price_sensitivity: PriceSensitivity = Field(
        description="Way the guest was referring to monetary values"
    )


async def main():

    agent = create_agent(
        "gpt-5.4-mini", response_format=ProviderStrategy(ExtractedFeatures)
    )

    EXTRACTION_PROMPT = """
    Analyze this hotel chatbot conversation and extract the following features.

    Conversation:
    {conversation_text}
    """

    sem = asyncio.Semaphore(2)

    async def call_llm(prompt: str) -> ExtractedFeatures:
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

    class Messages(TypedDict):
        role: str
        text: str

    class Conversation(TypedDict):
        messages: list[Messages]
        true_label: str
        session_id: str

    async def extract_features(
        data: Conversation,
    ) -> dict:
        extracted_features = await call_llm(
            EXTRACTION_PROMPT.format(conversation_text=orjson.dumps(data["messages"]))
        )

        result = data | {"features": extracted_features.model_dump()}

        return result

    with open("data/first_iteration/2_embedded_data.jsonl", "r") as f:
        conversations = [orjson.loads(l) for l in f.readlines()]

    tasks = [extract_features(c) for c in conversations]

    conversations_with_features = await asyncio.gather(*tasks)

    with open("data/first_iteration/3_featured_extracted_data.jsonl", "w") as f:
        for conv in conversations_with_features:
            f.write(json.dumps(conv) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
