import asyncio
import json
from langchain_openai import OpenAIEmbeddings
from loguru import logger

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

sem = asyncio.Semaphore(10)


def user_turns_to_text(messages):
    return "\n".join([m["text"] for m in messages if m["role"] == "user"])


async def embed_conversation(record: dict) -> list[float] | None:
    async with sem:
        try:
            text = user_turns_to_text(record["messages"])
            return await embeddings_model.aembed_query(text)
        except Exception as e:
            logger.warning(f"Embedding failed for session {record['session_id']}: {e}")
            return None


async def main():
    with open("data/test/1_base_data.jsonl", "r") as f:
        conversations = [json.loads(l) for l in f.readlines()]

    tasks = [embed_conversation(c) for c in conversations]

    embeddings = await asyncio.gather(*tasks)

    with open("data/test/2_embedded_data.jsonl", "w") as f:
        for c, e in zip(conversations, embeddings):
            f.writelines(json.dumps(c | {"embeddings": e}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
