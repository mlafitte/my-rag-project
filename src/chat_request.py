from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import pathlib
import re
import yaml
from jinja2 import Template
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from agent_framework.openai import OpenAIChatClient

from ai_search import retrieve_documentation

PROMPTY_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "chat.prompty")

_ROLE_PATTERN = re.compile(r"^(system|user|assistant):\s*$", re.IGNORECASE | re.MULTILINE)


def get_embedding(question: str):
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
        azure_ad_token_provider=token_provider,
    )
    return client.embeddings.create(
        input=question,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", ""),
    ).data[0].embedding


def get_context(question, embedding):
    return retrieve_documentation(question=question, index_name="rag-index", embedding=embedding)


def render_prompty(path=PROMPTY_PATH, **template_vars):
    """Load a .prompty file and render its role sections (system:/user:/assistant:)
    as Jinja2 templates, returning (config, {role: rendered_text})."""
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    _, frontmatter, body = raw.split("---", 2)
    config = yaml.safe_load(frontmatter)
    body = body.strip()

    matches = list(_ROLE_PATTERN.finditer(body))
    sections = {}
    if matches:
        for i, m in enumerate(matches):
            role = m.group(1).lower()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            sections[role] = body[start:end].strip()
    else:
        sections["system"] = body

    rendered = {role: Template(text).render(**template_vars) for role, text in sections.items()}
    return config, rendered


async def _invoke_agent(instructions: str, question: str, max_tokens: int) -> str:
    # Responses API client - required for gpt-5-class reasoning deployments, which on
    # Azure OpenAI are not guaranteed to be reachable through the Chat Completions API.
    # api_version is omitted on purpose: this client targets {endpoint}/openai/v1/,
    # which rejects dated api-versions. AZURE_OPENAI_API_VERSION stays dated because
    # get_embedding() and the evaluators still call the older /deployments/ path.
    client = OpenAIChatClient(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        credential=DefaultAzureCredential(),
    )
    agent = client.as_agent(
        name="RagChatAgent",
        instructions=instructions,
        default_options={"max_tokens": max_tokens},
    )
    result = await agent.run(question)
    return str(result)


def get_response(question, chat_history):
    print("inputs:", question)
    embedding = get_embedding(question)
    context = get_context(question, embedding)
    print("context:", context)
    print("getting result...")

    _, sections = render_prompty(question=question, documents=context, history=chat_history)
    max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "512"))
    instructions = sections.get("system", "")
    user_message = sections.get("user", question)

    result = asyncio.run(_invoke_agent(instructions, user_message, max_tokens))

    print("result: ", result)

    return {"answer": result, "context": context}


if __name__ == "__main__":
    get_response("What is the size of the moon?", [])
