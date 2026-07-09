import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import evaluate, SexualEvaluator, ViolenceEvaluator, SelfHarmEvaluator, HateUnfairnessEvaluator
from azure.ai.evaluation.simulator import AdversarialScenario, AdversarialSimulator, DirectAttackSimulator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chat_request import get_response

async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
) -> dict:
    query = messages["messages"][0]["content"]

    # Add file contents for summarization or re-write
    if 'file_content' in messages["template_parameters"]:
        query += messages["template_parameters"]['file_content']

    response = get_response(query, [])['answer']

    # Format responses in OpenAI message protocol
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {},
    }

    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state
    }


def conversations_to_jsonl(conversations, path):
    """AdversarialSimulator/DirectAttackSimulator return raw conversation dicts
    (each with a 'messages' list). Evaluators expect a 'conversation' column."""
    with open(path, 'w') as f:
        for item in conversations:
            f.write(json.dumps({"conversation": {"messages": item["messages"]}}) + '\n')
    return path


async def main():
    # Read environment variables
    azure_location = os.getenv("AZURE_LOCATION")
    azure_subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    azure_resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    azure_project_name = os.getenv("AZUREAI_PROJECT_NAME")
    prefix = os.getenv("PREFIX", datetime.now().strftime("%y%m%d%H%M%S"))[:14]

    print("AZURE_LOCATION=", azure_location)
    print("AZURE_SUBSCRIPTION_ID=", azure_subscription_id)
    print("AZURE_RESOURCE_GROUP=", azure_resource_group)
    print("AZUREAI_PROJECT_NAME=", azure_project_name)
    print("PREFIX=", prefix)

    valid_locations = ["eastus2", "francecentral", "uksouth", "swedencentral"]

    if azure_location not in valid_locations:
        print(f"Invalid AZURE_LOCATION: {azure_location}. Must be one of {valid_locations}.")
    else:
        azure_ai_project = {
            "subscription_id": azure_subscription_id,
            "resource_group_name": azure_resource_group,
            "project_name": azure_project_name,
        }
        credential = DefaultAzureCredential()

        sexual_evaluator = SexualEvaluator(credential, azure_ai_project)
        self_harm_evaluator = SelfHarmEvaluator(credential, azure_ai_project)
        hate_unfairness_evaluator = HateUnfairnessEvaluator(credential, azure_ai_project)
        violence_evaluator = ViolenceEvaluator(credential, azure_ai_project)

        evaluators = {
            "sexual": sexual_evaluator,
            "self_harm": self_harm_evaluator,
            "hate_unfairness": hate_unfairness_evaluator,
            "violence": violence_evaluator
        }

        scenario = AdversarialScenario.ADVERSARIAL_QA

        # Plain adversarial conversations (no jailbreak prompt injection)
        simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)
        adversarial_conversations = await simulator(
            scenario=scenario,
            target=callback,
            max_conversation_turns=1,
            max_simulation_results=10,
        )
        print(f"Adversarial conversation results: {adversarial_conversations}.")
        adversarial_data = conversations_to_jsonl(adversarial_conversations, "adversarial_conversations.jsonl")

        try:
            adversarial_eval_result = evaluate(
                evaluation_name=f"{prefix} Adversarial Tests",
                data=adversarial_data,
                evaluators=evaluators,
                azure_ai_project=azure_ai_project,
                output_path="./adversarial_test.json"
            )
        except Exception as e:
            print(f"An error occurred during evaluation: {e}\n Retrying without reporting results in Azure AI Project.")
            adversarial_eval_result = evaluate(
                evaluation_name=f"{prefix} Adversarial Tests",
                data=adversarial_data,
                evaluators=evaluators,
                output_path="./adversarial_test.json"
            )

        # Jailbreak (UPIA prompt-injection) conversations, via the dedicated simulator
        direct_attack_simulator = DirectAttackSimulator(azure_ai_project=azure_ai_project, credential=credential)
        direct_attack_result = await direct_attack_simulator(
            scenario=scenario,
            target=callback,
            max_conversation_turns=1,
            max_simulation_results=10,
        )
        jailbreak_conversations = direct_attack_result["jailbreak"]
        print(f"Adversarial conversation w/ jailbreak results: {jailbreak_conversations}.")
        jailbreak_data = conversations_to_jsonl(jailbreak_conversations, "adversarial_conversations_jailbreak.jsonl")

        try:
            adversarial_eval_w_jailbreak_result = evaluate(
                evaluation_name=f"{prefix} Adversarial Tests w/ Jailbreak",
                data=jailbreak_data,
                evaluators=evaluators,
                azure_ai_project=azure_ai_project,
                output_path="./adversarial_test_w_jailbreak.json"
            )
        except Exception as e:
            print(f"An error occurred during evaluation: {e}\n Retrying without reporting results in Azure AI Project.")
            adversarial_eval_w_jailbreak_result = evaluate(
                evaluation_name=f"{prefix} Adversarial Tests w/ Jailbreak",
                data=jailbreak_data,
                evaluators=evaluators,
                output_path="./adversarial_test_w_jailbreak.json"
            )

if __name__ == '__main__':
    asyncio.run(main())
