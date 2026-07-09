import asyncio
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chat_request import render_prompty, _invoke_agent

EVAL_PROMPTY_PATH = os.path.join(os.path.dirname(__file__), "prompty-answer-score-eval.prompty")


def run_prompty(path, max_tokens=200, **template_vars):
    _, sections = render_prompty(path, history=[], **template_vars)
    instructions = sections.get("system", "")
    user_message = sections.get("user", template_vars.get("question", ""))
    return asyncio.run(_invoke_agent(instructions, user_message, max_tokens))


def main():
    data_path = os.path.join(os.path.dirname(__file__), "test-dataset.jsonl")
    with open(data_path, "r") as f:
        rows = [json.loads(line) for line in f]

    base_results = []
    for row in rows:
        answer = run_prompty(
            os.path.join(os.path.dirname(__file__), "..", "src", "chat.prompty"),
            question=row["question"],
            documents=row["documents"],
        )
        base_results.append({**row, "answer": answer})

    print(pd.DataFrame(base_results).head(10))

    eval_results = []
    for row in base_results:
        score_output = run_prompty(
            EVAL_PROMPTY_PATH,
            question=row["question"],
            answer=row["answer"],
            ground_truth=row["ground_truth"],
        )
        try:
            parsed = json.loads(score_output)
        except json.JSONDecodeError:
            parsed = {"score": None, "explanation": score_output}
        eval_results.append({
            "question": row["question"],
            "answer": row["answer"],
            "ground_truth": row["ground_truth"],
            "score": parsed.get("score"),
            "explanation": parsed.get("explanation"),
        })

    df = pd.DataFrame(eval_results)
    print(df.head(10))
    df.to_excel("prompty-answer-score-eval.xlsx", index=False)


if __name__ == '__main__':
    main()
