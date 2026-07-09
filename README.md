# RAG with Azure AI and Microsoft Agent Framework

## Migration off Prompt Flow

Prompt Flow is being retired by Microsoft (feature-frozen April 2026, fully retired April
2027) and its Chat Completions-era LLM connector does not speak the request format required
by GPT-5-class deployments. This branch replaces the Prompt Flow flow under `src/` with a
plain Python implementation on [Microsoft Agent Framework](https://github.com/microsoft/agent-framework),
still driven by the same `chat.prompty` template. See `src/chat_request.py`, `src/app.py`,
and the root `Dockerfile` for the new implementation, and `infra/ai.yaml` /
`infra/main.bicep` for the (placeholder) GPT-5-class model deployment - update those to a
model/version your subscription can actually deploy before running `azd provision`.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

---

Let me know if any further adjustments are needed!