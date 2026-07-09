from unittest.mock import patch, AsyncMock
import pytest
from chat_request import get_response


@patch('chat_request._invoke_agent', new_callable=AsyncMock)
@patch('chat_request.render_prompty')
@patch('chat_request.get_context')
@patch('chat_request.get_embedding')
def test_get_response_valid_question(mock_get_embedding, mock_get_context, mock_render_prompty, mock_invoke_agent):
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]
    mock_get_context.return_value = ["context1", "context2"]
    mock_render_prompty.return_value = ({}, {"system": "system prompt body"})
    mock_invoke_agent.return_value = "The moon's size is about 3,474 km in diameter."

    response = get_response("What is the size of the moon?", [])

    assert response == {
        "answer": "The moon's size is about 3,474 km in diameter.",
        "context": ["context1", "context2"]
    }

    mock_get_embedding.assert_called_once_with("What is the size of the moon?")
    mock_get_context.assert_called_once_with("What is the size of the moon?", [0.1, 0.2, 0.3])
    mock_invoke_agent.assert_called_once()


@patch('chat_request._invoke_agent', new_callable=AsyncMock)
@patch('chat_request.render_prompty')
@patch('chat_request.get_context')
@patch('chat_request.get_embedding')
def test_get_response_empty_question(mock_get_embedding, mock_get_context, mock_render_prompty, mock_invoke_agent):
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]
    mock_get_context.return_value = []
    mock_render_prompty.return_value = ({}, {"system": "system prompt body"})
    mock_invoke_agent.return_value = ""

    response = get_response("", [])

    assert response == {
        "answer": "",
        "context": []
    }

    mock_get_embedding.assert_called_once_with("")
    mock_get_context.assert_called_once_with("", [0.1, 0.2, 0.3])
    mock_invoke_agent.assert_called_once()
