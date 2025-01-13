import textwrap
from unittest.mock import Mock

from sqlalchemy.orm.state import InstanceState

from ypl.backend.llm.chat import RESPONSE_SEPARATOR, _get_assistant_messages
from ypl.backend.prompts import ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE
from ypl.db.chats import ChatMessage, MessageType, MessageUIStatus


class MockLanguageModel:
    def __init__(self, internal_name: str) -> None:
        self.internal_name = internal_name
        mock_manager = Mock()
        mock_manager.__getitem__ = lambda _, key: Mock()
        mock_manager.configure_mock(key="assistant_language_model")
        self._sa_instance_state = InstanceState(self, mock_manager)


def test_get_assistant_messages() -> None:
    turn_messages = [
        ChatMessage(
            message_type=MessageType.USER_MESSAGE,
            content="User prompt",
        ),
        ChatMessage(
            message_type=MessageType.ASSISTANT_MESSAGE,
            content="Response from assistant 1.",
            assistant_language_model=MockLanguageModel(internal_name="model_1"),
        ),
        ChatMessage(
            message_type=MessageType.ASSISTANT_MESSAGE,
            content="Response from assistant 2 (selected).",
            assistant_language_model=MockLanguageModel(internal_name="model_2"),
            ui_status=MessageUIStatus.SELECTED,
        ),
        ChatMessage(
            message_type=MessageType.ASSISTANT_MESSAGE,
            content="Response from assistant 3.",
            assistant_language_model=MockLanguageModel(internal_name="model_3"),
        ),
        ChatMessage(
            message_type=MessageType.ASSISTANT_MESSAGE,
            content=None,
            assistant_language_model=MockLanguageModel(internal_name="model_4"),
        ),
        ChatMessage(
            message_type=MessageType.ASSISTANT_MESSAGE,
            content="",
            assistant_language_model=MockLanguageModel(internal_name="model_1"),
        ),
    ]
    model = "model_1"

    messages = _get_assistant_messages(turn_messages, model, use_all_models_in_chat_history=False)
    assert len(messages) == 1
    assert messages[0].content == "Response from assistant 2 (selected)."

    messages = _get_assistant_messages(turn_messages, model, use_all_models_in_chat_history=True)
    assert len(messages) == 1
    expected = ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE + textwrap.dedent(
        """
        This was your response:

        Response from assistant 1.

        ---

        A response from another assistant:

        Response from assistant 2 (selected).

        (This response was preferred by the user)

        ---

        A response from another assistant:

        Response from assistant 3.

        ---

        (Your response was empty)
        """
    ).strip().replace("\n\n---\n\n", RESPONSE_SEPARATOR)

    assert messages[0].content == expected
