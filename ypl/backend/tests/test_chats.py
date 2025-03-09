import textwrap
from unittest.mock import Mock

from sqlalchemy.orm.state import InstanceState

from ypl.backend.llm.chat import ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE, RESPONSE_SEPARATOR
from ypl.backend.llm.context import _get_assistant_messages
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

    # When no response is selected, use the response from the model collecting the history.
    turn_messages_without_selected = [t for t in turn_messages if t.ui_status != MessageUIStatus.SELECTED]
    messages = _get_assistant_messages(turn_messages_without_selected, "model_3", use_all_models_in_chat_history=False)
    assert len(messages) == 1
    assert messages[0].content == "Response from assistant 3."

    # When no response is selected, and no response from the current model exists, use the first response.
    messages = _get_assistant_messages(turn_messages_without_selected, "model_X", use_all_models_in_chat_history=False)
    assert len(messages) == 1
    assert messages[0].content == "Response from assistant 1."

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
        """
    ).strip().replace("\n\n---\n\n", RESPONSE_SEPARATOR)  # noqa: F821

    assert messages[0].content == expected
