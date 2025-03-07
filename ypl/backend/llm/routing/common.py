from enum import Enum


class SelectIntent(str, Enum):
    NEW_CHAT = "new_chat"
    NEW_TURN = "new_turn"
    # TODO(bhanu): remove this after consolidating enum in YuppHead
    SHOW_ME_MORE = "show_me_more"
    SHOW_MORE_WITH_SAME_TURN = "show_more_with_same_turn"
    RETRY = "retry"
    NEW_STYLE = "new_style"
    TALK_TO_OTHER_MODELS = "talk_to_other_models"
