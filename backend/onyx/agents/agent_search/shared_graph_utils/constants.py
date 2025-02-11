from enum import Enum
from typing import Any

AGENT_LLM_TIMEOUT_MESSAGE = "The agent timed out. Please try again."
AGENT_LLM_ERROR_MESSAGE = "The agent encountered an error. Please try again."
AGENT_LLM_RATELIMIT_MESSAGE = (
    "The agent encountered a rate limit error. Please try again."
)
LLM_ANSWER_ERROR_MESSAGE = "The question was not answered due to an LLM error."

AGENT_POSITIVE_VALUE_STR = "yes"
AGENT_NEGATIVE_VALUE_STR = "no"

BaseMessage_Content = str | list[str | dict[str, Any]]


class AgentLLMErrorType(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    GENERAL_ERROR = "general_error"
