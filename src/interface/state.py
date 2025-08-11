from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional

class ChatState(TypedDict):
    query: str
    result: Optional[str]
    escalate: Optional[bool]