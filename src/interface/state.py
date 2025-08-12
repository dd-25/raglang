from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional

class ChatState(TypedDict):
    query: str
    context: Optional[str]  # Changed to list of documents
    result: Optional[str]
    escalate: Optional[bool]