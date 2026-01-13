from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Bubble:
    id: str
    bbox: List[int]
    original_text: str
    translation: str
    final_text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PagePayload:
    page: str
    target_language: str
    style: str
    bubbles: List[Bubble] = field(default_factory=list)
