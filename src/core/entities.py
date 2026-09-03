from dataclasses import dataclass

@dataclass
class Document:
    """
    Properties:
        page_content (str)
        metadata (dict)
        score (float)
    """
    page_content: str   # Text retrieved
    metadata: dict      # Article metadata
    score: float = 0.0  # Score