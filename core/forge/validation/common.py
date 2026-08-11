from typing import List


class ValidationLayerBase:
    @staticmethod
    def _append_unique(collection: List[str], value: str) -> None:
        if value not in collection:
            collection.append(value)
