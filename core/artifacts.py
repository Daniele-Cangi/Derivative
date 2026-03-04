from dataclasses import dataclass


@dataclass
class DesignArtifact:
    artifact_id: str
    artifact_type: str
    filename: str
    language: str
    content: str
    execution_note: str
