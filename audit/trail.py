import json
import dataclasses
from dataclasses import dataclass, field
from typing import List

@dataclass
class AuditEntry:
    step_id: str
    timestamp: str
    problem: str
    lenses_applied: List[str]
    reasoning_chain: List[str]
    epistemic_tags: List[str]
    confidence: float
    validation_result: str
    generated_design_titles: List[str] = field(default_factory=list)
    generated_artifact_names: List[str] = field(default_factory=list)
    artifact_workspace_path: str = ""
    execution_session_summaries: List[str] = field(default_factory=list)
    launch_session_summaries: List[str] = field(default_factory=list)
    execution_code: str = ""
    execution_prediction: str = ""
    execution_output: str = ""
    execution_delta: float = 0.0
    execution_residual: float = 0.0
    
class AuditTrail:
    def __init__(self, log_file: str = "audit_trail.json"):
        self.log_file = log_file
        self.entries: List[AuditEntry] = self._load()
        
    def _load(self) -> List[AuditEntry]:
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        entries: List[AuditEntry] = []
        if not isinstance(data, list):
            return entries

        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(AuditEntry(**item))
            except TypeError:
                continue
        return entries

    def log(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
        self.export_json(self.log_file)

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([dataclasses.asdict(e) for e in self.entries], f, indent=2)

    def get_full_trace(self) -> List[AuditEntry]:
        return self.entries
