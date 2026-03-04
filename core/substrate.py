import concurrent.futures
from typing import List, Optional

from lenses.base import BaseLens, CognitiveLens
from lenses.causal import CausalLens
from lenses.symbolic import SymbolicLens
from lenses.topological import TopologicalLens
from lenses.probabilistic import ProbabilisticLens
from lenses.quantum import QuantumLens
from lenses.formal import FormalLens
from lenses.physical import PhysicalLens

class CognitiveSubstrate:
    def __init__(self, api_key: Optional[str] = None, execution_mode: Optional[str] = None):
        # Load all lenses at startup — not lazily
        self.api_key = api_key
        self.execution_mode = execution_mode
        self.lenses = self._load_all_lenses()
        self.last_errors: List[str] = []
    
    def decompose(self, problem: str) -> List[CognitiveLens]:
        """
        Apply ALL applicable lenses to the problem simultaneously.
        Returns minimum 2 framings, ideally 4+.
        Never returns a single framing — that is a substrate failure.
        """
        framings: List[CognitiveLens] = []
        successful_lenses: set[str] = set()
        self.last_errors = []
        
        # Parallel execution of lenses
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.lenses))) as executor:
            future_to_lens = {
                executor.submit(self._apply_lens, lens, problem): lens 
                for lens in self.lenses
            }
            
            for future in concurrent.futures.as_completed(future_to_lens):
                lens = future_to_lens[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.last_errors.append(f"{lens.lens_name}: {exc}")
                    result = None
                if result:
                    framings.append(result)
                    successful_lenses.add(result.lens_name)

        if len(framings) < 2:
            for lens in self.lenses:
                if lens.lens_name in successful_lenses:
                    continue
                try:
                    fallback = lens.frame(problem)
                except Exception as exc:
                    self.last_errors.append(f"{lens.lens_name}: {exc}")
                    continue
                framings.append(fallback)
                successful_lenses.add(lens.lens_name)
                if len(framings) >= 2:
                    break

        if len(framings) < 2:
            if self.last_errors:
                raise RuntimeError(
                    f"Substrate failure: only {len(framings)} framings returned. Errors: {'; '.join(self.last_errors)}"
                )
            raise RuntimeError(f"Substrate failure: only {len(framings)} framings returned, minimum 2 required.")

        framings.sort(key=lambda framing: framing.confidence, reverse=True)
            
        return framings

    def _apply_lens(self, lens: BaseLens, problem: str) -> Optional[CognitiveLens]:
        if lens.is_applicable(problem):
            return lens.frame(problem)
        return None
    
    def _load_all_lenses(self) -> List[BaseLens]:
        # Instantiate every lens from lenses/
        return [
            CausalLens(api_key=self.api_key, execution_mode=self.execution_mode),
            SymbolicLens(api_key=self.api_key, execution_mode=self.execution_mode),
            TopologicalLens(api_key=self.api_key, execution_mode=self.execution_mode),
            ProbabilisticLens(api_key=self.api_key, execution_mode=self.execution_mode),
            QuantumLens(api_key=self.api_key, execution_mode=self.execution_mode),
            FormalLens(api_key=self.api_key, execution_mode=self.execution_mode),
            PhysicalLens(api_key=self.api_key, execution_mode=self.execution_mode)
        ]
