"""Read-only inputs for postprocessors that need exact serving-time rendering.

A hook opts in with an explicit keyword parameter named ``sample_context``.
Legacy two-argument hooks keep their existing call contract. Records belong to
the collection snapshot; hooks must not mutate them or the shared tokenizer.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import SessionStateV2
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


@dataclass(frozen=True)
class SamplePostprocessContext:
    tito_tokenizer: TITOTokenizer
    records_by_leaf: Mapping[int, tuple[SessionRecord, ...]]

    @classmethod
    def from_snapshot(cls, state: SessionStateV2, tito_tokenizer: TITOTokenizer):
        return cls(
            tito_tokenizer=tito_tokenizer,
            records_by_leaf=MappingProxyType(
                {leaf.seq: tuple(node.record for node in leaf.path_nodes()) for leaf in state.tree.leaves()}
            ),
        )
