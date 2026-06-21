"""Cross-venue market matching: cheap lexical pre-filter -> local-LLM confirmation."""

from bot.matching.embed import candidate_pairs, lexical_similarity
from bot.matching.llm_match import MatchVerdict, confirm_match

__all__ = ["MatchVerdict", "candidate_pairs", "confirm_match", "lexical_similarity"]
