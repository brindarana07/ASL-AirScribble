from difflib import get_close_matches
from typing import List, Sequence


class TextAssist:
    def __init__(self, vocabulary: Sequence[str]):
        self.vocabulary = sorted({word.upper() for word in vocabulary})

    def suggestions(self, partial_word: str, limit: int = 3) -> List[str]:
        word = partial_word.upper().strip()
        if not word:
            return []

        prefix_matches = [candidate for candidate in self.vocabulary if candidate.startswith(word)]
        fuzzy_matches = get_close_matches(word, self.vocabulary, n=limit, cutoff=0.55)

        suggestions = []
        for candidate in prefix_matches + fuzzy_matches:
            if candidate not in suggestions:
                suggestions.append(candidate)
            if len(suggestions) >= limit:
                break
        return suggestions

    def best_correction(self, word: str) -> str:
        matches = self.suggestions(word, limit=1)
        return matches[0] if matches else word.upper()
