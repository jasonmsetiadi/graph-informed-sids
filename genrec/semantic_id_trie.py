from collections import defaultdict


class SemanticIDTrie:
    """Prefix constraints and terminal states for variable-length catalog SIDs."""

    def __init__(self, sequences):
        self.children = defaultdict(set)
        self.terminals = set()
        for sequence in sequences:
            sequence = tuple(map(int, sequence))
            if not sequence:
                raise ValueError("Semantic IDs must not be empty")
            for index, token in enumerate(sequence):
                self.children[sequence[:index]].add(token)
            self.terminals.add(sequence)

    def allowed_tokens(self, prefix, eos_token):
        prefix = tuple(map(int, prefix))
        if prefix in self.terminals:
            return [int(eos_token)]
        children = self.children.get(prefix)
        return sorted(children) if children else [int(eos_token)]
