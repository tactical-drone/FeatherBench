"""
agent_parts // parts written by AI agents, kept apart from V's my_parts.py.

One module per idea, registering into the nncore registries on import:

    # agent_parts/snake2.py
    import torch, torch.nn as nn
    from nncore import ACTIVATIONS

    class Snake2(nn.Module):
        def forward(self, x):
            return x + torch.sin(2 * x) ** 2 / 2

    ACTIVATIONS.register("snake2", Snake2, doc="x + sin^2(2x)/2")

Load it next to my_parts (the console fingerprints both, so cached results stay honest):

    python -m rsi check  --parts my_parts --parts agent_parts.snake2 --activation snake2
    python -m rsi evolve --parts my_parts --parts agent_parts.snake2 ...

Nothing is registered by this package itself. Part of nn-playground. AGPL-3.0;
for other licensing see COMMERCIAL.md.
"""
