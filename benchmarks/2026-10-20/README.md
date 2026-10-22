# Merlin 3.0.0 bench cards — 2026-10-20

One card per checkpoint, measured on the reference host (4× RTX 3090, 220 W, PCIe P2P) with the release wheel installed
the way a user installs it, except the sections each card marks "a pre-release engine build, 2026-10-17" (part or all of section 6 on every card but Ornith 1.5, whose card states its own build). Every card has a single-stream decode ladder from 4K to 261K prompt tokens with thinking on
and off, an exact 262K row (261,888 prompt tokens + 128 out), cold prefill, sessions held at once at the design point,
8 MP images, and the release card routine (real-text decode, agents at once, GSM8K-200, needle panel); it compares
with the previous release of that checkpoint where one was published.

| card | checkpoint | GPUs | compared with |
|---|---|---|---|
| [Flash-Next](Flash-Next-BENCH-CARD.md) | Qwen3.8-Flash-Next | 4 | v2.5.2 |
| [Swift1.5](Swift1.5-BENCH-CARD.md) | Swift1.5 | 4 | v2.5.2 (prefill) |
| [Swift1.5-Huihui](Swift1.5-Huihui-BENCH-CARD.md) | Swift1.5-Huihui | 4 | v2.5.2 (prefill) |
| [Qwen3.8-27B](Qwen3.8-27B-BENCH-CARD.md) | Qwen3.8-27B | 1, 2, 4 | v1.3.0 |
| [ThinkingCap](ThinkingCap-BENCH-CARD.md) | ThinkingCap | 2, 4 | ThinkingCap 2026-10-12 |
| [Nemotron 3.5 Super VL](Super-VL-BENCH-CARD.md) | Nemotron 3.5 Super VL | 4 | new |
| [Ornith 1.5](Ornith-1.5-BENCH-CARD.md) | Ornith 1.5 (measured 2026-10-21) | 4 | new |

Ornith 1.5 was added on 2026-10-21. Its card has no real-text decode or colour-answer image cases; it has the 8 MP headroom
with a 262K prompt, an agent-trace replay and a without-P2P comparison instead.
