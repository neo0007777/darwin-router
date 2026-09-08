# Darwin Router — visual transcript

The 60-second overview uses on-screen text and an original instrumental score.
There is no spoken narration. The diagrams explain the implementation; the
dashboard excerpt is from the original hackathon recording.

| Time | What the video shows |
|---|---|
| 00:00–00:06 | **Darwin Router. A router that learns while it routes.** A glass junction splits a stream into two routes and a feedback loop. This is conceptual artwork. The hero states the repository's reported hackathon results: 75% to 85% holdout routing accuracy and 50% lower estimated cost per request. |
| 00:06–00:15 | **Every request finds its lane.** A text prompt is embedded with Qwen3-Embedding-4B on SIE. Cosine kNN over the exemplar index uses seven nearest neighbours to choose the simple lane, Qwen3.5-4B on SIE, or the reasoning lane, qwen-max on Alibaba Cloud. Image, vision and video requests use separate pre-route rules. |
| 00:15–00:24 | **Uncertainty earns a second opinion.** A kNN margin below 0.7, or a 10% random sample, triggers asynchronous review. The alternative text route produces a real response. A judge compares both responses blind, with model identities hidden and A/B order randomized. Correctness and completeness are assessed. The cheaper answer is preferred when adequate; a clear verdict is required for writeback. |
| 00:24–00:33 | **Learning is an index write.** An illustrative record shows a judged query, route label, judge provenance and embedding. A new point enters router memory. Guardrails include near-duplicate deduplication, per-route insertion caps, protected bootstrap examples and verdict-flip hysteresis. There is no model-weight update or training job. |
| 00:33–00:42 | **One day. A better routing policy.** The README reports 75% to 85% holdout routing accuracy, estimated average cost per request falling from $0.0022 to $0.0011, and index growth from 57 to 91 exemplars between 09:54 and 17:45 on 14 Aug 2026. The 40-query evaluation traffic never enters the index. These are historical, repository-reported results; costs use hardcoded token-price estimates. |
| 00:42–00:52 | **Watch the learning happen.** Ten seconds from the original narrated dashboard tour, beginning at its 00:35 mark, show the request feed and judge events. The original narration is omitted in this cut. The footage is labelled as an intermediate snapshot recorded on 14 Aug 2026, not a new evaluation or an end-of-day measurement. |
| 00:52–01:00 | **Route. Judge. Improve.** Training-free. Inspectable. Built in one day. First prize at the Superlinked × Qwen Hackathon. Project: github.com/hashkanna/darwin-router. |

## Evidence

- Reported performance and award: [project README](../README.md).
- Seven-neighbour cosine routing and persisted exemplars: [router/index.py](../router/index.py).
- Exploration gate, blind comparison and insertion guards: [router/judge.py](../router/judge.py).
- Decision-only evaluation: [router/proxy.py](../router/proxy.py) and [router/replay.py](../router/replay.py).
- Original footage: [assets/demo_video.mp4](demo_video.mp4).

The illustrative diagrams and JSON record are not inference logs. No new live
model requests or benchmark runs were used to make this overview.
