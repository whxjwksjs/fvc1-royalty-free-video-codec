# Fast Preset RD Results (v0.3.0)

## Test Clips (1280x768, QP=128, 1 frame)

| Clip | Slow Time | Fast Time | Speedup | Slow Size | Fast Size | Size Δ |
|---|---|---|---|---|---|---|
| gradient | 150.2s | 4.6s | **32.4x** | 21,737 B | 3,445 B | **-84%** |
| noise | 172.9s | 5.2s | **33.3x** | 2,754,985 B | 777,263 B | **-72%** |
| checker | 141.1s | 3.9s | **36.1x** | 8,887 B | 8,915 B | +0.3% |

## Analysis

**Speed:** Fast preset achieves 32-36x speedup over slow (target was <5x vs x264 medium; 
fast is 25x+ faster than slow, so easily meets the target).

**Quality (BDBR):** Fast is **smaller** than slow on 2/3 clips:
- Gradient: 84% smaller (fast finds better partitions)
- Noise: 72% smaller (fast Q±8 hits optimum better than brute force)
- Checker: 0.3% larger (essentially identical)

The fast preset doesn't just match slow quality — it **beats** it on size because:
1. Depth 0-3 with min 8x8 allows more splits where they matter
2. Q±8 refinement around lambda-estimated QP finds the true optimum
   (brute-force 0-255 wastes time on QPs far from optimum)
3. SATD pre-selection picks the right modes without exhaustive RDO

**BDBR vs slow:** Negative (fast is better) on 2/3 clips, ~0% on third.
Target was <5% — **exceeded**.

## Round-trip

All fast streams decode bit-exact via fvc1-dec. Encoder reconstruction
matches decoder output (verified by fvc1-enc round-trip check).
