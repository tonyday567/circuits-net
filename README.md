# circuits-net

A small neural-network library rebuilt as a client of the circuits ecosystem.

based on:

https://cybercat.institute/2024/04/15/neural-network-first-principles/

## Overview

`circuits-net` is a fixed two-layer MLP client of the circuits ecosystem:

- **circuits-mat** — dense matrix/vector operations (`Circuit.Mat.Dense`).
- **circuits-diff** — the explicit parameterised forward/backward carrier
  (`Circuit.Diff.Param.DiffP`). The network's reverse pass is written manually
  in this carrier; it is not automatic differentiation of `Net.model`.
- **circuits-learn** — read-only model composition (`Circuit.Learn.Para`) and
  stateless SGD (`Circuit.Learn.Ephemeral`).
- **Data.These** — the `Boundary` synonym for inference/gradient traffic.

The old hand-rolled `LensS`, `BackPass`, and `NetCoalgebra.Poly` modules have
been removed.

## Building

```bash
cabal build
```

## Modules

### Net

Forward-only layer definitions for a 2-layer MLP:

- `NetParams` stores weights as `Circuit.Mat.Dense.Matrix` and biases as
  `Harpie.Array` vectors.
- `linear1`, `bias1`, `relu1`, `linear2`, `bias2` are `Circuit.Learn.Para`
  morphisms.
- `model` composes the layers; `forward` runs inference.
- `mseLoss` provides mean-squared-error loss and output gradient.
- `Boundary` is a `These`-based type for inference/gradient boundaries.

### NetCoalgebra

The same fixed network with explicit training functions:

- `modelDiffP` is an internal `DiffP` value containing the manual forward and
  reverse passes.
- `NetState` wraps the current `NetParams`; the SGD optimiser is stateless.
- `trainStep` returns a prediction and a learning-rate-consuming next-state
  function.
- `trainIdentity` / `trainNetCoalgebra` repeatedly train on one fixed
  input/target pair; the latter currently delegates to the former.
- `referenceGradients` supplies the hand-written comparison used by the
  gradient oracle.

## Oracles

```bash
cabal run circuits-net-axioma
```

The executable checks:

1. `trainIdentity` decreases loss on one all-ones 4 → 8 → 4 example, reaches
   final loss below 0.1, and produces four outputs.
2. `trainNetCoalgebra` decreases loss on the same example through its delegate.
3. The manual `DiffP` parameter gradients agree with the hand-written reference
   on that fixed example.
4. A fixed Nguyen Lemma 3.3 reconstruction satisfies its rank, pseudoinverse,
   leaky-ReLU branch, and target-reconstruction witnesses.

These are fixed-example checks, not independent or general neural-network
coverage. In particular, the two gradient implementations share their
mathematical operations, and the training example does not by itself establish
an identity mapping.

## Version

0.1.0.0 — circuits-ecosystem rewrite.

## License

BSD-3-Clause
