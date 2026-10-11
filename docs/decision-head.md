# Decision head

`decision_head_core` is a reusable C11 CPU classifier for feature vectors. It
supports a linear classifier or one tanh hidden layer, an ensemble averaged in
logit space, input standardization/clipping, and temperature softmax. Class
labels, dimensions and ensemble size come from the model. There are no built-in
turn-taking actions, confidence thresholds, or speech controls.

The core uses FP32 weights and arithmetic, with four dot-product accumulators.
It needs neither CUDA nor BLAS, allocates nothing during prediction, and uses
no worker threads. Each concurrent caller owns a workspace; the immutable head
can be shared. A host supplies features and decides what the returned label
means. The existing engine's generation path is unchanged by building this tool.

## Build and use

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target decision-head decision_head_test

# Offline conversion only: Python + NumPy, no C runtime dependency.
python tools/export_decision_head.py /path/to/npz-package decision-head-e4b.gguf
build/decision-head inspect decision-head-e4b.gguf
build/decision-head predict decision-head-e4b.gguf hidden-vectors.f32
```

On multi-configuration Windows builds the executable is
`build/Release/decision-head.exe`. Prediction reads consecutive little-endian
FP32 vectors (or `-` for stdin) and emits one JSON line per vector: class index,
label, logits, probabilities and measured inference time. `--repeat 100` performs
one excluded warmup and 100 measured repetitions per vector; the reported mean
excludes loading and I/O. It is neither prefill latency nor speech latency.
The offline CLI checks vector dimensions, not their provenance.

The converter accepts the existing `head.npz` + `contract.json` package. It
checks the source hash, shape, precision, class order and feature contract, and
refuses to overwrite its destination. `linear` and `mlp64` source packages map
to the generic linear/MLP schema. The originals remain the training artifacts.

## Embedding

The API is in [`include/decision-head.h`](../include/decision-head.h).
Link `decision_head_gguf` for loading a sidecar GGUF, or only
`decision_head_core` and call `decision_head_create()` with tensors loaded by
the host. The latter is the integration seam for a host such as llama.cpp:
it avoids combining two different GGUF reader implementations.

1. Load/create the head once. The constructor owns copies of its arrays and
   metadata; temporary loader storage can be released.
2. Call `decision_head_check_features()` with the **actual host's** model hash,
   dimension, input format, layer, capture boundary and answer-channel state.
   A matching width alone is insufficient. Another backend's hidden-state
   compatibility needs separate validation even if the metadata matches.
3. Allocate one workspace for each concurrent caller.
4. Call `decision_head_predict()` with that observation's immutable feature
   vector, separate output buffers, and an output capacity of `output_dim`.
5. Only consume outputs on success. Failures return `-1`, invalidate the chosen
   index with `SIZE_MAX`, and leave output buffers untouched.
6. Free workspaces before the head.

The host remains responsible for snapshot lifetime, connection/observation
association, scheduling, and applying a result. The library does not introduce
a second model prefill. Scores are model outputs; calibration and policy
priority are separate concerns. The current schema represents one categorical
output per package, not a multitask graph or a reasoning model.

## GGUF schema v1

The format is a little-endian GGUF v3 sidecar with FP32 tensors. Its
`general.architecture` is `decision_head`. Metadata under `decision_head.*`:

| Key | Type | Meaning |
| --- | --- | --- |
| `version` | uint32 | `1` |
| `architecture` | string | `linear` or `mlp` |
| `input_dim`, `hidden_dim`, `output_dim`, `ensemble_size` | uint32 | Shape; linear uses zero hidden width |
| `activation` | string | `identity` for linear, `tanh` for MLP |
| `normalization` | string | `standardize_clip` |
| `ensemble_reduction` | string | `mean_logits` |
| `output_transform` | string | `temperature_softmax` |
| `clip` | float32 | Symmetric clipping bound after normalization |
| `temperature` | float64 | Stored without source precision loss; used as FP32 by inference |
| `labels` | string array | Ordered, unique class names |
| `task` | string | Application-defined task identifier |
| `feature.model_sha256`, `feature.input_format`, `feature.capture_boundary` | string | Required feature provenance |
| `feature.layer` | int32 | Captured layer; `-1` can describe non-layer features |
| `feature.answer_channel_seed` | bool | Prompt-boundary state |
| `source_contract` | string | Original training-contract JSON, kept as provenance |

Tensor names: `input.mean`, `input.scale`, and for each zero-based ensemble
member `member.N.output.weight` and `.output.bias`. MLPs also contain
`member.N.hidden.weight` and `.hidden.bias`. Matrices have row-major logical
shape `[output, input]`; GGUF lists the input/fastest dimension first. The
loader validates exact tensor shapes, types, complete byte spans, unique keys,
class labels, finite values, and positive scales/temperature. Unknown schema
versions, activations, or reductions are rejected rather than approximated.

Historical fields inside `source_contract` such as `deployment_status` describe
the original training/export checkpoint. They do not authorize live deployment.

## Validation

`decision_head_test` uses analytic linear/MLP fixtures and tests ownership,
clipping, ensemble averaging, class ordering, feature mismatch and invalid input.
When the configured Python has NumPy, CTest also runs `decision_head_gguf_test`:
deterministic export, numerical comparison to NumPy, generic/escaped labels,
truncated/malformed GGUF rejection and source hash validation. The [published E4B/12B weights](https://huggingface.co/cortexist/gemma4-decision-head)
include feature contracts and evaluation limits. Raw training/evaluation vectors
are not part of the public source release.
# Resident shared library

Build `cmake --build build --target decision_head --config Release` to produce
`build/libdecision_head.so` on Linux or `build/Release/decision_head.dll` with
the Windows Visual Studio generator. This CPU-only target does not rebuild the
LLM/CUDA executable. The existing static libraries and CLI remain available.

FFI consumers must check `decision_head_abi_version() == 1` before using the
public structs in `include/decision-head.h`. The Windows ABI is cdecl. Keep the
library loaded while any head/workspace exists; free workspaces before their
head. Sharing a head is safe, but each concurrent call needs its own workspace.
Version 1 includes the current struct layouts and all exported function
signatures; incompatible changes require an ABI version increment.
