# Multimodal fusion with missing modalities

`kale.embed.multimodal_fusion` provides `FusionMethod`, `FusionContext`, `NamedInputConcat`, `MaskedMean` and `MajorityVote` for named inputs with explicit missingness, and the numerical functions `stack_present` and `masked_mean` they are built on. These additions originate from KaleCancer's fusion design and preserve its numerical behavior. Existing sequence-based `Concat` and probabilistic `ProductOfExperts` APIs remain unchanged.

## Combine named embeddings

```python
import torch
from kale.embed.multimodal_fusion import FusionContext, MaskedMean

values = {
    "clinical": torch.tensor([[2., 4.], [6., 8.]]),
    "image": torch.tensor([[10., 12.], [float("nan"), float("nan")]]),
}
present = {"clinical": torch.tensor([True, True]), "image": torch.tensor([True, False])}
method = MaskedMean()
context = FusionContext(
    present=torch.stack(list(present.values()), dim=1),
    required=torch.tensor([True, True]),
    widths={"clinical": 2, "image": 2},
)
method.check(context)
embedding = method(values, present)  # [[6., 8.], [6., 8.]]
```

Context columns follow `widths` order. `defined` takes a boolean `(batch, inputs)` matrix; `forward` takes named `(batch, features)` vectors and named `(batch,)` boolean masks. Values determine concatenation order, while masks are matched by name. Known widths are positive integers; `FusionContext.without_data(widths)` accepts unknown widths as `None` for construction-time checks. The context copies width mappings but callers must not mutate tensor contents during validation. Contexts compare by identity.

Select rows accepted by `defined` before calling `forward`. All-empty rows are invalid predictions. An entirely absent input may be omitted from values for averaging/voting, with an all-false mask retained. Absent entries may contain NaN and contribute no gradients. Present entries must be finite for averaging and voting. Concatenation accepts non-finite present entries, because raw inputs can carry missing cells for an encoder that handles them, and it promotes differing floating dtypes as `torch.cat` does. Averaging and voting require one floating dtype. Values share a device, and masks share it too. Empty batches with known feature widths are supported.

## Choose an operation

| Class | Valid rows | Result |
| --- | --- | --- |
| `NamedInputConcat()` | Every input present | Concatenates vectors in value mapping order without flattening; widths and floating dtypes may differ. |
| `MaskedMean()` | At least one input present | Mean of present equal-width embeddings. |
| `MajorityVote(tie_break)` | At least one branch present | Vote scores over aligned class-probability vectors, with at least two classes. |

Voting requires probabilities in `[0, 1]` summing to one within an absolute tolerance of `max(1e-4, 4 * eps)` for the input dtype, so float16 and bfloat16 softmax outputs are accepted; it does not convert logits. The class order must agree across branches. Scores are `(votes + 0.5 * mean_probability) / (present_branches + 0.5)`. With `tie_break="mean_probability"`, probabilities break equal vote counts; exact score ties select the first class under argmax. With `tie_break="error"`, tied top vote counts raise even if probabilities differ. Scores are not calibrated probabilities. Gradients flow through the probability contribution, not discrete argmax votes.

## Validation cost

By default, `forward` validates its inputs on every call: names, shapes, dtypes, finite present values, defined rows and, for voting, probability sums. These checks read tensor values back to Python, which synchronises CUDA execution on every batch. When the caller already validates once, for example with `check` and a dry run before training, and passes only rows accepted by `defined`, set `validate_inputs = False` on the subclass or instance. Results are unchanged; malformed inputs then become the caller's responsibility. `check` always validates.

```python
method = MaskedMean()
method.validate_inputs = False  # inputs validated before training
```

## Extend and adapt

Subclass `FusionMethod`, implement `defined` and `forward`, and optionally override `check` calling `super().check(context)`. Custom parameters belong in `__init__` as for other PyTorch modules. The core context carries no dataset IDs, stage labels, target classes or prediction-column names.

Application adapters retain indexed row alignment, stage restrictions, target compatibility and output naming. In particular, averaging different subsets of Cox branch log-hazards can change rankings because branch offsets are arbitrary; applications must enforce their survival-specific completeness policy. Raw core classes do not automatically satisfy an application's own subclass guard.

KaleCancer adoption is separate and waits for a released core API. Existing cancer imports and YAML paths must be preserved through compatibility classes, with custom-subclass and historical saved-model checks at adoption. No application dependencies or adapters are introduced by these core additions.

## Numerical precision and compatibility

`masked_mean` and `MaskedMean` accumulate in at least float32 and return the input dtype. This avoids float16 summation overflow when the intermediate sum fits in float32; output rounding and underflow remain possible, and extreme sums can still overflow the accumulation dtype. Float64 inputs retain float64 accumulation.

`MajorityVote` returns float32 scores for float16 and bfloat16 inputs, preserving small probability differences used to break vote-count ties. Float32 and float64 inputs retain their dtype. This intentionally differs from KaleCancer's original half-precision output and the first development draft of this API; application adoption must check downstream dtype assumptions. Float32 promotion uses additional memory for half-precision inputs. These are new core APIs, and the existing core fusion classes are unchanged.
