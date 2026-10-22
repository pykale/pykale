"""Named-input fusion parity cases adapted from KaleCancer (MIT, PyKale contributors)."""

import pickle

import pytest
import torch

from kale.embed.multimodal_fusion import FusionContext, FusionMethod, MajorityVote, masked_mean, MaskedMean
from kale.embed.multimodal_fusion import NamedInputConcat as Concat
from kale.embed.multimodal_fusion import stack_present


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_masked_mean_precision_and_absent_gradients(dtype):
    values = torch.tensor([[[2.0], [float("nan")]], [[6.0], [8.0]]], dtype=dtype, requires_grad=True)
    result = masked_mean(values, torch.tensor([[True, False], [True, True]]))
    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.tensor([[4.0], [8.0]], dtype=dtype))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[[0.5], [0.0]], [[0.5], [1.0]]], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_voting_output_dtype_and_gradient_contract(dtype):
    values = torch.tensor([[0.75, 0.25]], dtype=dtype, requires_grad=True)
    result = MajorityVote("error")({"a": values}, {"a": torch.tensor([True])})
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert result.dtype == expected_dtype
    torch.testing.assert_close(result, torch.tensor([[11 / 12, 1 / 12]], dtype=expected_dtype))
    result[0, 0].backward()
    assert values.grad.dtype == dtype
    torch.testing.assert_close(values.grad, torch.tensor([[1 / 3, 0.0]], dtype=dtype))


@pytest.mark.parametrize(
    "values,present,error,message",
    [
        ([], {}, TypeError, "values and present must be mappings"),
        ({}, {}, ValueError, "fusion needs at least one value tensor"),
        ({"a": torch.ones(1, 2)}, {"a": torch.ones(1)}, TypeError, "presence masks must be boolean tensors"),
        ({"": torch.ones(1, 2)}, {"": torch.tensor([True])}, TypeError, "input names must be nonempty strings"),
        ({"a": torch.ones(1, 2)}, {}, ValueError, "input 'a' has no presence mask"),
        (
            {"a": torch.ones(1, 2)},
            {"a": torch.ones(1, dtype=torch.bool, device="meta")},
            ValueError,
            "values and presence masks must share a device",
        ),
    ],
)
def test_validation_failure_diagnostics(values, present, error, message):
    with pytest.raises(error) as caught:
        MaskedMean()(values, present)
    assert str(caught.value) == message


def test_context_copies_order_and_accepts_unknown_widths():
    widths = {"b": None, "a": 2}
    context = FusionContext.without_data(widths)
    widths["c"] = 3
    assert list(context.widths) == ["b", "a"]
    assert context.present.shape == (0, 2)
    assert context.required.shape == (0,)
    MaskedMean().check(context)
    with pytest.raises(ValueError, match="one width"):
        MaskedMean().check(FusionContext.without_data({"a": 2, "b": 3}))
    with pytest.raises(ValueError, match="two classes"):
        MajorityVote("error").check(FusionContext.without_data({"a": 1}))


@pytest.mark.parametrize(
    "present, required, widths, error",
    [
        (torch.ones(1, 1), torch.tensor([True]), {"a": 2}, TypeError),
        (torch.ones(1, 1, dtype=torch.bool), torch.ones(1), {"a": 2}, TypeError),
        (torch.ones(1, dtype=torch.bool), torch.tensor([True]), {"a": 2}, ValueError),
        (torch.ones(1, 1, dtype=torch.bool), torch.tensor([True, False]), {"a": 2}, ValueError),
        (torch.ones(1, 1, dtype=torch.bool), torch.tensor([True]), {"a": 2, "b": 2}, ValueError),
        (torch.ones(1, 1, dtype=torch.bool), torch.tensor([True]), {"a": True}, TypeError),
        (torch.ones(1, 1, dtype=torch.bool), torch.tensor([True]), {"a": 0}, ValueError),
        (torch.ones(1, 1, dtype=torch.bool), torch.tensor([True]), {"": 2}, TypeError),
    ],
)
def test_context_rejects_malformed_fields(present, required, widths, error):
    with pytest.raises(error):
        FusionContext(present, required, widths)


def test_required_rows_and_abstract_extension_contract():
    with pytest.raises(TypeError):
        FusionMethod()

    class Custom(MaskedMean):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def check(self, context):
            super().check(context)

        def forward(self, values, present):
            return super().forward(values, present) * self.scale

    present = torch.tensor([[True, False], [False, False]])
    context = FusionContext(present, torch.tensor([True, False]), {"a": 2, "b": 2})
    method = Custom()
    method.check(context)
    assert list(method.state_dict()) == ["scale"]
    result = method({"a": torch.tensor([[2.0, 4.0]])}, {"a": torch.tensor([True])})
    torch.testing.assert_close(result, torch.tensor([[4.0, 8.0]]))
    result.sum().backward()
    assert method.scale.grad.item() == 6.0
    with pytest.raises(ValueError, match="required row"):
        method.check(FusionContext(present, torch.tensor([True, True]), context.widths))


@pytest.mark.parametrize("method", [Concat(), MaskedMean(), MajorityVote("error")])
def test_empty_batch_with_known_width(method):
    result = method({"a": torch.empty(0, 2)}, {"a": torch.empty(0, dtype=torch.bool)})
    assert result.shape == (0, 2)


@pytest.mark.parametrize("method", [Concat(), MaskedMean(), MajorityVote("error")])
def test_rejects_undefined_rows_and_missing_present_inputs(method):
    with pytest.raises(ValueError, match="defined"):
        method({"a": torch.tensor([[0.8, 0.2]])}, {"a": torch.tensor([False])})
    with pytest.raises(ValueError, match="no value tensor"):
        method({"a": torch.tensor([[0.8, 0.2]])}, {"a": torch.tensor([True]), "b": torch.tensor([True])})


@pytest.mark.parametrize("method", [MaskedMean(), MajorityVote("error")])
def test_rejects_unequal_actual_widths(method):
    with pytest.raises(ValueError, match="one width"):
        method({"a": torch.ones(1, 2), "b": torch.ones(1, 3)}, dict.fromkeys(["a", "b"], torch.tensor([True])))


@pytest.mark.parametrize("probabilities", [[[0.8, 0.8]], [[-0.2, 1.2]], [[float("nan"), 1.0]]])
def test_voting_rejects_invalid_present_probabilities(probabilities):
    with pytest.raises(ValueError):
        MajorityVote("error")({"a": torch.tensor(probabilities)}, {"a": torch.tensor([True])})


def test_invalid_tie_policy():
    with pytest.raises(ValueError, match="tie_break"):
        MajorityVote("first")


@pytest.mark.parametrize(
    "values,present,error",
    [
        ({}, {}, ValueError),
        ({"a": torch.ones(1, 2)}, {}, ValueError),
        ({"a": torch.ones(1, 2)}, {"a": torch.ones(1)}, TypeError),
        ({"a": torch.ones(1, 2)}, {"a": torch.ones(1, 1, dtype=torch.bool)}, ValueError),
        ({"a": torch.ones(2)}, {"a": torch.ones(1, dtype=torch.bool)}, ValueError),
        ({"a": torch.ones(1, 2, dtype=torch.int64)}, {"a": torch.tensor([True])}, TypeError),
        ({"a": torch.ones(1, 0)}, {"a": torch.tensor([True])}, ValueError),
        ({"a": torch.tensor([[float("inf"), 1.0]])}, {"a": torch.tensor([True])}, ValueError),
    ],
)
def test_named_input_validation(values, present, error):
    with pytest.raises(error):
        MaskedMean()(values, present)


def test_rejects_mixed_dtypes_and_batch_sizes():
    present = dict.fromkeys(["a", "b"], torch.tensor([True]))
    with pytest.raises(TypeError, match="dtype"):
        MaskedMean()({"a": torch.ones(1, 2), "b": torch.ones(1, 2, dtype=torch.float64)}, present)
    with pytest.raises(ValueError, match="shape"):
        MaskedMean()({"a": torch.ones(1, 2), "b": torch.ones(2, 2)}, present)


def test_context_rejects_empty_inputs_and_mixed_devices():
    with pytest.raises(ValueError, match="at least one"):
        FusionContext.without_data({})
    with pytest.raises(ValueError, match="device"):
        FusionContext(torch.ones(1, 1, dtype=torch.bool), torch.ones(1, dtype=torch.bool, device="meta"), {"a": 2})


def test_concat_does_not_treat_omitted_input_as_present():
    with pytest.raises(ValueError, match="defined"):
        Concat()({"a": torch.ones(1, 2)}, {"a": torch.tensor([True]), "b": torch.tensor([False])})


def test_voting_probability_tolerance_and_first_class_ties():
    present = {"a": torch.tensor([True])}
    result = MajorityVote("error")({"a": torch.tensor([[0.5, 0.5]])}, present)
    assert result.argmax(dim=1).item() == 0
    MajorityVote("error")({"a": torch.tensor([[0.8, 0.20001]])}, present)


def test_adapter_dispatch_uses_only_shared_base_and_documented_state():
    class ApplicationBase(FusionMethod):
        pass

    class ApplicationMean(ApplicationBase):
        defined = MaskedMean.defined
        forward = MaskedMean.forward

    class ApplicationVote(ApplicationBase):
        defined = MajorityVote.defined
        forward = MajorityVote.forward

        def __init__(self):
            super().__init__()
            self.tie_break = "error"

    values, present = {"a": torch.tensor([[0.8, 0.2]])}, {"a": torch.tensor([True])}
    mean, vote = ApplicationMean(), ApplicationVote()
    assert isinstance(mean, ApplicationBase) and isinstance(vote, ApplicationBase)
    torch.testing.assert_close(mean(values, present), values["a"])
    torch.testing.assert_close(vote(values, present), torch.tensor([[14 / 15, 1 / 15]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_concat_preserves_mapping_order_and_gradients(dtype):
    first = torch.tensor([[2.0, 3.0]], dtype=dtype, requires_grad=True)
    second = torch.tensor([[7.0]], dtype=dtype, requires_grad=True)
    present = {"second": torch.tensor([True]), "first": torch.tensor([True])}
    result = Concat()({"second": second, "first": first}, present)
    torch.testing.assert_close(result, torch.tensor([[7.0, 2.0, 3.0]], dtype=dtype))
    result.sum().backward()
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.ones_like(second))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_mean_fixed_outputs_and_gradients(dtype):
    first = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=dtype, requires_grad=True)
    second = torch.tensor([[10.0, 12.0], [float("nan"), float("nan")]], dtype=dtype, requires_grad=True)
    present = {"a": torch.tensor([True, True]), "b": torch.tensor([True, False])}
    result = MaskedMean()({"a": first, "b": second}, present)
    torch.testing.assert_close(result, torch.tensor([[6.0, 8.0], [6.0, 8.0]], dtype=dtype))
    result.sum().backward()
    torch.testing.assert_close(first.grad, torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=dtype))
    torch.testing.assert_close(second.grad, torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=dtype))


def test_empty_row_is_zero_only_at_helper_level():
    values = torch.full((2, 1, 3), float("nan"), requires_grad=True)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    result = masked_mean(values, mask)
    torch.testing.assert_close(result, torch.zeros(1, 3))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
    for method in (Concat(), MaskedMean(), MajorityVote(tie_break="error")):
        assert not method.defined(mask.T).item()


@pytest.mark.parametrize("method", [MaskedMean(), MajorityVote(tie_break="error")])
def test_entirely_absent_input_can_be_omitted(method):
    values = {"available": torch.tensor([[0.8, 0.2]])}
    present = {"absent": torch.tensor([False]), "available": torch.tensor([True])}
    stacked, mask = stack_present(values, present)
    torch.testing.assert_close(stacked, values["available"].unsqueeze(0))
    assert mask.tolist() == [[True]]
    result = method(values, present)
    expected = [[0.8, 0.2]] if isinstance(method, MaskedMean) else [[14 / 15, 1 / 15]]
    torch.testing.assert_close(result, torch.tensor(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tie_break", ["mean_probability", "error"])
def test_vote_scores_and_probability_gradients(dtype, tie_break):
    first = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=dtype, requires_grad=True)
    second = torch.tensor([[0.6, 0.4], [float("nan"), float("nan")]], dtype=dtype, requires_grad=True)
    present = {"a": torch.tensor([True, True]), "b": torch.tensor([True, False])}
    result = MajorityVote(tie_break=tie_break)({"a": first, "b": second}, present)
    torch.testing.assert_close(result, torch.tensor([[0.94, 0.06], [0.1, 0.9]], dtype=dtype))
    # One class avoids the constant sum of all vote scores; argmax boundaries are excluded.
    result[:, 0].sum().backward()
    torch.testing.assert_close(first.grad, torch.tensor([[0.1, 0.0], [1 / 3, 0.0]], dtype=dtype))
    torch.testing.assert_close(second.grad, torch.tensor([[0.1, 0.0], [0.0, 0.0]], dtype=dtype))


def test_vote_ties_have_fixed_scores_and_error_policy():
    values = {"a": torch.tensor([[0.9, 0.1], [0.6, 0.4]]), "b": torch.tensor([[0.3, 0.7], [0.4, 0.6]])}
    present = dict.fromkeys(values, torch.tensor([True, True]))
    result = MajorityVote(tie_break="mean_probability")(values, present)
    torch.testing.assert_close(result, torch.tensor([[0.52, 0.48], [0.5, 0.5]]))
    assert result.argmax(dim=1).tolist() == [0, 0]
    with pytest.raises(ValueError, match="2 patients have tied votes"):
        MajorityVote(tie_break="error")(values, present)


@pytest.mark.parametrize("method", [Concat(), MaskedMean(), MajorityVote(tie_break="error")])
def test_public_class_paths_and_current_pickle_roundtrip(method):
    from kale.embed import multimodal_fusion as fusion

    assert getattr(fusion, type(method).__name__) is type(method)
    restored = pickle.loads(pickle.dumps(method))
    assert type(restored) is type(method)
    assert isinstance(restored, FusionMethod)
    assert not restored.state_dict()
    if isinstance(method, MajorityVote):
        assert restored.tie_break == method.tie_break


def test_concat_keeps_raw_non_finite_values_and_promotes_dtypes():
    present = dict.fromkeys(["table", "other"], torch.tensor([True, True]))
    table = torch.tensor([[1.0, float("nan")], [float("inf"), 4.0]])
    result = Concat()({"table": table, "other": torch.tensor([[5.0], [6.0]], dtype=torch.float64)}, present)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result[:, :2], table.double(), equal_nan=True)
    with pytest.raises(ValueError, match="defined"):
        Concat()({"table": table}, {"table": torch.tensor([True, False])})


@pytest.mark.parametrize("method", [MaskedMean(), MajorityVote("error")])
def test_mean_and_voting_still_reject_non_finite_present_values(method):
    with pytest.raises(ValueError, match="finite"):
        method({"a": torch.tensor([[float("nan"), 1.0]])}, {"a": torch.tensor([True])})


def test_disabled_validation_skips_checks_without_changing_results():
    values = {"a": torch.tensor([[0.8, 0.2], [0.3, 0.7]]), "b": torch.tensor([[0.6, 0.4], [float("nan")] * 2])}
    present = {"a": torch.tensor([True, True]), "b": torch.tensor([True, False])}
    for make in (MaskedMean, lambda: MajorityVote("error"), Concat):
        checked, unchecked = make(), make()
        unchecked.validate_inputs = False
        if isinstance(checked, Concat):
            full = dict.fromkeys(present, torch.tensor([True, True]))
            torch.testing.assert_close(unchecked(values, full), checked(values, full), equal_nan=True)
        else:
            torch.testing.assert_close(unchecked(values, present), checked(values, present))
    # an undefined row is the caller's responsibility once validation is off
    undefined = MaskedMean()
    undefined.validate_inputs = False
    torch.testing.assert_close(undefined({"a": torch.ones(1, 2)}, {"a": torch.tensor([False])}), torch.zeros(1, 2))
    assert FusionMethod.validate_inputs and "validate_inputs" not in vars(MaskedMean())


def test_tie_error_ignores_rows_without_branches_when_unvalidated():
    vote = MajorityVote("error")
    vote.validate_inputs = False
    present = {"a": torch.tensor([True, False])}
    result = vote({"a": torch.tensor([[0.8, 0.2], [float("nan")] * 2])}, present)
    torch.testing.assert_close(result[0], torch.tensor([14 / 15, 1 / 15]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_voting_accepts_half_precision_softmax(dtype):
    torch.manual_seed(0)
    probabilities = {name: torch.randn(512, 7).to(dtype).softmax(dim=-1) for name in ("a", "b")}
    present = dict.fromkeys(probabilities, torch.ones(512, dtype=torch.bool))
    assert MajorityVote("mean_probability")(probabilities, present).dtype == torch.float32
    with pytest.raises(ValueError, match="summing to one"):
        MajorityVote("error")({"a": torch.tensor([[0.8, 0.8]], dtype=dtype)}, {"a": torch.tensor([True])})


@pytest.mark.parametrize(
    "dtype, first",
    [
        (torch.float16, [0.748046875, 0.251953125]),
        (torch.bfloat16, [0.74609375, 0.25390625]),
    ],
)
def test_half_precision_voting_keeps_the_probability_tie_breaker(dtype, first):
    # one vote each; class 1 has the higher mean probability, by less than half-precision spacing near 0.5
    values = {"a": torch.tensor([first], dtype=dtype), "b": torch.tensor([[0.25, 0.75]], dtype=dtype)}
    assert values["a"].float().tolist() == [first]
    result = MajorityVote("mean_probability")(values, dict.fromkeys(values, torch.tensor([True])))
    assert result.dtype == torch.float32
    assert result.argmax(dim=1).tolist() == [1]


def test_float16_mean_does_not_overflow():
    # bfloat16 shares float32's exponent range, so only float16 sums can overflow where float32 sums do not
    values = {name: torch.full((1, 2), 40000.0, dtype=torch.float16, requires_grad=True) for name in ("a", "b")}
    result = MaskedMean()(values, dict.fromkeys(values, torch.tensor([True])))
    assert result.dtype == torch.float16
    torch.testing.assert_close(result, values["a"].detach())
    result.sum().backward()
    for value in values.values():
        torch.testing.assert_close(value.grad, torch.full_like(value, 0.5))


def test_context_compares_by_identity():
    context = FusionContext.without_data({"a": 2})
    assert context == context
    assert context != FusionContext.without_data({"a": 2})
    assert len({context, context}) == 1


def test_adapter_dispatch_with_validation_disabled_on_the_application_base():
    class ApplicationBase(FusionMethod):
        validate_inputs = False

    class ApplicationMean(ApplicationBase):
        defined = MaskedMean.defined
        forward = MaskedMean.forward

    method = ApplicationMean()
    assert not method.validate_inputs
    values = {"a": torch.tensor([[2.0, 4.0]]), "b": torch.tensor([[6.0, 8.0]])}
    present = dict.fromkeys(values, torch.tensor([True]))
    torch.testing.assert_close(method(values, present), torch.tensor([[4.0, 6.0]]))
