"""Fuse modality tensors, Gaussian experts, or named vectors with availability masks.

``Concat``, ``BimodalInteractionFusion`` and ``LowRankTensorFusion`` combine sequences of modality tensors.
``ProductOfExperts`` combines stacked Gaussian means and log-variances.

Named-input fusion combines a mapping of ``(batch, features)`` vectors with a mapping of ``(batch,)`` boolean presence
masks, matched by name. ``NamedInputConcat`` needs every input in a row; ``MaskedMean`` and ``MajorityVote`` need at
least one. The order of the value mapping sets the output order. Absent entries may contain NaN and contribute neither
values nor gradients. ``FusionMethod.defined`` returns the rows a method can combine, and only those rows should be
passed to ``forward``. ``FusionMethod.check`` validates a ``FusionContext`` of availability, required rows and feature
widths, for example before training. To add a method, subclass ``FusionMethod`` and implement ``defined`` and
``forward``. ``stack_present`` and ``masked_mean`` are the numerical helpers the built-in methods use.

The sequence-fusion reference is https://github.com/pliang279/MultiBench/blob/main/fusions/common_fusions.py.
Named-input fusion is adapted from https://github.com/pykale/cancer.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Variable


@dataclass(frozen=True, eq=False)
class FusionContext:
    """Store input availability and widths for task-independent fusion validation.

    Fields are validated on construction and again by ``FusionMethod.check``.

    Args:
        present (Tensor): Boolean tensor of shape ``(batch, inputs)``. Columns follow the iteration
            order of ``widths``.
        required (Tensor): Boolean tensor of shape ``(batch,)`` marking rows that must have a fused
            result. Must share the presence tensor's device.
        widths (Mapping[str, Optional[int]]): Nonempty mapping from input names to positive feature
            widths, or ``None`` for widths not yet known. The mapping is copied at construction.

    Raises:
        TypeError: If names, known widths, or tensor dtypes have invalid types.
        ValueError: If shapes, devices, or known width values are inconsistent.
    """

    present: torch.Tensor
    required: torch.Tensor
    widths: Mapping[str, Optional[int]]

    def __post_init__(self):
        """Copy the width mapping and validate the context."""
        if not isinstance(self.widths, Mapping):
            raise TypeError("widths must be a mapping")
        object.__setattr__(self, "widths", dict(self.widths))
        self._validate()

    def _validate(self):
        """Reject invalid field types, widths, shapes, or devices."""
        _check_widths(self.widths)
        _check_presence(self.present)
        if self.present.shape[1] != len(self.widths):
            raise ValueError("presence columns must match widths")
        _check_required(self.required, self.present)

    @classmethod
    def without_data(cls, widths: Mapping[str, Optional[int]]) -> "FusionContext":
        """Create an empty-row CPU context for construction-time checks.

        Args:
            widths (Mapping[str, Optional[int]]): Ordered input names and known positive widths, or
                ``None`` when unknown.

        Returns:
            FusionContext: Context with boolean tensors shaped ``(0, inputs)`` and ``(0,)``.

        Raises:
            TypeError: If the mapping entries have invalid types.
            ValueError: If there are no inputs or a known width is nonpositive.
        """
        return cls(torch.empty(0, len(widths), dtype=torch.bool), torch.empty(0, dtype=torch.bool), widths)


def stack_present(
    values: Mapping[str, torch.Tensor], present: Mapping[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack named values and presence masks in value mapping order.

    This numerical helper does not validate availability or select rows. Callers must ensure omitted
    inputs are absent and align batch rows across inputs.

    Args:
        values (Mapping[str, Tensor]): Nonempty mapping of input names to equal-shaped ``(batch,
            features)`` tensors compatible with ``torch.stack``.
        present (Mapping[str, Tensor]): Boolean ``(batch,)`` masks keyed by input name on the
            values' device. Masks for names omitted from ``values`` are ignored.

    Returns:
        tuple[Tensor, Tensor]: Stacked values shaped ``(inputs, batch, features)`` and masks shaped
            ``(inputs, batch)``.
    """
    return torch.stack(list(values.values())), torch.stack([present[name] for name in values])


def masked_mean(stacked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average available input vectors independently for each batch row.

    Absent entries contribute neither values nor gradients. Inputs are not validated. Sums are
    accumulated in at least float32, so float16 sums do not overflow, and the result is returned in
    the input dtype.

    Args:
        stacked (Tensor): Floating values shaped ``(inputs, batch, features)``. Absent entries may
            contain NaN.
        mask (Tensor): Boolean availability shaped ``(inputs, batch)`` on the values' device.

    Returns:
        Tensor: Means shaped ``(batch, features)`` with the values' dtype and device. Rows without
            an available input return zero, which is not a valid prediction.
    """
    accumulate = torch.promote_types(stacked.dtype, torch.float32)
    values = stacked.to(accumulate)
    filled = torch.where(mask.unsqueeze(-1), values, torch.zeros((), dtype=accumulate, device=stacked.device))
    count = mask.sum(dim=0).clamp(min=1).unsqueeze(-1).to(accumulate)
    return (filled.sum(dim=0) / count).to(stacked.dtype)


def _check_input_name(name: object) -> None:
    """Reject an input name that is not a nonempty string.

    Raises:
        TypeError: If the name is not a nonempty string.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("input names must be nonempty strings")


def _check_known_width(width: object) -> None:
    """Reject a known width that is not a positive integer; ``None`` marks an unknown width.

    Raises:
        TypeError: If the width is a boolean or not an integer.
        ValueError: If the width is not positive.
    """
    if width is None:
        return
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("known widths must be integers")
    if width <= 0:
        raise ValueError("known widths must be positive")


def _check_widths(widths: Mapping[str, Optional[int]]) -> None:
    """Validate a nonempty mapping of input names to known widths or ``None``.

    Raises:
        TypeError: If a name or known width has an invalid type.
        ValueError: If there are no inputs or a known width is not positive.
    """
    if not widths:
        raise ValueError("fusion needs at least one input")
    for name, width in widths.items():
        _check_input_name(name)
        _check_known_width(width)


def _check_required(required: torch.Tensor, present: torch.Tensor) -> None:
    """Validate the required-row mask against an already validated presence matrix.

    Raises:
        TypeError: If the mask is not a boolean tensor.
        ValueError: If it has the wrong shape or another device.
    """
    if not isinstance(required, torch.Tensor) or required.dtype != torch.bool:
        raise TypeError("required must be a boolean tensor")
    if required.shape != (present.shape[0],):
        raise ValueError("required must have one entry per row")
    if required.device != present.device:
        raise ValueError("required and present must share a device")


def _check_presence(present: torch.Tensor) -> None:
    """Validate a batch-by-input availability matrix.

    Args:
        present (Tensor): Expected boolean ``(batch, inputs)`` tensor with at least one input
            column.

    Raises:
        TypeError: If the input is not a boolean tensor.
        ValueError: If the shape is invalid or there are no input columns.
    """
    if not isinstance(present, torch.Tensor) or present.dtype != torch.bool:
        raise TypeError("present must be a boolean tensor")
    if present.ndim != 2 or present.shape[1] == 0:
        raise ValueError("present must have shape (batch, inputs) with at least one input")


def _check_named_inputs(
    values: Mapping[str, torch.Tensor],
    present: Mapping[str, torch.Tensor],
    require_all: bool = False,
    require_finite: bool = True,
    same_dtype: bool = True,
) -> None:
    """Validate named vectors and masks without changing input order or contents.

    Value-dependent checks synchronize CUDA execution; use outside a validated training path when
    that overhead matters.

    Args:
        values (Mapping[str, Tensor]): Nonempty mapping of floating ``(batch, features)`` values
            with positive feature widths.
        present (Mapping[str, Tensor]): Boolean ``(batch,)`` masks on the values' device. An omitted
            value must have an entirely false mask.
        require_all (bool): Require every input for each row instead of at least one. Defaults to
            False.
        require_finite (bool): Require finite entries in present rows. Defaults to True; disabled
            for raw concatenation.
        same_dtype (bool): Require a common floating dtype. Defaults to True; disabled when
            concatenation may promote dtypes.

    Raises:
        TypeError: If mappings, names, tensors, or dtypes are invalid.
        ValueError: If shapes, devices, availability, or finite-value requirements fail.
    """
    reference = _reference_value(values, present)
    for name, mask in present.items():
        _check_mask(name, mask, reference, values)
    for name, value in values.items():
        _check_value(name, value, reference, present, same_dtype)
        if require_finite and not bool(torch.isfinite(value[present[name]]).all()):
            raise ValueError("present values must be finite")
    _check_rows_defined(present, require_all)


def _reference_value(values: Mapping[str, torch.Tensor], present: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the first value tensor, after checking the containers and its kind and shape.

    Raises:
        TypeError: If either container is not a mapping or the first value is not a floating tensor.
        ValueError: If there are no values or the first value is not two-dimensional.
    """
    if not isinstance(values, Mapping) or not isinstance(present, Mapping):
        raise TypeError("values and present must be mappings")
    if not values:
        raise ValueError("fusion needs at least one value tensor")
    reference = next(iter(values.values()))
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise TypeError("values must be floating tensors")
    if reference.ndim != 2:
        raise ValueError("values must have shape (batch, features)")
    return reference


def _check_mask(name: str, mask: torch.Tensor, reference: torch.Tensor, values: Mapping[str, torch.Tensor]) -> None:
    """Validate one named presence mask against the reference value.

    Raises:
        TypeError: If the name or mask has an invalid type.
        ValueError: If the mask has another shape or device, or marks an omitted input present.
    """
    _check_input_name(name)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("presence masks must be boolean tensors")
    if mask.shape != (reference.shape[0],):
        raise ValueError("presence masks must have shape (batch,)")
    if mask.device != reference.device:
        raise ValueError("values and presence masks must share a device")
    if name not in values and bool(mask.any()):
        raise ValueError(f"present input {name!r} has no value tensor")


def _check_value(
    name: str,
    value: torch.Tensor,
    reference: torch.Tensor,
    present: Mapping[str, torch.Tensor],
    same_dtype: bool,
) -> None:
    """Validate one named value tensor against the reference value, without reading its contents.

    Raises:
        TypeError: If the name, tensor kind, or dtype is invalid.
        ValueError: If the value has no mask, or another shape or device.
    """
    _check_input_name(name)
    if name not in present:
        raise ValueError(f"input {name!r} has no presence mask")
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("values must be floating tensors")
    if value.ndim != 2 or value.shape[0] != reference.shape[0] or value.shape[1] == 0:
        raise ValueError("values must have shape (batch, positive_features)")
    if same_dtype and value.dtype != reference.dtype:
        raise TypeError("values must share dtype")
    if value.device != reference.device:
        raise ValueError("values must share device")


def _check_rows_defined(present: Mapping[str, torch.Tensor], require_all: bool) -> None:
    """Reject rows missing any input when require_all is true, or all inputs otherwise.

    Raises:
        ValueError: If a row is not defined under the requirement.
    """
    masks = torch.stack(list(present.values()), dim=1)
    valid = masks.all(dim=1) if require_all else masks.any(dim=1)
    if not bool(valid.all()):
        raise ValueError("forward only accepts rows defined by the fusion method")


def _check_equal_widths(widths: Iterable[Optional[int]]) -> None:
    """Reject conflicting known input widths.

    Args:
        widths (Iterable[Optional[int]]): Widths already validated as positive integers or ``None``.

    Raises:
        ValueError: If at least two known widths differ.
    """
    known = [width for width in widths if width is not None]
    if len(set(known)) > 1:
        raise ValueError("fusion needs inputs of one width")


def _check_probabilities(stacked: torch.Tensor, mask: torch.Tensor) -> None:
    """Validate present class distributions using a dtype-dependent tolerance.

    Row sums use at least float32 precision. Missing rows are excluded; value checks synchronize
    CUDA execution.

    Args:
        stacked (Tensor): Floating ``(inputs, batch, classes)`` probabilities, already checked for
            finite present values.
        mask (Tensor): Boolean ``(inputs, batch)`` presence on the probabilities' device.

    Raises:
        ValueError: If a present probability is outside ``[0, 1]`` or its row sum differs from one
            by more than ``max(1e-4, 4 * eps)``.
    """
    probabilities = stacked[mask]
    # half-precision softmax rows miss one by up to about half an epsilon, so a fixed 1e-4 rejects valid inputs
    tolerance = max(1e-4, 4 * torch.finfo(stacked.dtype).eps)
    sums = probabilities.sum(dim=-1, dtype=torch.promote_types(stacked.dtype, torch.float32))
    if bool(((probabilities < 0) | (probabilities > 1)).any()) or not torch.allclose(
        sums, torch.ones_like(sums), atol=tolerance, rtol=0
    ):
        raise ValueError("voting expects class probabilities summing to one")


class FusionMethod(nn.Module, ABC):
    """Define fusion over named input vectors with explicit availability masks.

    Subclasses implement ``defined`` and ``forward``, and may extend ``check`` by calling
    ``super().check(context)``. Task-specific checks, class order and row selection stay with the caller.

    Attributes:
        validate_inputs (bool): Check inputs on every built-in ``forward`` call. Defaults to True.
            The checks read tensor values, which synchronizes CUDA execution. Set False on an
            instance or subclass when the inputs are already validated, for example with ``check``
            and a dry run. Results for valid inputs are unchanged, and ``check`` still validates.
    """

    validate_inputs: bool = True

    @abstractmethod
    def defined(self, present: torch.Tensor) -> torch.Tensor:
        """Identify rows that this fusion method can combine.

        Args:
            present (Tensor): Boolean availability shaped ``(batch, inputs)``.

        Returns:
            Tensor: Boolean ``(batch,)`` validity on the input device.
        """

    def check(self, context: FusionContext) -> None:
        """Validate context structure and coverage of required rows.

        The context is revalidated because its tensors may have changed since construction.

        Args:
            context (FusionContext): Input availability, required rows, and known widths.

        Raises:
            TypeError: If the context type or its field types are invalid.
            ValueError: If fields are inconsistent or a required row is undefined.
        """
        if not isinstance(context, FusionContext):
            raise TypeError("context must be a FusionContext")
        context._validate()
        if bool((context.required & ~self.defined(context.present)).any()):
            raise ValueError("fusion cannot combine every required row")

    @abstractmethod
    def forward(self, values: Mapping[str, torch.Tensor], present: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Combine named vectors for rows accepted by ``defined``.

        Args:
            values (Mapping[str, Tensor]): Floating ``(batch, features)`` vectors in the intended
                fusion order.
            present (Mapping[str, Tensor]): Boolean ``(batch,)`` masks matched by input name.
                Entirely absent inputs may be omitted from values when the method permits.

        Returns:
            Tensor: Fused values shaped ``(batch, output_features)``. Concrete methods define widths
                and numerical semantics.
        """


class NamedInputConcat(FusionMethod):
    """Concatenate named vectors in mapping order along their feature dimension.

    Every selected row must have every input. Inputs are two-dimensional and are not flattened,
    unlike sequence-based ``Concat``. Feature widths and floating dtypes may differ; ``torch.cat``
    determines output dtype promotion. Raw non-finite values are preserved for downstream encoders
    that handle missing cells.
    """

    def defined(self, present: torch.Tensor) -> torch.Tensor:
        """Identify rows with every input present.

        Args:
            present (Tensor): Boolean availability shaped ``(batch, inputs)`` with at least one
                input column.

        Returns:
            Tensor: Boolean validity shaped ``(batch,)`` on the input device.

        Raises:
            TypeError: If presence is not a boolean tensor.
            ValueError: If its shape or input-column count is invalid.
        """
        _check_presence(present)
        return present.all(dim=1)

    def forward(self, values: Mapping[str, torch.Tensor], present: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate named input vectors without flattening or changing them.

        Args:
            values (Mapping[str, Tensor]): Nonempty mapping of ``(batch, features_i)`` floating
                vectors. Iteration order determines output feature order.
            present (Mapping[str, Tensor]): Boolean ``(batch,)`` masks. Every input must be present
                for every selected row.

        Returns:
            Tensor: Shape ``(batch, sum(features_i))`` on the inputs' device, with PyTorch dtype
                promotion and non-finite entries preserved.

        Raises:
            TypeError: If validation is enabled and input kinds or dtypes are invalid.
            ValueError: If validation is enabled and shapes, devices, or availability are
                inconsistent.
        """
        if self.validate_inputs:
            _check_named_inputs(values, present, require_all=True, require_finite=False, same_dtype=False)
        return torch.cat(list(values.values()), dim=-1)


class MaskedMean(FusionMethod):
    """Average equal-width named vectors over available inputs in each row.

    At least one input must be present for every selected row. Absent NaNs are excluded without
    affecting values or gradients. With validation enabled, present values must be finite and share
    a floating dtype and device. Inputs must share one embedding space; equal widths alone do not
    ensure that.
    """

    def defined(self, present: torch.Tensor) -> torch.Tensor:
        """Identify rows with at least one input present.

        Args:
            present (Tensor): Boolean availability shaped ``(batch, inputs)`` with at least one
                input column.

        Returns:
            Tensor: Boolean validity shaped ``(batch,)`` on the input device.

        Raises:
            TypeError: If presence is not a boolean tensor.
            ValueError: If its shape or input-column count is invalid.
        """
        _check_presence(present)
        return present.any(dim=1)

    def check(self, context: FusionContext) -> None:
        """Check required-row coverage and equal known feature widths.

        Args:
            context (FusionContext): Availability and known widths; unknown widths may be None.

        Raises:
            TypeError: If the context or field types are invalid.
            ValueError: If required rows are undefined, context fields disagree, or known widths
                differ.
        """
        super().check(context)
        _check_equal_widths(context.widths.values())

    def forward(self, values: Mapping[str, torch.Tensor], present: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Average each row over its available input vectors.

        Args:
            values (Mapping[str, Tensor]): Equal-width ``(batch, features)`` floating vectors.
                Entirely absent inputs may be omitted.
            present (Mapping[str, Tensor]): Boolean ``(batch,)`` masks matched by name. At least one
                input must be present per selected row.

        Returns:
            Tensor: Means shaped ``(batch, features)`` preserving dtype and device. Absent entries,
                including NaNs, have zero contribution and gradient.

        Raises:
            TypeError: If enabled validation finds invalid tensor types or dtypes.
            ValueError: If enabled validation finds incompatible shapes, widths, devices, non-finite
                present values, or undefined rows.
        """
        if self.validate_inputs:
            _check_named_inputs(values, present)
            _check_equal_widths(value.shape[1] for value in values.values())
        return masked_mean(*stack_present(values, present))


class MajorityVote(FusionMethod):
    """Combine classification branch votes into majority-vote scores.

    Branches must use the same class order. Scores are ``(votes + 0.5 * mean_probability) /
    (present_branches + 0.5)`` and are not calibrated probabilities. An exact score tie selects the
    first class under downstream argmax. Validated probability rows sum to one within ``max(1e-4, 4
    * eps)`` for their dtype; logits are not converted. Scores are computed and returned in at least
    float32, so half-precision rounding cannot erase the probability tie-breaker. With
    ``tie_break="error"``, tie detection reads tensor values even when input validation is disabled.

    Args:
        tie_break (str): ``"mean_probability"`` uses mean class probabilities to break equal vote
            counts; ``"error"`` rejects tied top vote counts.

    Raises:
        ValueError: If the tie policy is unsupported.
    """

    def __init__(self, tie_break: Literal["mean_probability", "error"]):
        """Initialize the tie policy documented on the class."""
        super().__init__()
        if tie_break not in ("mean_probability", "error"):
            raise ValueError("tie_break must be 'mean_probability' or 'error'")
        self.tie_break = tie_break

    def defined(self, present: torch.Tensor) -> torch.Tensor:
        """Identify rows with at least one classification branch present.

        Args:
            present (Tensor): Boolean availability shaped ``(batch, inputs)`` with at least one
                input column.

        Returns:
            Tensor: Boolean validity shaped ``(batch,)`` on the input device.

        Raises:
            TypeError: If presence is not a boolean tensor.
            ValueError: If its shape or input-column count is invalid.
        """
        _check_presence(present)
        return present.any(dim=1)

    def check(self, context: FusionContext) -> None:
        """Check coverage and equal widths of at least two classes.

        Args:
            context (FusionContext): Availability and class widths. A width may be None when
                unknown.

        Raises:
            TypeError: If the context or field types are invalid.
            ValueError: If required rows are undefined, fields disagree, known widths differ, or
                fewer than two classes are declared.
        """
        super().check(context)
        _check_equal_widths(context.widths.values())
        if any(width is not None and width < 2 for width in context.widths.values()):
            raise ValueError("voting needs at least two classes")

    def forward(self, values: Mapping[str, torch.Tensor], present: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Score aligned class probabilities using votes and a bounded tie term.

        Args:
            values (Mapping[str, Tensor]): ``(batch, classes)`` probability vectors in a shared
                class order. Absent entries may contain NaN.
            present (Mapping[str, Tensor]): Boolean ``(batch,)`` branch masks. Each selected row
                needs at least one branch.

        Returns:
            Tensor: Vote scores shaped ``(batch, classes)`` on the input device, with dtype
                ``torch.promote_types(input_dtype, torch.float32)``: float32 for float16 and
                bfloat16 inputs, otherwise the input dtype. Gradients flow through the probability
                contribution, not discrete votes.

        Raises:
            TypeError: If enabled validation finds invalid input types or dtypes.
            ValueError: If fewer than two classes are supplied, the error policy encounters tied
                votes, or enabled validation rejects shapes, availability, devices or probabilities.
        """
        if self.validate_inputs:
            _check_named_inputs(values, present)
            _check_equal_widths(value.shape[1] for value in values.values())
        stacked, mask = stack_present(values, present)
        num_classes = stacked.shape[-1]
        if num_classes < 2:
            raise ValueError("voting needs at least two classes")
        if self.validate_inputs:
            _check_probabilities(stacked, mask)
        # half-precision scores round the bounded tie-breaker away, so score in at least float32
        stacked = stacked.to(torch.promote_types(stacked.dtype, torch.float32))
        choice = torch.nan_to_num(stacked, nan=-1.0).argmax(dim=-1)
        votes = (torch.nn.functional.one_hot(choice, num_classes).to(stacked.dtype) * mask.unsqueeze(-1)).sum(dim=0)
        num_branches = mask.sum(dim=0).to(stacked.dtype).unsqueeze(-1)
        if self.tie_break == "error":
            top = votes.max(dim=-1, keepdim=True).values
            # rows without branches have no votes to tie; they reach here only with validation disabled
            tied = ((votes == top).sum(dim=-1) > 1) & (num_branches.squeeze(-1) > 0)
            if bool(tied.any()):
                raise ValueError(f"{int(tied.sum())} rows have tied votes and tie_break='error'")
        return (votes + 0.5 * masked_mean(stacked, mask)) / (num_branches + 0.5)


class Concat(nn.Module):
    """Concat is a simple PyTorch module for fusing multimodal data by concatenating tensors along dimension 1.
    This fusion method is often used in multimodal learning where data from different modalities (e.g., image, audio)
    are processed separately and then fused together for further processing or decision-making. Each modality data is
    first flattened from its second dimension onward and then these flattened tensors are concatenated together.
    This approach to fusion maintains the independence of the modalities before the fusion point, allowing the network
    to learn separate representations for each modality before combining them.
    """

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


class BimodalInteractionFusion(nn.Module):
    """BimodalInteractionFusion is a PyTorch module that performs fusion of two data modalities through a
    hypernetwork-based interaction mechanism. The 'input_dims' argument specifies the input dimensions of the two
    modalities. The 'output_dim' argument specifies the output dimension after the fusion. The 'output' argument
    defines the type of bimodal matrix interactions to be performed, which can be 'matrix', 'vector', or 'scalar'.
        This fusion method supports three types of bimodal interactions:
            - Matrix: It implements a general hypernetwork mechanism where the interaction is multiplicative. It uses
            separate weight matrices and biases for the two modalities.
            - Vector: It uses diagonal forms and gating mechanisms, applying element-wise multiplication to combine the
            modalities.
            - Scalar: It applies scales and biases to the input modalities before combining them.
        This fusion method uses xavier normal distribution for initializing the weight matrices and normal distribution
        for the biases. It also provides options to clip the parameter values and their gradients within specified
        ranges to prevent them from exploding or vanishing.
        This fusion approach allows for complex interactions between the modalities and is well-suited for tasks that
        require the integration of heterogeneous data.
    Args:
        input_dims (list or tuple): list or tuple of 2 integers indicating input dimensions of the 2 modalities
        output_dim (int): output dimension after the fusion
        output (str): type of BimodalMatrix Interactions, options from 'matrix','vector','scalar'
        flatten (bool): whether we need to flatten the input modalities
        clip (tuple, optional): clip parameter values, None if no clip
        grad_clip (tuple, optional): clip grad values, None if no clip
        flip (bool): whether to swap the two input modalities in forward function or not
    """

    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        super(BimodalInteractionFusion, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == "matrix":
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        elif output == "vector":
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        elif output == "scalar":
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, grad_clip[0], grad_clip[1]))

    def _repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == "matrix":
            Wprime = torch.einsum("bn, nmd -> bmd", m1, self.W) + self.V  # bmd
            bprime = torch.matmul(m1, self.U) + self.b  # bmd
            output = torch.einsum("bm, bmd -> bd", m2, Wprime) + bprime  # bmd

        elif self.output == "vector":
            Wprime = torch.matmul(m1, self.W) + self.V  # bm
            bprime = torch.matmul(m1, self.U) + self.b  # b
            output = Wprime * m2 + bprime  # bm

        elif self.output == "scalar":
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self._repeatHorizontally(Wprime, self.input_dims[1]) * m2 + self._repeatHorizontally(
                bprime, self.input_dims[1]
            )
        return output


class LowRankTensorFusion(nn.Module):
    """LowRankTensorFusion is a PyTorch module that performs multimodal fusion using a low-rank tensor-based approach.
       The 'input_dims' argument specifies the input dimensions of each modality. The 'output_dim' argument defines the
       output dimension after the fusion. The 'rank' argument is a hyperparameter specifying the rank for the low-rank
       tensor decomposition.
       This fusion method performs fusion by assuming a low-rank structure for the interaction tensor, effectively
       compressing the interaction space. It leverages a set of low-rank factors, one for each modality, that are
       learned during training.
       These factors are initialized with xavier normal distribution and are applied to their corresponding modalities
       during the forward pass. A tensor product is computed across all modalities and their respective factors,
       resulting in a fused tensor.
       Next, a weighted summation of this fused tensor is computed using fusion weights, followed by the addition of a
       fusion bias. Both fusion weights and bias are learnable parameters initialized with xavier normal distribution
       and zero respectively.
       The final output is reshaped to the specified 'output_dim' and returned. If 'flatten' is set to True, each
       modality is first flattened before concatenation with a ones tensor and the subsequent multiplication with its
       factor.
       This approach provides an efficient and compact representation for capturing interactions among multiple
       modalities, making it suitable for tasks involving high-dimensional multimodal data.
    Args:
        input_dims (list or tuple): A list or tuple of integers indicating input dimensions of the modalities.
        output_dim (int): output dimension after the fusion.
        rank (int): A hyperparameter specifying the rank for the low-rank tensor decomposition.
        flatten (bool): Boolean to dictate if output should be flattened or not. Default: True

    Note:
        Adapted from https://github.com/Justin1904/Low-rank-Multimodal-Fusion.
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        super(LowRankTensorFusion, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(self.rank, input_dim + 1, self.output_dim)).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        batch_size = modalities[0].shape[0]
        fused_tensor = 1
        for modality, factor in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            modality = modality.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            factor = factor.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            if self.flatten:
                modality_withones = torch.cat((ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class ProductOfExperts(nn.Module):
    """Combine diagonal Gaussian experts by precision-weighted multiplication.

    ``mean`` and ``log_var`` have shape ``(M, B, D)`` for experts, batch samples, and latent
    dimensions; the result has shape ``(B, D)``. The optional ``mask`` has shape ``(M, B)``
    with boolean or zero/one values, where one means present. Different samples in a batch
    may have different masks. An absent expert receives mean zero and log-variance 20, giving
    it negligible finite precision and zero mean gradient.

    No prior is added automatically. An all-missing sample is finite but highly uncertain:
    with ``M`` absent experts and the default ``eps``, its log-variance is approximately
    ``20 - log(M)``. To fall back to the standard-normal prior instead, prepend an
    always-present expert with zero mean and log-variance, as in the second example.

    Masked inputs are computed in at least float32, because ``exp(20)`` exceeds the float16
    range, and are returned in the input dtype. Calls without ``mask`` keep the existing
    tuple result and third positional ``eps`` argument.

    Example:

        .. code-block:: pycon

            >>> mean = torch.tensor([[[1.0]], [[3.0]]])
            >>> log_var = torch.zeros_like(mean)
            >>> present = torch.tensor([[True], [False]])
            >>> ProductOfExperts()(mean, log_var, mask=present)
            (tensor([[1.]]), tensor([[0.]]))
            >>> prior = torch.zeros_like(mean[:1])
            >>> ProductOfExperts()(
            ...     torch.cat((prior, mean)),
            ...     torch.cat((prior, log_var)),
            ...     mask=torch.cat((torch.ones_like(present[:1]), present)),
            ... )
            (tensor([[0.5000]]), tensor([[-0.6931]]))
    """

    def forward(
        self, mean: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-8, *, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the product Gaussian as ``(mean, log_var)`` tensors of shape ``(B, D)``.

        Args:
            mean (Tensor): Expert means of shape ``(M, B, D)``.
            log_var (Tensor): Expert log-variances of the same shape.
            eps (float, optional): Numerical floor used by the existing unmasked calculation. Defaults to ``1e-8``.
            mask (Tensor, optional): Boolean or zero/one tensor of shape ``(M, B)``; one means present. Absent experts
                receive mean zero and finite log-variance 20, so they contribute negligible precision and zero mean
                gradient. An all-missing column is finite but highly uncertain, not a standard-normal prior. Masked
                inputs are computed in at least float32 and returned in the input dtype.

        Returns:
            tuple[Tensor, Tensor]: Product mean and log-variance, each of shape ``(B, D)``.

        Raises:
            ValueError: If the expert shapes, mask shape, or mask values are invalid.
            TypeError: If ``mask`` is not a tensor.
        """
        if mean.shape != log_var.shape or mean.ndim != 3 or mean.shape[0] == 0:
            raise ValueError("mean and log_var must have the same nonempty (experts, batch, latent_dim) shape")
        output_dtype = mean.dtype
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("mask must be a tensor of shape (experts, batch)")
            if mask.shape != mean.shape[:2]:
                raise ValueError(f"mask must have shape {tuple(mean.shape[:2])}, got {tuple(mask.shape)}")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("mask entries must be zero or one")
            present = mask.to(device=mean.device, dtype=torch.bool).unsqueeze(-1)
            mean = torch.where(present, mean, torch.zeros_like(mean))
            log_var = torch.where(present, log_var, torch.full_like(log_var, 20.0))
            # The absent-expert variance exp(20) exceeds the float16 range, so compute in at least float32.
            compute_dtype = torch.promote_types(output_dtype, torch.float32)
            mean = mean.to(compute_dtype)
            log_var = log_var.to(compute_dtype)

        var = torch.exp(log_var) + eps
        precision = 1.0 / (var + eps)
        prod_mean = torch.sum(mean * precision, dim=0) / torch.sum(precision, dim=0)
        prod_var = 1.0 / torch.sum(precision, dim=0)
        prod_log_var = torch.log(prod_var + eps)
        if mask is not None:
            return prod_mean.to(output_dtype), prod_log_var.to(output_dtype)
        return prod_mean, prod_log_var
