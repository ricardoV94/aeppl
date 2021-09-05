import abc
from functools import partial, singledispatch
from typing import Dict, List, Optional, Type, Union

import aesara.tensor as at
from aesara.gradient import jacobian
from aesara.graph.basic import Node, Variable, walk
from aesara.graph.features import AlreadyThere, Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import GlobalOptimizer, in2out, local_optimizer
from aesara.graph.utils import MetaType
from aesara.scalar.basic import Add, Exp, Log, Mul
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from aeppl.logprob import MeasurableRebroadcast, _logprob
from aeppl.utils import walk_model


@singledispatch
def _default_transformed_rv(
    op: Op,
    node: Node,
) -> Optional[TensorVariable]:
    """Create a graph for a transformed log-probability of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new transforms for a
    ``RandomVariable``, register a function on this dispatcher.

    """
    return None


class DistributionMeta(MetaType):
    def __new__(cls, name, bases, clsdict):
        cls_res = super().__new__(cls, name, bases, clsdict)

        base_op = clsdict.get("base_op", None)
        default = clsdict.get("default", False)
        forward = clsdict.get("forward", False)

        if base_op is not None and not forward and default:
            # Create dispatch functions
            @_default_transformed_rv.register(type(base_op))
            def class_transformed_rv(op, node):
                new_op = cls_res()
                res = new_op.make_node(*node.inputs)
                res.outputs[1].name = node.outputs[1].name
                return res

        return cls_res


class RVTransform(abc.ABC):
    @abc.abstractmethod
    def forward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Apply the transformation."""

    @abc.abstractmethod
    def backward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Invert the transformation."""

    def log_jac_det(self, value: TensorVariable, *inputs) -> TensorVariable:
        """Construct the log of the absolute value of the Jacobian determinant."""
        # jac = at.reshape(
        #     gradient(at.sum(self.backward(value, *inputs)), [value]), value.shape
        # )
        # return at.log(at.abs_(jac))
        phi_inv = self.backward(value, *inputs)
        return at.log(at.nlinalg.det(at.atleast_2d(jacobian(phi_inv, [value]))))

    def filter_non_rv_inputs(self, *inputs):
        """
        Filter inputs that were added to the TransformedRV (if any)

        This is needed because some transformations require graph inputs that
        are not necessarily part of the base random variable inputs (e.g., loc
        and scale transforms)
        """
        return inputs


class TransformedRV(RandomVariable, metaclass=DistributionMeta):
    r"""A base class for transformed `RandomVariable`\s."""


class ForwardTransformedRV(TransformedRV):
    r"""A base class for forward-transformed `RandomVariable`\s."""


@_logprob.register(TransformedRV)
def transformed_logprob(op, values, *inputs, **kwargs):
    """Compute the log-likelihood graph for a `TransformedRV`.

    We assume that the value variable was back-transformed to be on the natural
    support of the respective random variable.
    """
    (value,) = values

    rv_inputs = op.transform.filter_non_rv_inputs(*inputs)
    logprob = _logprob(op.base_op, (value,), *rv_inputs, **kwargs)

    original_forward_value = op.transform.forward(value, *inputs)
    jacobian = op.transform.log_jac_det(original_forward_value, *inputs)

    logprob.name = "logprob"
    jacobian.name = "logprob_jac"

    return logprob + jacobian


@_logprob.register(ForwardTransformedRV)
def forward_transformed_logprob(op, values, *inputs, **kwargs):
    """
    Compute logp graph for a value variable that must still be back-transformed
    to be on the natural support of the respective random variable.
    """
    (value,) = values

    backward_value = op.transform.backward(value, *inputs)
    rv_inputs = op.transform.filter_non_rv_inputs(*inputs)
    logprob = _logprob(op.base_op, (backward_value,), *rv_inputs, **kwargs)

    jacobian = op.transform.log_jac_det(value, *inputs)

    logprob.name = "logprob"
    jacobian.name = "logprob_jac"

    return logprob + jacobian


class DefaultTransformSentinel:
    pass


DEFAULT_TRANSFORM = DefaultTransformSentinel()


@local_optimizer(tracks=None)
def transform_values(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:
    """Apply transforms to value variables.

    It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are
    unconstrained on the real line.

    For example, if ``Y = halfnormal(...)``, we assume the respective value
    variable is specified on the log scale and back-transform it to obtain
    ``Y`` on the natural scale.
    """

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    values_to_transforms = getattr(fgraph, "values_to_transforms", None)

    if rv_map_feature is None or values_to_transforms is None:
        return None  # pragma: no cover

    try:
        rv_var = node.default_output()
    except ValueError:
        return None

    value_var = rv_map_feature.rv_values.get(rv_var, None)
    if value_var is None:
        return None

    transform = values_to_transforms.get(value_var, None)

    if transform is None:
        return None
    elif transform is DEFAULT_TRANSFORM:
        trans_node = _default_transformed_rv(node.op, node)
        if trans_node is None:
            return None
        transform = trans_node.op.transform
    else:
        new_op = _create_transformed_rv_op(node.op, transform)()
        trans_node = new_op.make_node(*node.inputs)
        trans_node.outputs[1].name = node.outputs[1].name

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    new_value_var = transform.backward(value_var, *trans_node.inputs)
    if value_var.name and getattr(transform, "name", None):
        new_value_var.name = f"{value_var.name}_{transform.name}"

    # Map TransformedRV to new value var and delete old mapping
    new_rv_var = trans_node.outputs[1]
    rv_map_feature.rv_values[new_rv_var] = new_value_var
    del rv_map_feature.rv_values[rv_var]

    return trans_node.outputs


class TransformValuesMapping(Feature):
    r"""A `Feature` that maintains a map between value variables and their transforms."""

    def __init__(self, values_to_transforms):
        self.values_to_transforms = values_to_transforms

    def on_attach(self, fgraph):
        if hasattr(fgraph, "values_to_transforms"):
            raise AlreadyThere()

        fgraph.values_to_transforms = self.values_to_transforms


class TransformValuesOpt(GlobalOptimizer):
    r"""Transforms value variables according to a map and/or per-`RandomVariable` defaults."""

    default_transform_opt = in2out(transform_values, ignore_newtrees=True)

    def __init__(
        self,
        values_to_transforms: Dict[
            TensorVariable, Union[RVTransform, DefaultTransformSentinel, None]
        ],
    ):
        """
        Parameters
        ==========
        values_to_transforms
            Mapping between value variables and their transformations.  Each
            value variable can be assigned one of `RVTransform`,
            ``DEFAULT_TRANSFORM``, or ``None``. If a transform is not specified
            for a specific value variable it will not be transformed.

        """

        self.values_to_transforms = values_to_transforms

    def add_requirements(self, fgraph):
        values_transforms_feature = TransformValuesMapping(self.values_to_transforms)
        fgraph.attach_feature(values_transforms_feature)

    def apply(self, fgraph: FunctionGraph):
        return self.default_transform_opt.optimize(fgraph)


def _get_single_vars_between_in_out(
    input_var: TensorVariable, output_var: TensorVariable, fgraph: FunctionGraph
) -> List[TensorVariable]:
    """
    Return ordered list of consecutive variables between input_var and output_var
    (both included).

    Search is done by navigating through fgraph clients from input to output. An
    empty list is returned if the output is not found downstream of the input
    or if any of the intermediate variables contains more than one client.

    In the edge case where the input and output variables are the same, the
    returned list includes the variable only once.

    Example
    =======
    a, b, c, d, e, f = at.scalars('abcdef')
    ab = a + b
    cd = c + d
    abcd = ab + cd
    ef = e + f
    y = abcd + ef

    get_single_vars_between_input_output(c, y) -> [(c, cd, abcd, y)]

    """

    def expand(node):
        clients = fgraph.clients[node]
        if len(clients) != 1 or clients[0][0] == "output":
            # This prevents the output var from being reached whenever there is
            # more than one client
            return []
        return [output for output in clients[0][0].outputs]

    pair_nodes = []
    for node in walk([input_var], expand=expand):
        pair_nodes.append(node)
        if node is output_var:
            break
    else:  # nobreak, output_var was not found
        return []

    return pair_nodes


@local_optimizer([Elemwise])
def chained_transform_finder(
    fgraph: FunctionGraph,
    node: Node,
) -> Optional[List[Union[ForwardTransformedRV, MeasurableRebroadcast]]]:
    """
    Find value vars that correspond to simple (potentially nested) transformations
    of base random variables, such as:
        Y = at.exp(at.Normal(0, 1))
        Y = a + at.Normal(0, 1) * b
    """

    if not isinstance(node.op, Elemwise):
        return None

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    out_var = node.outputs[0]
    if out_var not in rv_map_feature.rv_values:
        return None

    implemented_transform_ops = (Add, Mul, Exp, Log)
    if not isinstance(node.op.scalar_op, implemented_transform_ops):
        return None  # pragma: no cover

    # Check for valid base RV upstream of this node. This should only return
    # other nodes that correspond to RVs without associated value_vars
    ancestor_base_rvs = [
        ancestor_node
        for ancestor_node in walk_model(
            node.inputs,
            walk_past_rvs=False,
            stop_at_vars=rv_map_feature.rv_values,
        )
        if (
            ancestor_node.owner
            and isinstance(ancestor_node.owner.op, RandomVariable)
            and ancestor_node not in rv_map_feature.rv_values
            and not getattr(ancestor_node.tag, "ignore_logprob", False)
        )
    ]

    if len(ancestor_base_rvs) != 1:
        return None

    base_rv = ancestor_base_rvs[0]

    # TODO: We might want to support discrete transforms in the future
    if base_rv.dtype.startswith("int"):
        return None

    # Collect unique chain of nodes between base_rv and out_var (the derived RV)
    rv_chain_nodes = [
        var.owner
        for var in _get_single_vars_between_in_out(base_rv, out_var, fgraph)[1:]
    ]
    if not rv_chain_nodes:
        return None  # pragma: no cover

    # Exit quickly if any intermediate op is not implemented
    if any(
        node
        for node in rv_chain_nodes
        if (
            not (
                isinstance(node.op, Elemwise)
                and isinstance(node.op.scalar_op, implemented_transform_ops)
            )
        )
    ):
        return None  # pragma: no cover

    # Create list of RV transforms that link the base_rv support to that of the
    # value var / derived RV. If Y = at.exp(base_rv * 5), this should be
    # [ScaleTransform(scale=5), ExpTransform()]
    transform_inputs = base_rv.owner.inputs
    chain_transform_list = []
    for transform_node in rv_chain_nodes:
        transform_node_op = transform_node.op.scalar_op

        transform: RVTransform
        if isinstance(transform_node_op, (Add, Mul)):
            # Find non-rv branches
            other_branches = [
                inp
                for inp in transform_node.inputs
                if (inp.owner not in rv_chain_nodes and inp is not base_rv)
            ]

            if isinstance(transform_node_op, Add):
                new_input = (
                    at.add(other_branches)
                    if len(other_branches) > 1
                    else other_branches[0]
                )
                transform = LocTransform(
                    transform_args_fn=lambda *inputs: inputs[-1],
                    filter_non_rv_inputs_fn=lambda *inputs: inputs[:-1],
                )
            else:
                new_input = (
                    at.mul(other_branches)
                    if len(other_branches) > 1
                    else other_branches[0]
                )
                transform = ScaleTransform(
                    transform_args_fn=lambda *inputs: inputs[-1],
                    filter_non_rv_inputs_fn=lambda *inputs: inputs[:-1],
                )
            transform_inputs.append(new_input)

        elif isinstance(transform_node_op, Exp):
            transform = ExpTransform()
        elif isinstance(transform_node_op, Log):
            transform = LogTransform()

        chain_transform_list.append(transform)

    # Create Transformed RV
    if len(chain_transform_list) == 1:
        transform_class = chain_transform_list[0]
    else:
        transform_class = ChainedTransform(
            transform_list=chain_transform_list,
            base_op=base_rv.owner.op,
        )

    transform_op = _create_transformed_rv_op(
        base_rv.owner.op,
        transform_class,
        forward=True,
    )()

    transform_node = transform_op.make_node(*transform_inputs)
    transform_rv_out = transform_node.default_output()
    if out_var.name:
        transform_rv_out.name = out_var.name

    # Tag base_rv as having been used
    base_rv.tag.ignore_logprob = True

    # Change broadcasting pattern of TransformedRV if needed
    if transform_rv_out.broadcastable != out_var.broadcastable:
        transform_rv_out.tag.ignore_logprob = True
        transform_rv_out = MeasurableRebroadcast(
            *[(i, b) for i, b in enumerate(out_var.broadcastable)]
        )(transform_rv_out)

    # Link value var to new transformed RV
    value_var = rv_map_feature.rv_values.pop(out_var)
    rv_map_feature.rv_values[transform_rv_out] = value_var

    return [transform_rv_out]


class LocTransform(RVTransform):
    name = "loc"

    def __init__(self, transform_args_fn, filter_non_rv_inputs_fn):
        self.transform_args_fn = transform_args_fn
        self.filter_non_rv_inputs_fn = filter_non_rv_inputs_fn

    def forward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value + loc

    def backward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value - loc

    def log_jac_det(self, value, *inputs):
        return at.zeros_like(value)

    def filter_non_rv_inputs(self, *inputs):
        return self.filter_non_rv_inputs_fn(*inputs)


class ScaleTransform(RVTransform):
    name = "scale"

    def __init__(self, transform_args_fn, filter_non_rv_inputs_fn):
        self.transform_args_fn = transform_args_fn
        self.filter_non_rv_inputs_fn = filter_non_rv_inputs_fn

    def forward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value * scale

    def backward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value / scale

    def log_jac_det(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return -at.log(at.abs(scale))

    def filter_non_rv_inputs(self, *inputs):
        return self.filter_non_rv_inputs_fn(*inputs)


class LogTransform(RVTransform):
    name = "log"

    def forward(self, value, *inputs):
        return at.log(value)

    def backward(self, value, *inputs):
        return at.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class ExpTransform(RVTransform):
    name = "exp"

    def forward(self, value, *inputs):
        return at.exp(value)

    def backward(self, value, *inputs):
        return at.log(value)

    def log_jac_det(self, value, *inputs):
        return -at.log(value)


class IntervalTransform(RVTransform):
    name = "interval"

    def __init__(self, args_fn):
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return at.log(value - a) - at.log(b - value)
        elif a is not None:
            return at.log(value - a)
        elif b is not None:
            return at.log(b - value)

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = at.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return at.exp(value) + a
        elif b is not None:
            return b - at.exp(value)

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = at.softplus(-value)
            return at.log(b - a) - 2 * s - value
        else:
            return value


class LogOddsTransform(RVTransform):
    name = "logodds"

    def backward(self, value, *inputs):
        return at.expit(value)

    def forward(self, value, *inputs):
        return at.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = at.sigmoid(value)
        return at.log(sigmoid_value) + at.log1p(-sigmoid_value)


class StickBreaking(RVTransform):
    name = "stickbreaking"

    def forward(self, value, *inputs):
        log_value = at.log(value)
        shift = at.sum(log_value, -1, keepdims=True) / value.shape[-1]
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = at.concatenate([value, -at.sum(value, -1, keepdims=True)])
        exp_value_max = at.exp(value - at.max(value, -1, keepdims=True))
        return exp_value_max / at.sum(exp_value_max, -1, keepdims=True)

    def log_jac_det(self, value, *inputs):
        N = value.shape[-1] + 1
        sum_value = at.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = at.concatenate(
            [value_sum_expanded, at.zeros(sum_value.shape)], -1
        )
        logsumexp_value_expanded = at.logsumexp(value_sum_expanded, -1, keepdims=True)
        res = at.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return at.sum(res, -1)


class CircularTransform(RVTransform):
    name = "circular"

    def backward(self, value, *inputs):
        return at.arctan2(at.sin(value), at.cos(value))

    def forward(self, value, *inputs):
        return at.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return at.zeros(value.shape)


class ChainedTransform(RVTransform):
    name = "chain"

    def __init__(self, transform_list, base_op):
        self.transform_list = transform_list
        self.base_op = base_op

    def forward(self, value, *inputs):
        # Flip extra inputs (if any) for correct filtering during forward pass
        n_base_inputs = len(self.base_op.ndims_params) + 3
        inputs = (inputs[:n_base_inputs], *reversed(inputs[n_base_inputs:]))
        for transform in self.transform_list:
            value = transform.forward(value, *inputs)
            inputs = transform.filter_non_rv_inputs(*inputs)
        return value

    def backward(self, value, *inputs):
        for transform in reversed(self.transform_list):
            value = transform.backward(value, *inputs)
            inputs = transform.filter_non_rv_inputs(*inputs)
        return value

    def log_jac_det(self, value, *inputs):
        value = at.as_tensor_variable(value)
        det_list = []
        ndim0 = value.ndim
        for transform in reversed(self.transform_list):
            det_ = transform.log_jac_det(value, *inputs)
            det_list.append(det_)
            ndim0 = min(ndim0, det_.ndim)
            value = transform.backward(value, *inputs)
            inputs = transform.filter_non_rv_inputs(*inputs)
        # match the shape of the smallest jacobian_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det

    def filter_non_rv_inputs(self, *inputs):
        for transform in self.transform_list:
            inputs = transform.filter_non_rv_inputs(*inputs)
        return inputs


def _create_transformed_rv_op(
    rv_op: Op,
    transform: RVTransform,
    *,
    default: bool = False,
    forward: bool = False,
    cls_dict_extra: Optional[Dict] = None,
) -> Type[TransformedRV]:
    """Create a new `TransformedRV` given a base `RandomVariable` `Op`

    Parameters
    ==========
    rv_op
        The `RandomVariable` for which we want to construct a `TransformedRV`.
    transform
        The `RVTransform` for `rv_op`.
    default
        If ``False`` do not make `transform` the default transform for `rv_op`.
    forward
        If ``True`` the returned op will be a inherit from `ForwardTransformRV`,
        instead of `TransformedRV`. This is used for an RV whose value variable
        has to be back-transformed to be on the same support of the RV.
    cls_dict_extra
        Additional class members to add to the constructed `TransformedRV`.

    """

    trans_name = getattr(transform, "name", "transformed")
    rv_type_name = type(rv_op).__name__
    cls_dict = type(rv_op).__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = f"{rv_name}_{trans_name}"
    cls_dict["base_op"] = rv_op
    cls_dict["transform"] = transform
    cls_dict["default"] = default
    cls_dict["forward"] = forward

    if cls_dict_extra is not None:
        cls_dict.update(cls_dict_extra)

    super_class = ForwardTransformedRV if forward else TransformedRV
    new_op_type = type(f"Transformed{rv_type_name}", (super_class,), cls_dict)

    return new_op_type


create_default_transformed_rv_op = partial(
    _create_transformed_rv_op, default=True, forward=False
)


TransformedUniformRV = create_default_transformed_rv_op(
    at.random.uniform,
    # inputs[3] = lower; inputs[4] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[4])),
)
TransformedParetoRV = create_default_transformed_rv_op(
    at.random.pareto,
    # inputs[3] = alpha
    IntervalTransform(lambda *inputs: (inputs[3], None)),
)
TransformedTriangularRV = create_default_transformed_rv_op(
    at.random.triangular,
    # inputs[3] = lower; inputs[5] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[5])),
)
TransformedHalfNormalRV = create_default_transformed_rv_op(
    at.random.halfnormal,
    # inputs[3] = loc
    IntervalTransform(lambda *inputs: (inputs[3], None)),
)
TransformedWaldRV = create_default_transformed_rv_op(
    at.random.wald,
    LogTransform(),
)
TransformedExponentialRV = create_default_transformed_rv_op(
    at.random.exponential,
    LogTransform(),
)
TransformedLognormalRV = create_default_transformed_rv_op(
    at.random.lognormal,
    LogTransform(),
)
TransformedHalfCauchyRV = create_default_transformed_rv_op(
    at.random.halfcauchy,
    LogTransform(),
)
TransformedGammaRV = create_default_transformed_rv_op(
    at.random.gamma,
    LogTransform(),
)
TransformedInvGammaRV = create_default_transformed_rv_op(
    at.random.invgamma,
    LogTransform(),
)
TransformedChiSquareRV = create_default_transformed_rv_op(
    at.random.chisquare,
    LogTransform(),
)
TransformedWeibullRV = create_default_transformed_rv_op(
    at.random.weibull,
    LogTransform(),
)
TransformedBetaRV = create_default_transformed_rv_op(
    at.random.beta,
    LogOddsTransform(),
)
TransformedVonMisesRV = create_default_transformed_rv_op(
    at.random.vonmises,
    CircularTransform(),
)
TransformedDirichletRV = create_default_transformed_rv_op(
    at.random.dirichlet,
    StickBreaking(),
)
