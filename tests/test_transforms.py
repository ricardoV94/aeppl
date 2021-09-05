import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
from aesara.graph.fg import FunctionGraph
from numdifftools import Jacobian

from aeppl.joint_logprob import joint_logprob
from aeppl.transforms import (
    DEFAULT_TRANSFORM,
    ChainedTransform,
    ExpTransform,
    LocTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    ScaleTransform,
    TransformValuesMapping,
    TransformValuesOpt,
    _default_transformed_rv,
    _get_single_vars_between_in_out,
)
from tests.utils import assert_no_rvs


@pytest.mark.parametrize(
    "at_dist, dist_params, sp_dist, size",
    [
        (at.random.uniform, (0, 1), sp.stats.uniform, ()),
        (
            at.random.pareto,
            (1.5, 10.5),
            lambda b, scale: sp.stats.pareto(b, scale=scale),
            (),
        ),
        (
            at.random.triangular,
            (1.5, 3.0, 10.5),
            lambda lower, mode, upper: sp.stats.triang(
                (mode - lower) / (upper - lower), loc=lower, scale=upper - lower
            ),
            (),
        ),
        (
            at.random.halfnormal,
            (0, 1),
            sp.stats.halfnorm,
            (),
        ),
        (
            at.random.wald,
            (1.5, 10.5),
            lambda mean, scale: sp.stats.invgauss(mean / scale, scale=scale),
            (),
        ),
        (
            at.random.exponential,
            (1.5,),
            lambda mu: sp.stats.expon(scale=mu),
            (),
        ),
        pytest.param(
            at.random.lognormal,
            (-1.5, 10.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu)),
            (),
        ),
        (
            at.random.lognormal,
            (-1.5, 1.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, scale=np.exp(mu)),
            (),
        ),
        (
            at.random.halfcauchy,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.halfcauchy(loc=alpha, scale=beta),
            (),
        ),
        (
            at.random.gamma,
            (1.5, 10.5),
            lambda alpha, inv_beta: sp.stats.gamma(alpha, scale=1.0 / inv_beta),
            (),
        ),
        (
            at.random.invgamma,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.invgamma(alpha, scale=beta),
            (),
        ),
        (
            at.random.chisquare,
            (1.5,),
            lambda df: sp.stats.chi2(df),
            (),
        ),
        (
            at.random.weibull,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.weibull_min(alpha, scale=beta),
            (),
        ),
        (
            at.random.beta,
            (1.5, 1.5),
            lambda alpha, beta: sp.stats.beta(alpha, beta),
            (),
        ),
        (
            at.random.vonmises,
            (1.5, 10.5),
            lambda mu, kappa: sp.stats.vonmises(kappa, loc=mu),
            (),
        ),
        (
            at.random.dirichlet,
            (np.array([0.5, 0.5]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (),
        ),
        pytest.param(
            at.random.dirichlet,
            (np.array([0.5, 0.5]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (3, 2),
            marks=pytest.mark.xfail(
                reason="Need to make the test framework work for arbitrary sizes"
            ),
        ),
    ],
)
def test_transformed_logprob(at_dist, dist_params, sp_dist, size):
    """
    This test takes a `RandomVariable` type, plus parameters, and uses it to
    construct a variable ``a`` that's used in the graph ``b =
    at.random.normal(a, 1.0)``.  The transformed log-probability is then
    computed for ``b``.  We then test that the log-probability of ``a`` is
    properly transformed, as well as any instances of ``a`` that are used
    elsewhere in the graph (i.e. in ``b``), by comparing the graph for the
    transformed log-probability with the SciPy-derived log-probability--using a
    numeric approximation to the Jacobian term.
    """

    a = at_dist(*dist_params, size=size)
    a.name = "a"
    a_value_var = a.clone()
    a_value_var.name = "a_value"

    b = at.random.normal(a, 1.0)
    b.name = "b"
    b_value_var = b.clone()
    b_value_var.name = "b_value"

    transform_opt = TransformValuesOpt({a_value_var: DEFAULT_TRANSFORM})
    res = joint_logprob({a: a_value_var, b: b_value_var}, extra_rewrites=transform_opt)

    test_val_rng = np.random.RandomState(3238)

    decimals = 6 if aesara.config.floatX == "float64" else 4
    logp_vals_fn = aesara.function([a_value_var, b_value_var], res)

    a_trans_op = _default_transformed_rv(a.owner.op, a.owner).op
    transform = a_trans_op.transform

    a_forward_fn = aesara.function(
        [a_value_var], transform.forward(a_value_var, *a.owner.inputs)
    )
    a_backward_fn = aesara.function(
        [a_value_var], transform.backward(a_value_var, *a.owner.inputs)
    )

    for i in range(10):
        a_dist = sp_dist(*dist_params)
        a_val = a_dist.rvs(size=size, random_state=test_val_rng).astype(
            a_value_var.dtype
        )
        b_dist = sp.stats.norm(a_val, 1.0)
        b_val = b_dist.rvs(random_state=test_val_rng).astype(b_value_var.dtype)

        exp_logprob_val = a_dist.logpdf(a_val)

        a_trans_value = a_forward_fn(a_val)
        if a_val.ndim > 0:
            # exp_logprob_val = np.vectorize(a_dist.logpdf, signature="(n)->()")(a_val)
            jacobian_val = Jacobian(a_backward_fn)(a_trans_value)[:-1]
        else:
            jacobian_val = np.atleast_2d(
                sp.misc.derivative(a_backward_fn, a_trans_value, dx=1e-6)
            )

        exp_logprob_val += np.log(np.linalg.det(jacobian_val))
        exp_logprob_val += b_dist.logpdf(b_val).sum()

        logprob_val = logp_vals_fn(a_trans_value, b_val)

        np.testing.assert_almost_equal(exp_logprob_val, logprob_val, decimal=decimals)


def test_simple_transformed_logprob():
    x_rv = at.random.halfnormal(0, 3, name="x_rv")
    x = x_rv.clone()

    transform_opt = TransformValuesOpt({x: DEFAULT_TRANSFORM})
    tr_logp = joint_logprob({x_rv: x}, extra_rewrites=transform_opt)

    assert np.isclose(
        tr_logp.eval({x: np.log(2.5)}),
        sp.stats.halfnorm(0, 3).logpdf(2.5) + np.log(2.5),
    )


def test_fallback_log_jac_det():
    """
    Test fallback log_jac_det in RVTransform produces correct the graph for a
    simple transformation: x**2 -> -log(2*x)
    """

    class SquareTransform(RVTransform):
        name = "square"

        def forward(self, value, *inputs):
            return at.power(value, 2)

        def backward(self, value, *inputs):
            return at.sqrt(value)

    square_tr = SquareTransform()

    value = at.scalar("value")
    value_tr = square_tr.forward(value)
    log_jac_det = square_tr.log_jac_det(value_tr)

    assert np.isclose(log_jac_det.eval({value: 3}), -np.log(6))


def test_hierarchical_uniform_transform():
    """
    This model requires rv-value replacements in the backward transformation of
    the value var `x`
    """

    lower_rv = at.random.uniform(0, 1, name="lower")
    upper_rv = at.random.uniform(9, 10, name="upper")
    x_rv = at.random.uniform(lower_rv, upper_rv, name="x")

    lower = lower_rv.clone()
    upper = upper_rv.clone()
    x = x_rv.clone()

    transform_opt = TransformValuesOpt(
        {
            lower: DEFAULT_TRANSFORM,
            upper: DEFAULT_TRANSFORM,
            x: DEFAULT_TRANSFORM,
        }
    )
    logp = joint_logprob(
        {lower_rv: lower, upper_rv: upper, x_rv: x},
        extra_rewrites=transform_opt,
    )

    assert_no_rvs(logp)
    assert not np.isinf(logp.eval({lower: -10, upper: 20, x: -20}))


def test_nondefault_transforms():
    loc_rv = at.random.uniform(-10, 10, name="loc")
    scale_rv = at.random.uniform(-1, 1, name="scale")
    x_rv = at.random.normal(loc_rv, scale_rv, name="x")

    loc = loc_rv.clone()
    scale = scale_rv.clone()
    x = x_rv.clone()

    transform_opt = TransformValuesOpt(
        {
            loc: None,
            scale: LogOddsTransform(),
            x: LogTransform(),
        }
    )

    logp = joint_logprob(
        {loc_rv: loc, scale_rv: scale, x_rv: x},
        extra_rewrites=transform_opt,
    )

    # Check numerical evaluation matches with expected transforms
    loc_val = 0
    scale_val_tr = -1
    x_val_tr = -1

    scale_val = sp.special.expit(scale_val_tr)
    x_val = np.exp(x_val_tr)

    exp_logp = 0
    exp_logp += sp.stats.uniform(-10, 20).logpdf(loc_val)
    exp_logp += sp.stats.uniform(-1, 2).logpdf(scale_val)
    exp_logp += np.log(scale_val) + np.log1p(-scale_val)  # logodds log_jac_det
    exp_logp += sp.stats.norm(loc_val, scale_val).logpdf(x_val)
    exp_logp += x_val_tr  # log log_jac_det

    assert np.isclose(
        logp.eval({loc: loc_val, scale: scale_val_tr, x: x_val_tr}),
        exp_logp,
    )


def test_default_transform_multiout():
    r"""Make sure that `Op`\s with multiple outputs are handled correctly."""

    # This SVD value is necessarily `1`, but it's generated by an `Op` with
    # multiple outputs and no default output.
    sd = at.linalg.svd(at.eye(1))[1][0]
    x_rv = at.random.normal(0, sd, name="x")
    x = x_rv.clone()

    transform_opt = TransformValuesOpt({x: DEFAULT_TRANSFORM})

    logp = joint_logprob(
        {x_rv: x},
        extra_rewrites=transform_opt,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


def test_nonexistent_default_transform():
    """
    Test that setting `DEFAULT_TRANSFORM` to a variable that has no default
    transform does not fail
    """
    x_rv = at.random.normal(name="x")
    x = x_rv.clone()

    transform_opt = TransformValuesOpt({x: DEFAULT_TRANSFORM})

    logp = joint_logprob(
        {x_rv: x},
        extra_rewrites=transform_opt,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


def test_TransformValuesMapping():
    x = at.vector()
    fg = FunctionGraph(outputs=[x])

    tvm = TransformValuesMapping({})
    fg.attach_feature(tvm)

    tvm2 = TransformValuesMapping({})
    fg.attach_feature(tvm2)

    assert fg._features[-1] is tvm


def test_chained_transform():
    ch = ChainedTransform(
        transform_list=[
            ScaleTransform(
                transform_args_fn=lambda *inputs: inputs[-1],
                filter_non_rv_inputs_fn=lambda *inputs: inputs[:-1],
            ),
            ExpTransform(),
            LocTransform(
                transform_args_fn=lambda *inputs: inputs[-1],
                filter_non_rv_inputs_fn=lambda *inputs: inputs[:-1],
            ),
        ],
        base_op=at.random.multivariate_normal,
    )

    x = at.random.multivariate_normal(np.zeros(3), np.eye(3))
    x_val = x.eval()

    scale = 0.1
    loc = 5

    x_val_forward = ch.forward(x_val, *x.owner.inputs, scale, loc).eval()
    assert np.allclose(
        x_val_forward,
        np.exp(x_val * scale) + loc,
    )

    x_val_backward = ch.backward(x_val_forward, *x.owner.inputs, scale, loc).eval()
    assert np.allclose(
        x_val_backward,
        x_val,
    )

    log_jac_det = ch.log_jac_det(x_val_forward, *x.owner.inputs, scale, loc)
    assert np.isclose(
        log_jac_det.eval(),
        -np.log(scale) - np.sum(np.log(x_val_forward - loc)),
    )


@pytest.mark.parametrize(
    "rv_size, loc_type",
    [
        (None, at.scalar),
        (1, at.TensorType("floatX", (True,))),
        (2, at.vector),
        ((3, 2), at.matrix),
    ],
)
def test_loc_transform(rv_size, loc_type):

    loc = loc_type("loc")

    y_rv = loc + at.random.normal(0, 1, size=rv_size, name="base_rv")
    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y}, sum=False)
    assert_no_rvs(logp)
    logp_fn = aesara.function([loc, y], logp)

    loc_test_val = np.full(rv_size, 4.0)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, y_test_val),
        sp.stats.norm(loc_test_val, 1).logpdf(y_test_val),
    )


@pytest.mark.parametrize(
    "rv_size, scale_type",
    [
        (None, at.scalar),
        (1, at.TensorType("floatX", (True,))),
        (2, at.vector),
        ((3, 1), at.col),
    ],
)
def test_scale_transform(rv_size, scale_type):

    scale = scale_type("scale")

    y_rv = at.random.normal(0, 1, size=rv_size, name="base_rv") * scale
    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y}, sum=False)
    assert_no_rvs(logp)
    logp_fn = aesara.function([scale, y], logp)

    scale_test_val = np.full(rv_size, 4.0)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(scale_test_val, y_test_val),
        sp.stats.norm(0, scale_test_val).logpdf(y_test_val),
    )


@pytest.mark.parametrize(
    "rv_size, scale_type",
    [
        (None, at.scalar),
        (1, at.TensorType("floatX", (True,))),
        (2, at.vector),
        ((3, 1), at.col),
    ],
)
def test_affine_transform(rv_size, scale_type):
    loc = at.scalar("loc")
    scale = scale_type("scale")

    y_rv = loc + at.random.normal(0, 1, size=rv_size, name="base_rv") * scale
    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y}, sum=False)
    assert_no_rvs(logp)
    logp_fn = aesara.function([loc, scale, y], logp)

    loc_test_val = 4.0
    scale_test_val = np.full(rv_size, 0.5)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, scale_test_val, y_test_val),
        sp.stats.norm(loc_test_val, scale_test_val).logpdf(y_test_val),
    )


def test_double_loc_transform():
    loc1 = at.scalar("loc1")
    loc2 = at.scalar("loc2")
    y_rv = loc1 + loc2 + at.random.normal(0, 1, name="base_rv")
    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y})
    assert_no_rvs(logp)
    logp_fn = aesara.function([loc1, loc2, y], logp)

    loc1_test_val = 2.0
    loc2_test_val = 3.0
    y_test_val = 1.0

    assert np.isclose(
        logp_fn(loc1_test_val, loc1_test_val, y_test_val),
        sp.stats.norm(loc1_test_val + loc2_test_val, 1).logpdf(y_test_val),
    )


def test_double_scale_transform():
    scale1 = at.scalar("scale1")
    scale2 = at.scalar("scale2")
    y_rv = at.random.normal(0, 1, name="base_rv") * scale1 * scale2
    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y})
    assert_no_rvs(logp)
    logp_fn = aesara.function([scale1, scale2, y], logp)

    scale1_test_val = 2.0
    scale2_test_val = 3.0
    y_test_val = 1.0

    assert np.isclose(
        logp_fn(scale1_test_val, scale1_test_val, y_test_val),
        sp.stats.norm(0, scale1_test_val * scale2_test_val).logpdf(y_test_val),
    )


def test_exp_scale_transform():
    b = at.scalar("b")
    base_rv = at.random.normal(0, 1, size=2, name="base_rv")
    y_rv = at.exp(base_rv * b)
    y_rv.name = "y_rv"

    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y}, sum=False)
    logp_fn = aesara.function([b, y], logp)

    b_val = 1.5
    y_val = [0.1, 0.1]

    np.testing.assert_allclose(
        logp_fn(b_val, y_val),
        sp.stats.lognorm(b_val).logpdf(y_val),
    )


def test_affine_log_transform():
    a, b = at.scalars("a", "b")
    base_rv = at.random.lognormal(0, 1, name="base_rv", size=(1, 2))
    y_rv = a + at.log(base_rv) * b
    y_rv.name = "y_rv"

    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y}, sum=False)
    logp_fn = aesara.function([a, b, y], logp)

    a_val = -1.5
    b_val = 3.0
    y_val = [[0.1, 0.1]]

    assert np.allclose(
        logp_fn(a_val, b_val, y_val),
        sp.stats.norm(a_val, b_val).logpdf(y_val),
    )


def test_hierarchical_derived_model():
    scale_rv1 = at.random.halfnormal(0, 1, name="scale_rv1")
    derived_rv1 = at.random.normal(0, 1, name="base_rv1") * scale_rv1
    derived_rv1.name = "derived_rv1"

    scale_rv2 = at.random.exponential(1, name="scale_rv2")
    derived_rv2 = derived_rv1 + at.random.normal(0, 1, name="base_rv2") * scale_rv2
    derived_rv2.name = "derived_rv2"

    y_rv = at.random.normal(derived_rv2, 1, name="y_rv")

    scale1 = scale_rv1.clone()
    scale1.name = "scale1"
    derived1 = derived_rv1.clone()
    derived1.name = "derived1"

    scale2 = scale_rv2.clone()
    scale2.name = "scale2"
    derived2 = derived_rv2.clone()
    derived2.name = "derived2"

    y = y_rv.clone()
    y.name = "y"

    logp = joint_logprob(
        {
            scale_rv1: scale1,
            derived_rv1: derived1,
            scale_rv2: scale2,
            derived_rv2: derived2,
            y_rv: y,
        },
    )
    assert_no_rvs(logp)

    scale1_val = 0.5
    derived1_val = -1.0
    scale2_val = 2.0
    derived2_val = 3.0
    y_val = 1.0

    expected_logp = (
        sp.stats.halfnorm(0, 1).logpdf(scale1_val)
        + sp.stats.norm(0, scale1_val).logpdf(derived1_val)
        + sp.stats.expon(scale=1).logpdf(scale2_val)
        + sp.stats.norm(derived1_val, scale2_val).logpdf(derived2_val)
        + sp.stats.norm(derived2_val, 1).logpdf(y_val)
    )

    aeppl_logp = logp.eval(
        {
            scale1: scale1_val,
            derived1: derived1_val,
            scale2: scale2_val,
            derived2: derived2_val,
            y: y_val,
        }
    )

    assert np.isclose(expected_logp, aeppl_logp)


def test_transformed_rv_and_value():
    y_rv = at.random.halfnormal(-1, 1, name="base_rv") + 1
    y = y_rv.type()
    y.name = "y"

    transform_opt = TransformValuesOpt({y: LogTransform()})

    logp = joint_logprob({y_rv: y}, extra_rewrites=transform_opt)
    assert_no_rvs(logp)
    logp_fn = aesara.function([y], logp)

    y_test_val = -5

    assert np.isclose(
        logp_fn(y_test_val),
        sp.stats.halfnorm(0, 1).logpdf(np.exp(y_test_val)) + y_test_val,
    )


def test_multiple_base_rvs_fails():
    x_rv1 = at.random.normal(name="x_rv1")
    x_rv2 = at.random.normal(name="x_rv2")
    y_rv = x_rv1 + x_rv2

    y = y_rv.clone()

    with pytest.raises(NotImplementedError):
        joint_logprob({y_rv: y})


def test_ignore_logprob_respected():
    x_rv1 = at.random.normal(name="x_rv1")
    x_rv2 = at.random.normal(name="x_rv2")
    y_rv = x_rv1 + x_rv2

    x_rv1.tag.ignore_logprob = True
    y = y_rv.clone()

    assert joint_logprob({y_rv: y}) is not None


def test_discrete_rv_transform_fails():
    loc = at.lscalar("loc")
    y_rv = loc + at.random.poisson(1, name="base_rv")
    y = y_rv.type()
    y.name = "y"

    # Cannot match Error message in rewrite phase
    with pytest.raises(NotImplementedError):
        joint_logprob({y_rv: y})


def test_incompatible_size_fails():
    loc = at.vector("loc")
    y_rv = loc + at.random.normal(size=1, name="base_rv")
    y = y_rv.type()
    y.name = "y"

    # Cannot match Error message in rewrite phase
    with pytest.raises(NotImplementedError):
        joint_logprob({y_rv: y})


@pytest.mark.xfail(reason="Check not implemented yet, see #51")
def test_invalid_broadcasted_transform_fails():
    loc = at.vector("loc")
    y_rv = loc + at.random.normal(0, 1, size=2, name="base_rv")
    y = y_rv.type()
    y.name = "y"

    logp = joint_logprob({y_rv: y})
    logp.eval({y: [0, 0, 0, 0], loc: [0, 0, 0, 0]})
    assert False, "Should have failed before"


def test_get_single_vars_between_in_out():
    a, b, c, d, e, f = at.scalars("abcdef")
    ab = a + b
    cd = c + d
    abcd = ab + cd
    ef = e + f
    x = abcd + ef
    fgraph = FunctionGraph(outputs=[x], clone=False)

    assert tuple(_get_single_vars_between_in_out(a, x, fgraph)) == (a, ab, abcd, x)
    assert tuple(_get_single_vars_between_in_out(b, x, fgraph)) == (b, ab, abcd, x)
    assert tuple(_get_single_vars_between_in_out(c, x, fgraph)) == (c, cd, abcd, x)
    assert tuple(_get_single_vars_between_in_out(d, x, fgraph)) == (d, cd, abcd, x)
    assert tuple(_get_single_vars_between_in_out(e, x, fgraph)) == (e, ef, x)
    assert tuple(_get_single_vars_between_in_out(f, x, fgraph)) == (f, ef, x)
    assert tuple(_get_single_vars_between_in_out(ab, x, fgraph)) == (ab, abcd, x)
    assert tuple(_get_single_vars_between_in_out(cd, x, fgraph)) == (cd, abcd, x)
    assert tuple(_get_single_vars_between_in_out(ef, x, fgraph)) == (ef, x)
    assert tuple(_get_single_vars_between_in_out(abcd, x, fgraph)) == (abcd, x)
    assert tuple(_get_single_vars_between_in_out(x, x, fgraph)) == (x,)  # Edge case

    # Graph with two branches from a -> y
    a, b, c = at.scalars("abc")
    ab = a + b
    ac = a + c
    x = ab + ac
    fgraph = FunctionGraph(outputs=[x], clone=False)

    assert not _get_single_vars_between_in_out(a, x, fgraph)
    assert tuple(_get_single_vars_between_in_out(b, x, fgraph)) == (b, ab, x)
    assert tuple(_get_single_vars_between_in_out(c, x, fgraph)) == (c, ac, x)

    # Graph where input is not related to requseted output
    a, b = at.scalars("ab")
    y1 = a + 1.0
    y2 = b + 1.0
    fgraph = FunctionGraph(outputs=[y1, y2], clone=False)

    assert tuple(_get_single_vars_between_in_out(a, y1, fgraph)) == (a, y1)
    assert not _get_single_vars_between_in_out(a, y2, fgraph)
    assert not _get_single_vars_between_in_out(b, y1, fgraph)
    assert tuple(_get_single_vars_between_in_out(b, y2, fgraph)) == (b, y2)
