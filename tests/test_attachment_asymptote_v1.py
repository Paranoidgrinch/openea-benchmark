from openea_benchmark.attachment.asymptote import (
    BindingStatus,
    DissociationChannel,
    evaluate_binding_status,
)


def channel(e):
    return DissociationChannel(
        "c1", "A", "B", e
    )


def test_bound_state():
    result = evaluate_binding_status(
        -75.0,
        "BRACKETED_SINGLE_MINIMUM",
        (channel(-74.9),),
    )
    assert result.status == BindingStatus.BOUND


def test_unbound_state():
    result = evaluate_binding_status(
        -74.8,
        "BRACKETED_SINGLE_MINIMUM",
        (channel(-74.9),),
    )
    assert result.status == BindingStatus.UNBOUND


def test_missing_asymptote_is_unresolved():
    result = evaluate_binding_status(
        -75.0,
        "BRACKETED_SINGLE_MINIMUM",
        tuple(),
    )
    assert result.status == BindingStatus.UNRESOLVED


def test_missing_minimum_is_unresolved():
    result = evaluate_binding_status(
        None,
        "NO_BRACKETED_MINIMUM",
        (channel(-74.9),),
    )
    assert result.status == BindingStatus.UNRESOLVED
