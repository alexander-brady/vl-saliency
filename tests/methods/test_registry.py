import pytest

from vl_saliency.methods import registry as reg


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the registry so tests don't leak global state."""
    reg_snapshot = reg._REGISTRY.copy()
    alias_snapshot = reg._ALIASES.copy()
    reg._REGISTRY.clear()
    reg._ALIASES.clear()
    try:
        yield
    finally:
        reg._REGISTRY.clear()
        reg._ALIASES.clear()
        reg._REGISTRY.update(reg_snapshot)
        reg._ALIASES.update(alias_snapshot)


def test_register_and_resolve_by_name_and_alias():
    @reg.register("gradcam", "gradattn", "gc")
    def f(a, b):  # dummy
        return "ok"

    # resolve by canonical name
    fn = reg.resolve("gradcam")
    assert fn is f

    # resolve by alias (case-insensitive)
    assert reg.resolve("GradAttn") is f
    assert reg.resolve("GC") is f


def test_listings_only_names_and_alias_map():
    @reg.register("attn_raw", "attn")
    def f1(a, b): ...

    @reg.register("grad_raw", "grad")
    def f2(a, b): ...

    # list_methods returns only canonical names (sorted)
    methods = reg.list_methods()
    assert methods == ["attn_raw", "grad_raw"]

    # list_aliases returns alias -> canonical
    aliases = reg.list_aliases()
    assert aliases == {"attn": "attn_raw", "grad": "grad_raw"}


def test_unknown_raises_value_error():
    with pytest.raises(ValueError) as e:
        reg.resolve("nope")
    assert "Unknown saliency method" in str(e.value)


def test_list_aliases_returns_copy_not_live_view():
    @reg.register("x", "alias1")
    def fx(a, b): ...

    before = reg.list_aliases()
    before["alias1"] = "mutated"
    # Internal mapping should be unchanged; resolve still works
    assert reg.resolve("alias1") is fx


def test_alias_overwrite_last_registration_wins():
    @reg.register("a", "shared")
    def fa(a, b): ...

    @reg.register("b", "shared")
    def fb(a, b): ...

    # The alias 'shared' should now point to 'b'
    assert reg.resolve("shared") is fb
    # Canonical names still resolve correctly
    assert reg.resolve("a") is fa
    assert reg.resolve("b") is fb