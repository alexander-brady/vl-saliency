from vl_saliency.ops.fuse import FusableMixin, fusable, is_fusable


def test_fusable_decorator_marks_function():
    @fusable
    def foo(x):
        return x + 1

    assert foo(1) == 2
    assert getattr(foo, "_is_fusable", False) is True
    assert is_fusable(foo) is True


def test_is_fusable_false_for_regular_function():
    def bar(x):
        return x

    assert is_fusable(bar) is False


def test_is_fusable_none():
    assert is_fusable(None) is True


def test_fusable_mixin_marks_class():
    class Op(FusableMixin):
        def __call__(self, scores, mask):
            return scores

    op = Op()
    assert getattr(op, "_is_fusable", False) is True
    assert is_fusable(op) is True
