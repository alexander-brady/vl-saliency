from vl_saliency.core.map import SaliencyMap
from vl_saliency.transforms.pipe import Chainable, Pipeline, chainable


class DummyTransform(Chainable):
    def __call__(self, map):
        return map


# ------------------------------ Chainable Mixin ------------------------------ #


def test_chainable_rshift():
    t1 = DummyTransform()
    t2 = DummyTransform()
    pipeline = t1 >> t2
    assert isinstance(pipeline, Pipeline)
    assert pipeline.transforms == [t1, t2]

    t3 = Pipeline(t1)
    combined_pipeline = t2 >> t3
    assert isinstance(combined_pipeline, Pipeline)
    assert combined_pipeline.transforms == [t2, t1]


def test_chainable_wrapped(smap):
    @chainable
    def dummy_transform(map: SaliencyMap) -> SaliencyMap:
        return map

    # EAGER usage
    result_map = dummy_transform(smap)
    assert isinstance(result_map, SaliencyMap)

    # LAZY usage
    pipeline = dummy_transform()
    assert isinstance(pipeline, Pipeline)


# ------------------------------ Pipeline ------------------------------ #


def test_pipeline_call(smap):
    t1 = DummyTransform()
    t2 = DummyTransform()
    pipeline = Pipeline(t1, t2)
    result_map = pipeline(smap)
    assert isinstance(result_map, SaliencyMap)


def test_pipeline_rshift():
    t1 = DummyTransform()
    t2 = DummyTransform()
    pipeline1 = Pipeline(t1)
    pipeline2 = pipeline1 >> t2
    assert isinstance(pipeline2, Pipeline)
    assert pipeline2.transforms == [t1, t2]
    assert len(pipeline2) == 2

    pipeline3 = Pipeline()
    combined_pipeline = pipeline1 >> pipeline3
    assert isinstance(combined_pipeline, Pipeline)
    assert combined_pipeline.transforms == [t1]


def test_pipeline_repr():
    t1 = DummyTransform()
    t2 = DummyTransform()
    pipeline = Pipeline(t1, t2)
    repr_str = repr(pipeline)
    assert repr_str == "Pipeline(DummyTransform, DummyTransform)"


def test_pipeline_empty(smap):
    pipeline = Pipeline()
    result_map = pipeline(smap)
    assert result_map == smap
    assert len(pipeline) == 0


def test_pipeline_append_len():
    t1 = DummyTransform()
    t2 = DummyTransform()
    pipeline = Pipeline(t1)
    pipeline.append(t2)
    assert pipeline.transforms == [t1, t2]

    t3 = DummyTransform()
    other_pipeline = Pipeline(t3)
    pipeline.append(other_pipeline)
    assert pipeline.transforms == [t1, t2, t3]
