from transmitter.analysis.entropy import shannon_entropy
from transmitter.analysis.message_metrics import measure
from transmitter.communication import json_format, natural_language, vector_format
from transmitter.dataset.schema import ImageAttributes, ShapeObject


def sample() -> ImageAttributes:
    return ImageAttributes("sample", (ShapeObject("circle", "red", "centre", "large"), ShapeObject("triangle", "blue", "upper-left", "small")))


def test_all_encoders_preserve_objects():
    attributes = sample()
    assert len(json_format.decode(json_format.encode(attributes))) == 2
    assert vector_format.decode(vector_format.encode(attributes)).objects == attributes.objects
    assert "large red circle" in natural_language.encode(attributes)


def test_metrics_and_entropy():
    metrics = measure("a a!")
    assert metrics.character_count == 4
    assert metrics.word_count == 2
    assert metrics.token_count == 3
    assert shannon_entropy("aaaa") == 0.0


def test_json_is_canonical():
    assert json_format.encode(sample()) == '{"objects":[{"shape":"circle","colour":"red","position":"centre","size":"large"},{"shape":"triangle","colour":"blue","position":"upper-left","size":"small"}]}'
