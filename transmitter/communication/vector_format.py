"""Fixed-size numerical encoding with documented categorical mappings."""

from transmitter.dataset.schema import ImageAttributes, ShapeObject

SHAPE_CODES = {"circle": 1, "square": 2, "triangle": 3}
COLOUR_CODES = {"red": 1, "blue": 2, "green": 3, "yellow": 4}
POSITION_CODES = {"centre": 1, "upper-left": 2, "upper-right": 3, "lower-left": 4, "lower-right": 5}
SIZE_CODES = {"small": 1, "medium": 2, "large": 3}
MAX_OBJECTS = 3


def encode(attributes: ImageAttributes) -> list[int]:
    """Return [object_count, shape, colour, position, size] blocks; zero pads unused blocks."""
    if len(attributes.objects) > MAX_OBJECTS:
        raise ValueError(f"at most {MAX_OBJECTS} objects are supported")
    vector = [len(attributes.objects)]
    for obj in attributes.objects:
        vector.extend((SHAPE_CODES[obj.shape], COLOUR_CODES[obj.colour], POSITION_CODES[obj.position], SIZE_CODES[obj.size]))
    vector.extend([0] * (4 * (MAX_OBJECTS - len(attributes.objects))))
    return vector


def decode(vector: list[int], image_id: str = "decoded") -> ImageAttributes:
    if len(vector) != 1 + 4 * MAX_OBJECTS or not 1 <= vector[0] <= MAX_OBJECTS:
        raise ValueError("vector has an invalid length or object count")
    reverse = lambda mapping: {value: key for key, value in mapping.items()}
    shape, colour, position, size = map(reverse, (SHAPE_CODES, COLOUR_CODES, POSITION_CODES, SIZE_CODES))
    objects = []
    for offset in range(vector[0]):
        values = vector[1 + offset * 4 : 5 + offset * 4]
        try:
            objects.append(ShapeObject(shape[values[0]], colour[values[1]], position[values[2]], size[values[3]]))
        except KeyError as error:
            raise ValueError(f"invalid category code: {error.args[0]}") from error
    return ImageAttributes(image_id, tuple(objects))
