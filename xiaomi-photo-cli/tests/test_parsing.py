import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from photoflow.parsing import FaceParser  # noqa: E402


def _fake_bisenet(path):
    """1x1 conv 3->19: class 0 = 0, class 1 ('skin') = 2*green(normalised). Bright green -> skin, dark -> background."""
    w = np.zeros((19, 3, 1, 1), np.float32)
    w[1, 1, 0, 0] = 2.0
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1])
    g = helper.make_graph([node], "fake",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 512, 512])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 19, 512, 512])],
                          [numpy_helper.from_array(w, "w")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def test_face_parser_skin_mask(tmp_path):
    p = tmp_path / "fake.onnx"
    _fake_bisenet(p)
    parser = FaceParser(p)
    img = np.full((300, 400, 3), 0.1, np.float32)
    img[:, :200] = 0.9
    m = parser.skin_mask(img)
    assert m.shape == (300, 400) and m[150, 50] > 0.95 and m[150, 350] < 0.05
