from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
from napari.components import ViewerModel

from napari_nd2._reader import get_reader, read_nd2


TEST_FILE = Path(__file__).resolve().parents[1] / "test_ND2" / "YK662 Rab7 miRFP007.nd2"


def test_get_reader_handles_nd2_paths():
    assert get_reader("image.nd2") is read_nd2
    assert get_reader(["a.nd2", "b.nd2"]) is read_nd2
    assert get_reader("image.tif") is None


def test_reader_uses_to_dask_for_lazy_loading(monkeypatch, tmp_path):
    class FakeND2File:
        def __init__(self, path):
            self.path = path
            self.sizes = {"T": 2, "C": 2, "Y": 3, "X": 4}
            self.metadata = SimpleNamespace(
                channels=[
                    SimpleNamespace(
                        channel=SimpleNamespace(
                            name="magenta",
                            color=SimpleNamespace(r=255, g=0, b=255),
                            emissionLambdaNm=670.0,
                            excitationLambdaNm=561.0,
                        )
                    ),
                    SimpleNamespace(
                        channel=SimpleNamespace(
                            name="yellow",
                            color=SimpleNamespace(r=255, g=255, b=0),
                            emissionLambdaNm=530.0,
                            excitationLambdaNm=488.0,
                        )
                    ),
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def to_dask(self):
            return da.from_array(np.arange(48).reshape(2, 2, 3, 4), chunks=(1, 2, 3, 4))

        def asarray(self):
            raise AssertionError("reader should stay lazy and never call asarray()")

        def voxel_size(self):
            return SimpleNamespace(x=0.25, y=0.5, z=1.5)

        def events(self, orient="list"):
            assert orient == "list"
            return {"Time [s]": [0.25, 1.25]}

    monkeypatch.setattr("napari_nd2._reader.nd2.ND2File", FakeND2File)

    result = read_nd2(tmp_path / "lazy.nd2")
    assert len(result) == 1

    data, add_kwargs, layer_type = result[0]
    assert isinstance(data, da.Array)
    assert data.shape == (2, 2, 3, 4)
    assert layer_type == "image"
    assert add_kwargs["channel_axis"] == 1
    assert add_kwargs["name"] == ["magenta", "yellow"]
    assert add_kwargs["colormap"] == ["magenta", "yellow"]
    assert add_kwargs["scale"] == pytest.approx((1.0, 0.5, 0.25))
    assert add_kwargs["translate"] == pytest.approx((0.25, 0.0, 0.0))
    assert add_kwargs["metadata"]["time_coordinates_s"] == pytest.approx([0.25, 1.25])
    assert add_kwargs["metadata"]["time_axis_irregular"] is False


def test_reader_output_is_accepted_by_napari(monkeypatch, tmp_path):
    class FakeND2File:
        def __init__(self, path):
            self.path = path
            self.sizes = {"T": 2, "C": 2, "Y": 3, "X": 4}
            self.metadata = SimpleNamespace(
                channels=[
                    SimpleNamespace(
                        channel=SimpleNamespace(
                            name="magenta",
                            color=SimpleNamespace(r=255, g=0, b=255),
                            emissionLambdaNm=670.0,
                            excitationLambdaNm=561.0,
                        )
                    ),
                    SimpleNamespace(
                        channel=SimpleNamespace(
                            name="yellow",
                            color=SimpleNamespace(r=255, g=255, b=0),
                            emissionLambdaNm=530.0,
                            excitationLambdaNm=488.0,
                        )
                    ),
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def to_dask(self):
            return da.from_array(np.arange(48).reshape(2, 2, 3, 4), chunks=(1, 2, 3, 4))

        def voxel_size(self):
            return SimpleNamespace(x=0.25, y=0.5, z=1.5)

        def events(self, orient="list"):
            assert orient == "list"
            return {"Time [s]": [0.25, 1.25], "T Index": [0, 1]}

    monkeypatch.setattr("napari_nd2._reader.nd2.ND2File", FakeND2File)

    data, add_kwargs, _ = read_nd2(tmp_path / "viewer.nd2")[0]
    viewer = ViewerModel()
    layers = viewer.add_image(data, **add_kwargs)

    assert len(layers) == 2
    assert [layer.name for layer in layers] == ["magenta", "yellow"]
    assert [tuple(layer.scale) for layer in layers] == [
        pytest.approx((1.0, 0.5, 0.25)),
        pytest.approx((1.0, 0.5, 0.25)),
    ]


@pytest.mark.skipif(not TEST_FILE.exists(), reason="local 4 GB ND2 test file is not available")
def test_real_nd2_file_is_lazy_and_keeps_time_channel_metadata():
    result = read_nd2(TEST_FILE)
    assert len(result) == 1

    data, add_kwargs, layer_type = result[0]
    assert isinstance(data, da.Array)
    assert data.shape == (201, 2, 2304, 2304)
    assert layer_type == "image"
    assert add_kwargs["channel_axis"] == 1
    assert add_kwargs["name"] == ["miRFP1", "EYFP"]
    assert add_kwargs["colormap"] == ["magenta", "yellow"]
    assert add_kwargs["scale"] == pytest.approx((3.0002656298828128, 0.064830620007835, 0.064830620007835))
    assert add_kwargs["translate"] == pytest.approx((0.9387972800731659, 0.0, 0.0))
    assert add_kwargs["metadata"]["axes"] == ["T", "C", "Y", "X"]
    assert len(add_kwargs["metadata"]["time_coordinates_s"]) == 201
    assert add_kwargs["metadata"]["time_axis_irregular"] is True
    assert add_kwargs["metadata"]["channel_metadata"][0]["name"] == "miRFP1"
    assert add_kwargs["metadata"]["channel_metadata"][1]["name"] == "EYFP"
    assert add_kwargs["metadata"]["channel_metadata"][0]["color"] == "#ff00ff"
    assert add_kwargs["metadata"]["channel_metadata"][1]["color"] == "#ffff00"

    sample = data[0, 0, :8, :8].compute()
    assert sample.shape == (8, 8)
    assert sample.dtype == np.uint16
