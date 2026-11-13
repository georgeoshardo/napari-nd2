"""The reader on a real ND2 file: two frames, two positions, phase contrast and mCherry,
64 x 96 pixels of a mother-machine acquisition (``tests/data/tiny.nd2``)."""

from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import nd2
import numpy as np
import pytest
from napari.components import ViewerModel

from napari_nd2._reader import (
    _axis_scale,
    _axis_units,
    _calibration_um,
    _frame_times_s,
    _text_overlay,
    _time_coordinates_from_events,
    get_reader,
    read_nd2,
    show_position,
)

TINY = Path(__file__).parent / "data" / "tiny.nd2"


@pytest.fixture
def tiny():
    with nd2.ND2File(TINY) as f:
        return {
            "pixels": f.asarray(),  # (T, P, C, Y, X)
            "pixel_um": f.voxel_size().x,
            "times_s": [e["Time [s]"] for e in f.events(orient="records")][::2],
        }


def test_get_reader_takes_nd2_files_only():
    assert get_reader("image.nd2") is read_nd2
    assert get_reader("IMAGE.ND2") is read_nd2
    assert get_reader(["a.nd2", "b.nd2"]) is read_nd2
    assert get_reader("image.tif") is None
    assert get_reader(["a.nd2", "b.tif"]) is None


def test_napari_opens_a_dropped_nd2_with_this_plugin(monkeypatch):
    """What a drop does: napari finds the reader from the installed manifest."""
    from napari.components import viewer_model
    from napari.plugins import _initialize_plugins

    _initialize_plugins()
    monkeypatch.setattr(viewer_model, "get_preferred_reader", lambda path: None)
    layers = ViewerModel().open(TINY, plugin="napari-nd2")
    assert [layer.name for layer in layers] == ["PC", "mCherry"]


def test_layers_are_lazy_named_coloured_and_scaled(tiny):
    viewer = ViewerModel()
    ((data, kwargs, _),) = read_nd2(TINY)
    assert isinstance(data, da.Array)  # read frame by frame as napari shows it
    layers = viewer.add_image(data, **kwargs)
    assert [layer.name for layer in layers] == ["PC", "mCherry"]
    assert [layer.colormap.name for layer in layers] == ["gray", "red"]
    for c, layer in enumerate(layers):
        assert layer.data.shape == (2, 2, 64, 96)
        np.testing.assert_array_equal(np.asarray(layer.data), tiny["pixels"][:, :, c])
        t_step = tiny["times_s"][1] - tiny["times_s"][0]
        np.testing.assert_allclose(
            layer.scale, (t_step, 1, tiny["pixel_um"], tiny["pixel_um"])
        )
        assert layer.translate[0] == pytest.approx(tiny["times_s"][0])
        assert [str(unit) for unit in layer.units] == [
            "second",
            "pixel",
            "micrometer",
            "micrometer",
        ]
        lo, hi = layer.contrast_limits
        channel = tiny["pixels"][:, :, c]
        assert channel.min() <= lo < hi <= channel.max()
        assert layer.metadata["napari_nd2"]["positions"] == ["XYPos:0", "XYPos:1"]


def test_caption_gives_the_time_of_the_frame_on_screen():
    """Positions are imaged one after another, so each (time point, position) frame
    has its own time; the caption gives that one, not the time point's."""
    with nd2.ND2File(TINY) as f:
        time_s = {
            (e["T Index"], e["P Index"]): e["Time [s]"]
            for e in f.events(orient="records")
        }
    viewer = ViewerModel()
    ((data, kwargs, _),) = read_nd2(TINY)
    viewer.add_image(data, **kwargs)
    show_position(viewer)
    assert viewer.dims.axis_labels == ("T", "P", "Y", "X")
    for t, p in [(1, 1), (1, 0), (0, 1)]:
        viewer.dims.current_step = (t, p, 0, 0)
        minutes = time_s[t, p] / 60
        assert _text_overlay(viewer).text == f"XYPos:{p} · frame {t}, {minutes:.1f} min"
    assert round(time_s[1, 1] / 60, 1) != round(time_s[1, 0] / 60, 1)


def test_frame_times_take_each_stack_s_first_frame_and_leave_gaps_empty():
    file = SimpleNamespace(sizes={"T": 2, "P": 2, "Z": 2, "Y": 4, "X": 4})
    events = {  # T0 P0 has two Z frames; T1 P1 was never imaged
        "T Index": [0, 0, 0, 1],
        "P Index": [0, 0, 1, 0],
        "Time [s]": [0.5, 0.7, 30.0, 60.0],
    }
    times = _frame_times_s(file, events)
    np.testing.assert_array_equal(times, [[0.5, 30.0], [60.0, np.nan]])
    no_positions = SimpleNamespace(sizes={"T": 2, "Y": 4, "X": 4})
    assert _frame_times_s(no_positions, events) is None


def test_two_files_are_named_after_their_file():
    layer_data = read_nd2([TINY, TINY])
    assert [kwargs["name"] for _, kwargs, _ in layer_data] == [
        ["tiny - PC", "tiny - mCherry"]
    ] * 2


def test_frame_times_come_from_the_events_of_each_time_point():
    times = _time_coordinates_from_events([0.5, 0.6, 60.5, 60.6], [0, 0, 1, 1], 2)
    np.testing.assert_allclose(times, [0.5, 60.5])
    assert _time_coordinates_from_events([0.5], [0], 2) is None  # a time point missing


# Colour-camera files from nd2's own test data
# (github.com/tlambert03/nd2, release `test-data`): a brightfield image, and two.
RGB = Path(__file__).parent / "data" / "dims_rgb.nd2"
RGB_TWO_CHANNELS = Path(__file__).parent / "data" / "dims_rgb_c2x64y64.nd2"


def test_a_colour_camera_image_opens_as_rgb_in_its_own_colours():
    with nd2.ND2File(RGB) as f:
        pixels = f.asarray()  # (Y, X, S)
        pixel_um = f.voxel_size().x
    ((data, kwargs, _),) = read_nd2(RGB)
    layer = ViewerModel().add_image(data, **kwargs)
    assert layer.rgb
    np.testing.assert_array_equal(np.asarray(layer.data), pixels)
    np.testing.assert_allclose(layer.scale, (pixel_um, pixel_um))
    assert [str(unit) for unit in layer.units] == ["micrometer", "micrometer"]
    assert tuple(layer.contrast_limits) == (0, 255)  # not stretched


def test_each_channel_of_a_colour_camera_is_an_rgb_layer():
    ((data, kwargs, _),) = read_nd2(RGB_TWO_CHANNELS)
    layers = ViewerModel().add_image(data, **kwargs)
    assert len(layers) == 2
    assert all(layer.rgb and layer.data.shape == (64, 64, 3) for layer in layers)


def test_axes_the_file_does_not_calibrate_stay_in_pixels():
    calibration = {"X": 0.5, "Y": 0.5, "Z": None}
    axes = ("T", "Z", "Y", "X")
    assert _axis_scale(axes, calibration, None) == (1.0, 1.0, 0.5, 0.5)
    assert _axis_units(axes, calibration, None) == ("pixel", "pixel", "µm", "µm")
    uncalibrated = {"X": None, "Y": None, "Z": None}
    assert _axis_units(("Y", "X"), uncalibrated, None) == ("pixel", "pixel")
    with nd2.ND2File(TINY) as f:
        assert _calibration_um(f) == {
            "X": f.voxel_size().x,
            "Y": f.voxel_size().y,
            "Z": None,
        }


def test_layers_keep_the_optics_and_nikons_description():
    ((_, kwargs, _),) = read_nd2(TINY)
    metadata = kwargs["metadata"]
    assert metadata["channel_metadata"][0]["objective"] == "Plan Apo λ 40x Ph2 DM"
    assert metadata["channel_metadata"][0]["objective_magnification"] == 40
    assert set(metadata["text_info"]) >= {"description", "date", "optics"}
