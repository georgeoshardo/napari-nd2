"""Read ND2 files into napari, lazily.

Each file becomes one image layer per channel. The pixels are a dask array that the
``nd2`` package reads one frame at a time as napari shows it, so a file of any size
opens in seconds. Channels are named and coloured as in the file. Z, Y and X are
scaled in µm where the file calibrates them (in pixels where it doesn't), and T in
seconds from the frames' recorded times. A colour camera's image opens as RGB.
Each layer's metadata keeps the channels' optics and Nikon's text description.

Contrast limits come from up to nine frames (first, middle and last time point of the
first, middle and last position), so napari doesn't read the whole file for them and
a field whose cells have gone by the middle of the experiment doesn't set them.

On napari's current viewer, a caption gives the position name and the frame time the
sliders are on.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

# napari's names for the colours ND2 channels most often have
COMMON_COLORMAPS = {
    "#ff0000": "red",
    "#00ff00": "green",
    "#0000ff": "blue",
    "#ffff00": "yellow",
    "#00ffff": "cyan",
    "#ff00ff": "magenta",
    "#ffffff": "gray",
}
# Contrast: these percentiles of the sampled frames' pixels. Fluorescent cells cover
# a few percent of a field, so the top one is high.
CONTRAST_PERCENTILES = (1.0, 99.5)
SAMPLE_STEP = 4  # every 4th row and column of the sampled frames
METADATA_KEY = "napari_nd2"


def get_reader(path):
    """napari's reader hook: :func:`read_nd2` for ``.nd2`` files, else ``None``."""
    paths = _coerce_paths(path)
    if not paths:
        return None
    if any(file_path.suffix.lower() != ".nd2" for file_path in paths):
        return None
    return read_nd2


def read_nd2(path):
    """The layers of one or more ND2 files, and the caption on napari's viewer."""
    paths = _coerce_paths(path)
    multiple_files = len(paths) > 1
    layer_data = [
        _read_single_file(file_path, multiple_files=multiple_files)
        for file_path in paths
    ]
    viewer = _current_viewer()
    if viewer is not None:
        show_position(viewer)
    return layer_data


def _read_single_file(path, *, multiple_files):
    nd2 = _nd2()
    with nd2.ND2File(path) as nd2_file:
        # read frame by frame; the reads reopen the file after this block closes it
        data = nd2_file.to_dask()
        axes = tuple(str(axis) for axis in nd2_file.sizes)  # the order of data's axes
        channel_axis = axes.index("C") if "C" in axes else None
        # a colour camera's red, green and blue: the last axis, shown as RGB
        rgb = axes[-1] == "S"
        # the axes of each layer: without the channels napari splits off, and
        # without the colour components, which aren't a slider or a scaled axis
        layer_axes = tuple(axis for axis in axes if axis not in ("C", "S"))
        channel_metadata = _channel_metadata(nd2_file)
        channel_names = _channel_names(
            channel_metadata, nd2_file.sizes.get("C", 1), path, multiple_files
        )
        time_coordinates = _time_coordinates_s(nd2_file)
        calibration = _calibration_um(nd2_file)
        scale = _axis_scale(layer_axes, calibration, time_coordinates)
        translate = _axis_translate(layer_axes, time_coordinates)
        metadata = _layer_metadata(
            path, nd2_file, axes, channel_metadata, time_coordinates
        )
        metadata[METADATA_KEY] = {
            "axes": list(layer_axes),
            "positions": _position_names(nd2_file) if "P" in axes else None,
            "time_coordinates_s": None
            if time_coordinates is None
            else time_coordinates.tolist(),
        }
        contrast = None if rgb else _contrast_limits(data, axes, channel_axis)

    add_kwargs: dict[str, Any] = {
        "metadata": metadata,
        "name": channel_names if channel_axis is not None else path.stem,
        "scale": scale,
    }
    if not rgb:  # a colour camera's image keeps its true colours: no stretch
        add_kwargs["contrast_limits"] = (
            contrast if channel_axis is not None else contrast[0]
        )
    if _image_takes("axis_labels"):  # napari >= 0.8
        add_kwargs["axis_labels"] = tuple(metadata[METADATA_KEY]["axes"])
    if _image_takes("units"):  # napari >= 0.6; the scale bar reads them
        add_kwargs["units"] = _axis_units(layer_axes, calibration, time_coordinates)
    if any(translate):
        add_kwargs["translate"] = translate
    if rgb:
        add_kwargs["rgb"] = True
    if channel_axis is not None:
        add_kwargs["channel_axis"] = channel_axis
        add_kwargs["blending"] = ["additive"] * len(channel_names)
        colormaps = None if rgb else _channel_colormaps(channel_metadata)
        if colormaps is not None:
            add_kwargs["colormap"] = colormaps

    return (data, add_kwargs, "image")


def show_position(viewer) -> None:
    """Caption ``viewer``'s canvas with the position name, frame and time the sliders
    are on, from the top layer an ND2 file was read into; and name the sliders after
    the file's axes.
    """
    overlay = _text_overlay(viewer)

    def update(_event=None) -> None:
        where = next(
            (
                layer.metadata[METADATA_KEY]
                for layer in reversed(viewer.layers)
                if METADATA_KEY in layer.metadata
            ),
            None,
        )
        if where is None or viewer.dims.ndim < len(where["axes"]):
            return
        n = len(where["axes"])
        labels = list(viewer.dims.axis_labels)
        if all(label.lstrip("-").isdigit() for label in labels[-n:]):  # napari < 0.8
            viewer.dims.axis_labels = tuple(labels[:-n] + where["axes"])
        step = dict(zip(where["axes"], viewer.dims.current_step[-n:], strict=True))
        parts = []
        positions = where["positions"]
        if "P" in step and positions and step["P"] < len(positions):
            parts.append(positions[step["P"]])
        if "T" in step:
            times = where["time_coordinates_s"]
            minutes = (
                f", {times[step['T']] / 60:.1f} min"
                if times and step["T"] < len(times)
                else ""
            )
            parts.append(f"frame {step['T']}{minutes}")
        overlay.text = " · ".join(parts)
        overlay.visible = bool(parts)

    viewer.dims.events.current_step.connect(update)
    viewer.layers.events.inserted.connect(update)
    update()


def _contrast_limits(data, axes, channel_axis) -> list[tuple[float, float]]:
    """Per channel, ``CONTRAST_PERCENTILES`` of up to nine sampled frames."""
    sample = data
    for k, axis in enumerate(axes):  # one axis at a time: dask indexes one list
        n = data.shape[k]
        if axis in ("Y", "X"):
            pick = slice(None, None, SAMPLE_STEP)
        elif axis in ("C", "S"):
            continue
        else:
            pick = sorted({0, n // 2, n - 1}) if axis in ("T", "P") else [n // 2]
        sample = sample[(slice(None),) * k + (pick,)]
    sample = np.asarray(sample.compute())
    if channel_axis is None:
        channels = [sample]
    else:
        channels = list(np.moveaxis(sample, channel_axis, 0))
    limits = []
    for pixels in channels:
        lo, hi = (float(v) for v in np.percentile(pixels, CONTRAST_PERCENTILES))
        limits.append((lo, hi if hi > lo else lo + 1))
    return limits


def _position_names(nd2_file):
    """Each position's name, or ``XYPos:<index>`` for one without, as nd2 names
    them."""
    for loop in nd2_file.experiment:
        if loop.type == "XYPosLoop":
            points = loop.parameters.points
            return [point.name or f"XYPos:{i}" for i, point in enumerate(points)]
    return ["XYPos:0"]


def _coerce_paths(path):
    if path is None:
        return []
    if isinstance(path, (str, Path)):
        return [Path(path)]
    if isinstance(path, Sequence):
        return [Path(item) for item in path]
    return [Path(path)]


def _nd2():
    return import_module("nd2")


def _current_viewer():
    try:
        import napari
    except ImportError:
        return None
    return napari.current_viewer()


def _image_takes(parameter: str) -> bool:
    from napari.layers import Image

    return parameter in inspect.signature(Image).parameters


def _text_overlay(viewer):
    canvas = getattr(viewer, "canvas", None)  # napari >= 0.8
    return canvas.overlays.text if canvas is not None else viewer.text_overlay


def _channel_metadata(nd2_file):
    channels = []
    for index, channel in enumerate(getattr(nd2_file.metadata, "channels", None) or ()):
        color = getattr(channel.channel, "color", None)
        channels.append(
            {
                "index": index,
                "name": channel.channel.name or f"channel {index}",
                "color": _color_to_hex(color),
                "emission_lambda_nm": getattr(
                    channel.channel, "emissionLambdaNm", None
                ),
                "excitation_lambda_nm": getattr(
                    channel.channel, "excitationLambdaNm", None
                ),
                **_microscope_metadata(getattr(channel, "microscope", None)),
            }
        )
    return channels


def _microscope_metadata(microscope):
    """A channel's objective and magnifications, as the file records them."""
    if microscope is None:
        return {}
    fields = {
        "objective": "objectiveName",
        "objective_magnification": "objectiveMagnification",
        "objective_na": "objectiveNumericalAperture",
        "zoom_magnification": "zoomMagnification",
        "immersion_refractive_index": "immersionRefractiveIndex",
        "modalities": "modalityFlags",
    }
    return {key: getattr(microscope, name, None) for key, name in fields.items()}


def _channel_names(channel_metadata, channel_count, path, multiple_files):
    if channel_metadata:
        names = [channel["name"] for channel in channel_metadata]
    else:
        names = [f"channel {index}" for index in range(channel_count)]

    if len(names) < channel_count:
        names.extend(f"channel {index}" for index in range(len(names), channel_count))

    if not multiple_files:
        return names

    return [f"{path.stem} - {name}" for name in names]


def _channel_colormaps(channel_metadata):
    """napari's name for each channel's colour, or a black-to-colour map."""
    if not channel_metadata:
        return None
    colormaps = []
    for channel in channel_metadata:
        color = channel["color"]
        if color is None:
            return None
        colormaps.append(
            COMMON_COLORMAPS.get(color) or _colormap(channel["name"], color)
        )
    return colormaps


def _colormap(name, color):
    from napari.utils.colormaps import Colormap

    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return Colormap([[0, 0, 0, 1], [*rgb, 1]], name=name)


def _layer_metadata(path, nd2_file, axes, channel_metadata, time_coordinates):
    metadata = {
        "path": str(path),
        "axes": list(axes),
        "sizes": dict(nd2_file.sizes),
        "channel_metadata": channel_metadata,
        "voxel_size_um": _voxel_size_metadata(nd2_file),
        "calibrated_um": _calibration_um(nd2_file),
        # Nikon's description of the acquisition, its date, capture and optics
        "text_info": {key: str(value) for key, value in nd2_file.text_info.items()},
    }
    if time_coordinates is not None:
        metadata["time_coordinates_s"] = time_coordinates.tolist()
        metadata["time_axis_irregular"] = not _is_regular(time_coordinates)
    return metadata


def _voxel_size_metadata(nd2_file):
    voxel_size = nd2_file.voxel_size()
    return {
        "x": float(voxel_size.x),
        "y": float(voxel_size.y),
        "z": float(voxel_size.z),
    }


def _calibration_um(nd2_file):
    """The size of a pixel in µm along X, Y and Z, or None along an axis the file
    doesn't calibrate (nd2 then reports 1)."""
    voxel_size = nd2_file.voxel_size()
    sizes = {"X": voxel_size.x, "Y": voxel_size.y, "Z": voxel_size.z}
    channels = getattr(nd2_file.metadata, "channels", None) or ()
    volume = getattr(channels[0], "volume", None) if channels else None
    calibrated = getattr(volume, "axesCalibrated", None)
    if calibrated is None:  # no record: trust a size other than nd2's default of 1
        calibrated = [sizes[axis] != 1 for axis in "XYZ"]
    return {
        axis: float(sizes[axis]) if is_calibrated else None
        for axis, is_calibrated in zip("XYZ", calibrated, strict=True)
    }


def _axis_scale(axes, calibration, time_coordinates):
    """µm per pixel along calibrated X, Y and Z, seconds per frame along T when the
    frame times are known, 1 otherwise."""
    scale_by_axis = {axis: size for axis, size in calibration.items() if size}
    if time_coordinates is not None and len(time_coordinates) > 1:
        scale_by_axis["T"] = float(np.mean(np.diff(time_coordinates)))
    return tuple(float(scale_by_axis.get(axis, 1.0)) for axis in axes)


def _axis_units(axes, calibration, time_coordinates):
    """The units of :func:`_axis_scale`: µm, seconds, or pixels where neither is
    known (positions, and axes the file doesn't calibrate)."""
    units = {axis: "µm" for axis, size in calibration.items() if size}
    if time_coordinates is not None and len(time_coordinates) > 1:
        units["T"] = "s"
    return tuple(units.get(axis, "pixel") for axis in axes)


def _axis_translate(axes, time_coordinates):
    translate = dict.fromkeys(axes, 0.0)
    if time_coordinates is not None and len(time_coordinates):
        translate["T"] = float(time_coordinates[0])
    return tuple(float(translate[axis]) for axis in axes)


def _time_coordinates_s(nd2_file):
    """Each time point's time in seconds: the first position's frame at it."""
    if "T" not in nd2_file.sizes:
        return None

    try:
        events = nd2_file.events(orient="list")
    except Exception:
        return None

    time_values = events.get("Time [s]")
    if time_values is None:
        return None

    return _time_coordinates_from_events(
        time_values, events.get("T Index"), nd2_file.sizes["T"]
    )


def _time_coordinates_from_events(time_values, t_indices, expected_length):
    if t_indices is None:
        time_coordinates = np.asarray(time_values, dtype=float)
        if len(time_coordinates) != expected_length or not np.all(
            np.isfinite(time_coordinates)
        ):
            return None
        return time_coordinates

    coordinates = np.full(expected_length, np.nan, dtype=float)
    for frame_index, time_value in zip(t_indices, time_values, strict=False):
        if frame_index is None:
            continue
        if np.isnan(frame_index) or np.isnan(time_value):
            continue
        frame_index = int(frame_index)
        if 0 <= frame_index < expected_length and np.isnan(coordinates[frame_index]):
            coordinates[frame_index] = float(time_value)

    if np.any(~np.isfinite(coordinates)):
        return None

    return coordinates


def _is_regular(values):
    if len(values) < 3:
        return True
    diffs = np.diff(values)
    return bool(np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-3))


def _color_to_hex(color):
    if color is None:
        return None
    return f"#{int(color.r):02x}{int(color.g):02x}{int(color.b):02x}"
