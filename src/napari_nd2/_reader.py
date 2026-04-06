from collections.abc import Sequence
from pathlib import Path

import nd2
import numpy as np


COMMON_COLORMAPS = {
    "#ff0000": "red",
    "#00ff00": "green",
    "#0000ff": "blue",
    "#ffff00": "yellow",
    "#00ffff": "cyan",
    "#ff00ff": "magenta",
    "#ffffff": "gray",
}


def get_reader(path):
    paths = _coerce_paths(path)
    if not paths:
        return None
    if any(file_path.suffix.lower() != ".nd2" for file_path in paths):
        return None
    return read_nd2


def read_nd2(path):
    paths = _coerce_paths(path)
    multiple_files = len(paths) > 1

    layer_data = []
    for file_path in paths:
        layer_data.append(_read_single_file(file_path, multiple_files=multiple_files))
    return layer_data


def _read_single_file(path, *, multiple_files):
    with nd2.ND2File(path) as nd2_file:
        data = nd2_file.to_dask()
        axes = tuple(nd2_file.sizes.keys())
        channel_axis = axes.index("C") if "C" in axes else None
        channel_metadata = _channel_metadata(nd2_file)
        channel_names = _channel_names(channel_metadata, nd2_file.sizes.get("C", 1), path, multiple_files)
        scale = _layer_axis_values(_axis_scale(nd2_file, axes), channel_axis)
        translate = _layer_axis_values(_axis_translate(nd2_file, axes), channel_axis)
        metadata = _layer_metadata(path, nd2_file, axes, channel_metadata)

    add_kwargs = {
        "metadata": metadata,
        "name": channel_names if channel_axis is not None else path.stem,
        "scale": scale,
    }

    if any(translate):
        add_kwargs["translate"] = translate

    if channel_axis is not None:
        add_kwargs["channel_axis"] = channel_axis
        add_kwargs["blending"] = ["additive"] * len(channel_names)
        colormaps = _channel_colormaps(channel_metadata)
        if colormaps is not None:
            add_kwargs["colormap"] = colormaps

    return (data, add_kwargs, "image")


def _coerce_paths(path):
    if path is None:
        return []
    if isinstance(path, (str, Path)):
        return [Path(path)]
    if isinstance(path, Sequence):
        return [Path(item) for item in path]
    return [Path(path)]


def _channel_metadata(nd2_file):
    channels = []
    for index, channel in enumerate(getattr(nd2_file.metadata, "channels", ())):
        color = getattr(channel.channel, "color", None)
        channels.append(
            {
                "index": index,
                "name": channel.channel.name or f"channel {index}",
                "color": _color_to_hex(color),
                "emission_lambda_nm": getattr(channel.channel, "emissionLambdaNm", None),
                "excitation_lambda_nm": getattr(channel.channel, "excitationLambdaNm", None),
            }
        )
    return channels


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
    if not channel_metadata:
        return None

    colormaps = []
    for channel in channel_metadata:
        colormap = COMMON_COLORMAPS.get(channel["color"])
        if colormap is None:
            return None
        colormaps.append(colormap)
    return colormaps


def _layer_metadata(path, nd2_file, axes, channel_metadata):
    metadata = {
        "path": str(path),
        "axes": list(axes),
        "sizes": dict(nd2_file.sizes),
        "channel_metadata": channel_metadata,
        "voxel_size_um": _voxel_size_metadata(nd2_file),
    }

    time_coordinates = _time_coordinates_s(nd2_file)
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


def _axis_scale(nd2_file, axes):
    voxel_size = nd2_file.voxel_size()
    scale_by_axis = {
        "X": float(voxel_size.x),
        "Y": float(voxel_size.y),
        "Z": float(voxel_size.z),
    }

    time_coordinates = _time_coordinates_s(nd2_file)
    if time_coordinates is not None and len(time_coordinates) > 1:
        scale_by_axis["T"] = float(np.mean(np.diff(time_coordinates)))

    return tuple(float(scale_by_axis.get(axis, 1.0)) for axis in axes)


def _axis_translate(nd2_file, axes):
    translate = {axis: 0.0 for axis in axes}
    time_coordinates = _time_coordinates_s(nd2_file)
    if time_coordinates is not None and len(time_coordinates):
        translate["T"] = float(time_coordinates[0])
    return tuple(float(translate[axis]) for axis in axes)


def _layer_axis_values(values, channel_axis):
    if channel_axis is None:
        return values
    return tuple(value for index, value in enumerate(values) if index != channel_axis)


def _time_coordinates_s(nd2_file):
    if "T" not in nd2_file.sizes:
        return None

    try:
        events = nd2_file.events(orient="list")
    except Exception:
        return None

    time_values = events.get("Time [s]")
    if time_values is None:
        return None

    time_coordinates = _time_coordinates_from_events(time_values, events.get("T Index"), nd2_file.sizes["T"])
    if time_coordinates is None:
        return None

    return time_coordinates


def _time_coordinates_from_events(time_values, t_indices, expected_length):
    if t_indices is None:
        time_coordinates = np.asarray(time_values, dtype=float)
        if len(time_coordinates) != expected_length or not np.all(np.isfinite(time_coordinates)):
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
