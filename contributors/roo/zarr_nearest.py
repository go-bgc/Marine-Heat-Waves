"""Nearest-neighbor sampling of rectilinear xarray/Zarr data onto a DataFrame."""

from __future__ import annotations

from os import PathLike
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr


XarraySource = Union[str, PathLike[str], xr.Dataset, xr.DataArray]


def sample_zarr_nearest(
    source: XarraySource,
    points: pd.DataFrame,
    *,
    variables: Optional[Sequence[str]] = None,
    time_col: str = "time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_coord: str = "time",
    lat_coord: str = "lat",
    lon_coord: str = "lon",
    output_prefix: str = "zarr_",
    max_distance: Optional[Mapping[str, Any]] = None,
    longitude_convention: str = "auto",
    include_match_coordinates: bool = False,
    include_distances: bool = False,
    batch_size: Optional[int] = 100_000,
    copy: bool = True,
    open_zarr_kwargs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Sample a rectilinear gridded dataset at scattered space-time points.

    Parameters
    ----------
    source
        An open ``xarray.Dataset``/``DataArray``, or a path/URL/store accepted
        by ``xarray.open_zarr``.
    points
        DataFrame containing one time, latitude, and longitude per row. Row
        order and index are preserved, including duplicate index labels.
    variables
        Data variables to sample. By default, all variables whose dimensions
        are fully indexed by the selected time/latitude/longitude coordinates
        are used. Explicitly selected variables may not have unselected
        dimensions (for example, ``depth`` or ``ensemble``).
    *_col, *_coord
        Names in ``points`` and in the xarray object, respectively.
    output_prefix
        Prefix for sampled value columns. Use ``""`` for original names.
    max_distance
        Optional per-axis acceptance thresholds. Keys are ``"time"``,
        ``"lat"``, and/or ``"lon"``. Time values are accepted by
        ``pandas.Timedelta`` (for example ``"3h"``); spatial values are in
        coordinate units, normally degrees. If any threshold is exceeded, all
        sampled outputs for that row are missing.
    longitude_convention
        ``"auto"`` maps query longitudes to the dataset's apparent convention;
        ``"-180_180"`` and ``"0_360"`` force a convention; ``"none"`` leaves
        them unchanged.
    include_match_coordinates
        Add the actual grid coordinates chosen by nearest-neighbor lookup.
    include_distances
        Add absolute offsets from each query to its chosen grid point. The time
        offset is a timedelta; latitude/longitude offsets are numeric.
    batch_size
        Number of valid rows selected per xarray operation. Set to ``None`` to
        process all valid rows at once.
    copy
        If true, return a copy of ``points``; otherwise add columns in place.
    open_zarr_kwargs
        Extra keyword arguments for ``xarray.open_zarr``. Ignored when
        ``source`` is already an xarray object.

    Returns
    -------
    pandas.DataFrame
        The input rows plus sampled columns.

    Notes
    -----
    This function targets rectilinear grids with three one-dimensional indexed
    coordinates. Curvilinear grids (two-dimensional latitude/longitude) need a
    spatial index such as a KD-tree instead.

    Examples
    --------
    >>> sampled = sample_zarr_nearest(
    ...     "s3://bucket/weather.zarr",
    ...     observations,
    ...     variables=["air_temperature", "precipitation"],
    ...     time_coord="valid_time",
    ...     lat_coord="latitude",
    ...     lon_coord="longitude",
    ...     max_distance={"time": "90min", "lat": 0.25, "lon": 0.25},
    ...     open_zarr_kwargs={"consolidated": True},
    ... )
    """
    required_columns = [time_col, lat_col, lon_col]
    missing_columns = [name for name in required_columns if name not in points]
    if missing_columns:
        raise KeyError(f"Missing DataFrame columns: {missing_columns}")

    if longitude_convention not in {"auto", "-180_180", "0_360", "none"}:
        raise ValueError(
            "longitude_convention must be 'auto', '-180_180', '0_360', or 'none'"
        )
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive or None")

    owns_dataset = not isinstance(source, (xr.Dataset, xr.DataArray))
    if owns_dataset:
        dataset: Union[xr.Dataset, xr.DataArray] = xr.open_zarr(
            source, **dict(open_zarr_kwargs or {})
        )
    else:
        dataset = source

    try:
        if isinstance(dataset, xr.DataArray):
            data_name = dataset.name or "value"
            dataset = dataset.to_dataset(name=data_name)

        coord_names = {
            "time": time_coord,
            "lat": lat_coord,
            "lon": lon_coord,
        }
        missing_coords = [name for name in coord_names.values() if name not in dataset.coords]
        if missing_coords:
            raise KeyError(f"Missing xarray coordinates: {missing_coords}")

        query_dims = set()
        for logical_name, coord_name in coord_names.items():
            coord = dataset[coord_name]
            if coord.ndim != 1:
                raise ValueError(
                    f"Coordinate {coord_name!r} ({logical_name}) must be one-dimensional; "
                    f"got dimensions {coord.dims}."
                )
            query_dims.update(coord.dims)

        if variables is None:
            selected_variables = [
                name
                for name, array in dataset.data_vars.items()
                if set(array.dims).issubset(query_dims)
            ]
            if not selected_variables:
                raise ValueError(
                    "No data variables depend only on the selected coordinates; "
                    "pass variables explicitly after selecting any extra dimensions."
                )
        else:
            selected_variables = list(variables)
            missing_variables = [name for name in selected_variables if name not in dataset]
            if missing_variables:
                raise KeyError(f"Missing xarray variables: {missing_variables}")

        for name in selected_variables:
            extra_dims = set(dataset[name].dims) - query_dims
            if extra_dims:
                raise ValueError(
                    f"Variable {name!r} has unselected dimensions {sorted(extra_dims)}. "
                    "Select them on the xarray object before calling this function."
                )

        # Auxiliary 1-D coordinates are selectable only when backed by an xindex.
        working = dataset[selected_variables]
        for coord_name in coord_names.values():
            if coord_name not in working.xindexes:
                working = working.set_xindex(coord_name)

        times = pd.to_datetime(points[time_col], errors="coerce")
        if isinstance(times.dtype, pd.DatetimeTZDtype):
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        latitudes = pd.to_numeric(points[lat_col], errors="coerce").to_numpy(float)
        longitudes = pd.to_numeric(points[lon_col], errors="coerce").to_numpy(float)
        longitudes = _normalize_longitudes(
            longitudes, working[lon_coord], longitude_convention
        )

        valid = times.notna().to_numpy() & np.isfinite(latitudes) & np.isfinite(longitudes)
        valid_positions = np.flatnonzero(valid)
        output = points.copy() if copy else points

        # Collect values with integer row positions, never DataFrame index labels.
        collected: dict[str, list[pd.Series]] = {
            name: [] for name in selected_variables
        }
        if include_match_coordinates:
            for logical_name in coord_names:
                collected[f"matched_{logical_name}"] = []
        if include_distances:
            for logical_name in coord_names:
                collected[f"distance_{logical_name}"] = []

        if batch_size is None:
            batch_size = max(len(valid_positions), 1)

        point_dim = "__sample_point__"
        while point_dim in working.dims or point_dim in working.coords:
            point_dim += "_"

        limits = dict(max_distance or {})
        unknown_limits = set(limits) - {"time", "lat", "lon"}
        if unknown_limits:
            raise KeyError(
                f"Unknown max_distance keys: {sorted(unknown_limits)}; "
                "use 'time', 'lat', and/or 'lon'."
            )

        for start in range(0, len(valid_positions), batch_size):
            positions = valid_positions[start : start + batch_size]
            requested_time = times.iloc[positions].to_numpy(dtype="datetime64[ns]")
            requested_lat = latitudes[positions]
            requested_lon = longitudes[positions]

            indexers = {
                time_coord: xr.DataArray(requested_time, dims=point_dim),
                lat_coord: xr.DataArray(requested_lat, dims=point_dim),
                lon_coord: xr.DataArray(requested_lon, dims=point_dim),
            }
            nearest = working.sel(indexers, method="nearest").load()

            matched_time = np.asarray(nearest[time_coord].values)
            matched_lat = np.asarray(nearest[lat_coord].values, dtype=float)
            matched_lon = np.asarray(nearest[lon_coord].values, dtype=float)
            distance_time = np.abs(
                pd.to_datetime(matched_time) - pd.to_datetime(requested_time)
            ).to_numpy()
            distance_lat = np.abs(matched_lat - requested_lat)
            # Circular difference makes dateline-adjacent longitudes comparable.
            distance_lon = np.abs((matched_lon - requested_lon + 180.0) % 360.0 - 180.0)

            accepted = np.ones(len(positions), dtype=bool)
            if "time" in limits:
                accepted &= distance_time <= pd.Timedelta(limits["time"]).to_timedelta64()
            if "lat" in limits:
                accepted &= distance_lat <= float(limits["lat"])
            if "lon" in limits:
                accepted &= distance_lon <= float(limits["lon"])

            for name in selected_variables:
                array = nearest[name]
                remaining_dims = set(array.dims) - {point_dim}
                if remaining_dims:
                    raise ValueError(
                        f"Variable {name!r} still has dimensions {sorted(remaining_dims)} "
                        "after nearest-neighbor selection."
                    )
                if point_dim in array.dims:
                    values = np.asarray(array.transpose(point_dim).values)
                else:
                    values = np.repeat(np.asarray(array.values).item(), len(positions))
                collected[name].append(
                    pd.Series(values[accepted], index=positions[accepted])
                )

            if include_match_coordinates:
                for name, values in (
                    ("matched_time", matched_time),
                    ("matched_lat", matched_lat),
                    ("matched_lon", matched_lon),
                ):
                    collected[name].append(
                        pd.Series(values[accepted], index=positions[accepted])
                    )
            if include_distances:
                for name, values in (
                    ("distance_time", distance_time),
                    ("distance_lat", distance_lat),
                    ("distance_lon", distance_lon),
                ):
                    collected[name].append(
                        pd.Series(values[accepted], index=positions[accepted])
                    )

        full_index = pd.RangeIndex(len(output))
        for name, pieces in collected.items():
            output_name = f"{output_prefix}{name}"
            if pieces:
                values = pd.concat(pieces).reindex(full_index)
            else:
                values = pd.Series(np.nan, index=full_index)
            # Use an array for positional assignment when output has a non-unique index.
            output[output_name] = values.array

        return output
    finally:
        if owns_dataset:
            dataset.close()


def _normalize_longitudes(
    values: np.ndarray,
    dataset_longitude: xr.DataArray,
    convention: str,
) -> np.ndarray:
    """Return a copy of query longitudes in the requested convention."""
    result = values.copy()
    if convention == "none":
        return result

    if convention == "auto":
        finite_grid_values = np.asarray(dataset_longitude.values, dtype=float)
        finite_grid_values = finite_grid_values[np.isfinite(finite_grid_values)]
        if finite_grid_values.size == 0:
            return result
        if finite_grid_values.max() > 180.0 and finite_grid_values.min() >= 0.0:
            convention = "0_360"
        elif finite_grid_values.min() < 0.0 and finite_grid_values.max() <= 180.0:
            convention = "-180_180"
        else:
            return result

    if convention == "0_360":
        result = result % 360.0
    else:
        result = (result + 180.0) % 360.0 - 180.0
    return result


__all__ = ["sample_zarr_nearest"]
