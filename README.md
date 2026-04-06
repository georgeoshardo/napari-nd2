# napari-nd2

`napari-nd2` is a small napari reader plugin for Nikon ND2 files using
[nd2](https://github.com/tlambert03/nd2) as the backend.

The reader keeps ND2 data lazy by using `ND2File.to_dask()` instead of reading
the full file into memory. Channel names, common channel colors, spatial
calibration, and time metadata are mapped into napari layer metadata.

## Development

Use `pixi` for the local environment:

```bash
pixi run -e test test
```

The repository includes optional tests against
`test_ND2/YK662 Rab7 miRFP007.nd2`. That file is ignored in git because it is
several gigabytes, so those tests only run when the file is present locally.
