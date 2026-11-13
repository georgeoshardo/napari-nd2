# napari-nd2

A napari reader for Nikon ND2 files. Drop an `.nd2` file onto napari and it opens
lazily: frames are read with [nd2](https://github.com/tlambert03/nd2) as you move the
sliders, so a file of hundreds of gigabytes opens in a second or two.

Each channel becomes an image layer, named and coloured as in the file. Y and X are in
µm, T in seconds from the frames' recorded times. Contrast is set from up to nine frames
spread across time and positions. A caption in the corner gives the position name and
the frame and time the sliders are on.

```
pip install napari-nd2
```
