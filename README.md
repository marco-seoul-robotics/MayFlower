## MayFlower

MayFlower is machine learning-based rain vulnerable region detector

## How to Run code?

### Prerequisites

- numpy
- matplotlib
- opencv-python
- pyproj

### Setup environment

For development environment, run script:
`pip install -r requirements.txt`

### (Extra) Deep Learning model

ResNet-based deep learning model is trained to predict rain-vulnerable region for future usage.
For further information, [dl/README.md](dl/README.md)

### Run code to generate region

- `python src/main.py -res=90 -epsg=5168 -save=json -vis=true`

### Arguments

- `-res`: grid resolution size of DEM (default = 90, unit = meters)
- `-epsg`: epsg code to convert from grid to latitude and longitude (default = 5168)
- `-vis` : visualize figure (options = true / false, default = false).
- `-save`: choose file format to save resulted polygons (options = csv / json, default = None)

## Copyright

All rights reserved by Team MayFlower.
