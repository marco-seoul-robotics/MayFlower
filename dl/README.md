Available Open access at: http://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html

## Dataset Access

The dataset is available for access through Google Cloud Storage bucket at: `gs://senfloods11/`

You can access the dataset bucket using the [gsutil](https://cloud.google.com/storage/docs/gsutil) command line tool. If you would like to download the entire dataset (~14 GB) you can use `gsutil rsync` to clone the bucket to a local directory. The `-m` flag is recommended to speed downloads. The `-r` flag will download sub-directories and folder recursively. See the example below.

```bash
$ gsutil -m rsync -r gs://sen1floods11 /YOUR/LOCAL/DIRECTORY/HERE
```

If using an example notebook, you can download the dataset to the folder that notebooks expect it to be in by running

```bash
$ mkdir /home/files
$ gsutil -m rsync -r gs://sen1floods11 /home/files
```

## Bucket Structure

The `sen1floods11` bucket is split into subfolders containing data, checkpoints, training/testing splits, and a [STAC](https://stacspec.org/) compliant catalog. More detail on each is provided in the docs README.

## Dataset Information

Each file follows the naming scheme EVENT_CHIPID_LAYER.tif (e.g. `Bolivia_103757_S2Hand.tif`). Chip IDs are unique, and not shared between events. Events are named by country and further information on each event (including dates) can be found in the event metadata below. Each layer has a separate GeoTIFF, and can contain multiple bands in a stacked GeoTIFF. All images are projected to WGS 84 (`EPSG:4326`) at 10 m ground resolution.

| Layer | Description                                                                                                                                              | Values                                                  | Format                                           | Bands                                                                                                                                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QC    | Hand labeled chips containing ground truth                                                                                                               | -1: No Data / Not Valid <br> 0: Not Water <br> 1: Water | GeoTIFF <br> 512 x 512 <br> 1 band <br> Int16    | 0: QC                                                                                                                                                                                                                                                                           |
| S1    | Raw Sentinel-1 imagery. <br> IW mode, GRD product <br> See [here](https://developers.google.com/earth-engine/sentinel1) for information on preprocessing | Unit: dB                                                | GeoTIFF <br> 512 x 512 <br> 2 bands <br> Float32 | 0: VV <br> 1: VH                                                                                                                                                                                                                                                                |
| S2    | Raw Sentinel-2 MSI Level-1C imagery <br> Contains all spectral bands (1 - 12) <br> Does not contain QA mask                                              | Unit: TOA reflectance <br> (scaled by 10000)            | GeoTIFF <br> 512 x 512 <br> 13 bands <br> UInt16 | 0: B1 (Coastal) <br> 1: B2 (Blue) <br> 2: B3 (Green) <br> 3: B4 (Red) <br> 4: B5 (RedEdge-1) <br> 5: B6 (RedEdge-2) <br> 6: B7 (RedEdge-3) <br> 7: B8 (NIR) <br> 8: B8A (Narrow NIR) <br> 9: B9 (Water Vapor) <br> 10: B10 (Cirrus) <br> 11: B11 (SWIR-1) <br> 12: B12 (SWIR-2) |


## How to run this code

```
python train.py 
```

And wait for the checkpoint. Download sat image ( sentinel-1 image) and put it into the models.

```
python infer.py 
```


