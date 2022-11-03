import numpy as np 
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift
import time

def segmentation(rasterPath, outputPath)

    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(rasterPath)
    nbands = naip_ds.RasterCount
    band_data = []
    print('bands', naip_ds.RasterCount, 'rows', naip_ds.RasterYSize, 'columns',
          naip_ds.RasterXSize)
    for i in range(1, nbands+1):
        band = naip_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(band)
    band_data = np.dstack(band_data)
    img = exposure.rescale_intensity(band_data)

    seg_start = time.time()
    # segments = quickshift(img, convert2lab=False)
    # segments = quickshift(img, ratio=0.8, convert2lab=False)
     
    segments = quickshift(img, ratio=0.99, max_dist=5, convert2lab=False)
    # segments = slic(img, n_segments=100000, compactness=0.1)
    # segments = slic(img, n_segments=500000, compactness=0.01)
    # segments = slic(img, n_segments=500000, compactness=0.1)
    print('segments complete', time.time() - seg_start)

    segmentsOut = driverTiff.Create(outputPath, naip_ds.RasterXSize, naip_ds.RasterYSize,
                                    1, gdal.GDT_Float32)
    segmentsOut.SetGeoTransform(naip_ds.GetGeoTransform())
    segmentsOut.SetProjection(naip_ds.GetProjectionRef())
    segmentsOut.GetRasterBand(1).WriteArray(segments)
    segmentsOut = None
