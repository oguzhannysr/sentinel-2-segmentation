import numpy as np 
import math
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as ss
import geopandas
import hydroeval
from pyproj import Geod
from shapely import wkt
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import euclidean_distances

def ShapeIntersectSegments(vectorfilePath, segmentsfilePath)
    referans = geopandas.read_file(vectorfilePath) 
    segmentler = geopandas.read_file(segmentsfilePath)
    referans = referans.to_crs(32636)  
    referans['Alan'] = 0
    segmentler['Alan'] = 0
    segment_wkt = segmentler.geometry.to_wkt()
    referans_wkt = referans.geometry.to_wkt()

    for i in range(0,len(referans)):
        geod = Geod(ellps="WGS84")
        referans2 = referans.to_crs(4326) 
        referans_wkt2 = referans2.geometry.to_wkt()
        calc = referans_wkt2[i]
        poly = wkt.loads(calc)
        alan = abs(geod.geometry_area_perimeter(poly)[0])
        referans['Alan'][i] = alan 
    segment2 = segmentler.to_crs(4326) 
    segment_wkt2 = segment2.geometry.to_wkt()
    for i in range(0,len(segmentler)):
        geod = Geod(ellps="WGS84")
        calc = segment_wkt2[i]
        poly = wkt.loads(calc)
        alan = abs(geod.geometry_area_perimeter(poly)[0])
        segmentler['Alan'][i] = alan 
    olculer = geopandas.GeoDataFrame(columns=['Segment_Id','Int/Referans','Segment Area',
                                              'Referans Area','Int Area','QR','AFI','OS'
                                              ,'US','RMS','EVS','Max Error','MAE','MSE',
                                              'MedAE','R2','MTD','D2','MPL','Korelasyon',
                                              'Cossim','NSE','MinDist','row_id'],index=referans.index)
    
    liste=[]
    id_no = []
    intersection_area=[]
    i_a = []
    poly2_alan = []
    j=0
    while j<len(referans):
        for i in range(0,len(segmentler)):
            calc1=referans_wkt[j]
            poly1 = wkt.loads(calc1)
            calc2 = segment_wkt[i]
            poly2 = wkt.loads(calc2) 
            intersection = poly1.intersection(poly2)
            if intersection.area > 0:
                print ('Int/Referans: ', intersection.area/poly1.area*100,'%')
                print(intersection.area)
                
                liste.append(intersection.area)
                id_no.append(i)
                i_a.append(intersection.area/poly1.area*100)
                poly2_alan.append(poly1.area)        
            if i==len(segmentler)-1 :
                print(j)
    
                eklenecek = max(i_a)
                olculer.iloc[j,1]=eklenecek
                olculer.iloc[j,4]=(i_a[np.argmax(liste)]*poly2_alan[np.argmax(i_a)])/100
                intersection_area.append(id_no[np.argmax(liste)])
                olculer.iloc[j,0]=intersection_area[0]
                del liste[:]
                del id_no[:]
                del intersection_area[:]
                del i_a[:]
                del poly2_alan[:]
                j= j+1
                print("Başarılı")  
    olculer.iloc[:,3]=referans['Alan']
    for k,w in zip((olculer['Segment_Id']),range(0,len(referans))):
        olculer.iloc[w,2] = segmentler.iloc[k,2] 
        
    olculer['row_id']=referans['_ROWID_']    
    
def segmentationErrorMetrics(j):    
    "QR,AFI,OS,US,RMS"
    OS = 1-((olculer.iloc[j,4])/olculer.iloc[j,3])
    olculer.iloc[j,7] = OS
    US = 1-((olculer.iloc[j,4])/olculer.iloc[j,2])
    olculer.iloc[j,8] = US
    RMS = math.sqrt((US**2 + OS**2)/2)
    olculer.iloc[j,9] = RMS
    AFI = ((olculer.iloc[j,3]-olculer.iloc[j,2])/olculer.iloc[j,3])
    olculer.iloc[j,6] = AFI
    QR = ((olculer.iloc[j,4])/(olculer.iloc[j,2]+olculer.iloc[j,3]-olculer.iloc[j,4]))
    olculer.iloc[j,5] = QR
    
for i in range(0,len(referans)):
    segmentationErrorMetrics(i)

dogruluk_matrisi=np.zeros((len(referans),5))
dogruluk_matrisi[:,0]=1
dogruluk_matrisi[:,1:5]=0
metrik_matrisi = olculer.iloc[:,5:10]
metrik_matrisi = metrik_matrisi.to_numpy()
metrik_matrisi=metrik_matrisi.astype('float64')

intersection = np.logical_and(olculer["Referans Area"][0], olculer["Segment Area"][0])
union = np.logical_or(olculer["Referans Area"][0], olculer["Segment Area"][0])
iou_score = np.sum(intersection) / np.sum(union) 


for i in range(0,len(dogruluk_matrisi)):
    
    evs = metrics.explained_variance_score(dogruluk_matrisi[i], metrik_matrisi[i])
    max_error = metrics.max_error(dogruluk_matrisi[i], metrik_matrisi[i])
    mae = metrics.mean_absolute_error(dogruluk_matrisi[i], metrik_matrisi[i])
    mse = metrics.mean_squared_error(dogruluk_matrisi[i], metrik_matrisi[i])
    medAE = metrics.median_absolute_error(dogruluk_matrisi[i], metrik_matrisi[i])
    #mape = metrics.mean_absolute_percentage_error(dogruluk_matrisi[i], metrik_matrisi[i])
    r2 = metrics.r2_score(dogruluk_matrisi[i], metrik_matrisi[i])
    mtd = metrics.mean_tweedie_deviance(dogruluk_matrisi[i], metrik_matrisi[i])
    d2 = metrics.d2_tweedie_score(dogruluk_matrisi[i], metrik_matrisi[i])
    mpl = metrics.mean_pinball_loss(dogruluk_matrisi[i], metrik_matrisi[i])
    min_dist = euclidean_distances(dogruluk_matrisi[i].reshape(1, -1), metrik_matrisi[i].reshape(1, -1))
    min_dist = float(min_dist)
    olculer.iloc[i,10] = evs
    olculer.iloc[i,11] = max_error
    olculer.iloc[i,12] = mae
    olculer.iloc[i,13] = mse
    olculer.iloc[i,14] = medAE
    #olculer.iloc[i,15] = mape
    olculer.iloc[i,15] = r2
    olculer.iloc[i,16] = mtd
    olculer.iloc[i,17] = d2
    olculer.iloc[i,18] = mpl
    olculer.iloc[i,22] = min_dist    
