
import pandas as pd
#import pandas.Series.dt as dt
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


print("Reading dataset.")
#Reading the train and test data from the file and storing it in an object
trainDataAug = pd.read_csv('uber-raw-data-aug14.csv', sep=',',header=0,keep_default_na=False)
trainDataSep = pd.read_csv('uber-raw-data-sep14.csv', sep=',',header=0,keep_default_na=False)
trainDataLyft = pd.read_csv('other-Lyft.csv', sep=',',header=0,keep_default_na=False)

trainDataAug = trainDataAug.dropna()
trainDataSep = trainDataSep.dropna()
trainDataLyft = trainDataLyft.dropna()
print(trainDataAug.shape)
print(trainDataSep.shape)
print(trainDataLyft.shape)


trainData = trainDataAug.append(trainDataLyft).append(trainDataSep)
#trainData.append(trainDataSep)
#trainData.append(trainDataLyft)
trainData["taxi_code"] = trainData['Taxi Service'].map({'Uber':1,'Lyft':2})
trainData = trainData.dropna()


trainData['Date/Time'] = pd.to_datetime(trainData['Date/Time'], infer_datetime_format =True)
#print(trainData['Date/Time'][1])
trainData['hour'] = trainData['Date/Time'].dt.hour
plttrainData = trainData
#print(trainData['hour'][1])

X = trainData[["Lat","Lon","hour"]]
y = trainData[["taxi_code"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

from sklearn.cluster import KMeans

for k in [2]:
    uberCluster = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=1000)
    print(uberCluster)
    model=uberCluster.fit(X_train, y_train)
    predictions = model.predict(X_test)
    #pred_cluster_centers = [uberCluster.cluster_centers_[i] for i in predictions]
    centroids = uberCluster.cluster_centers_
    print("Centroid: " , centroids)
    labels = uberCluster.labels_
    print("Labels: ", labels)
    #print("Score: " + model.score(predictions))
    print("Accuracy: ")
    print( accuracy_score(predictions,y_test))
    print("Confusion matrix: ")
    print(confusion_matrix(predictions,y_test))
    







import matplotlib.pyplot as plt

trainDataAug["taxi_code"] = trainDataAug['Taxi Service'].map({'Uber':1,'Lyft':2})
trainDataAug['Date/Time'] = pd.to_datetime(trainDataAug['Date/Time'], infer_datetime_format =True)
#print(trainDataAug['Date/Time'][1])
trainDataAug['hour'] = trainDataAug['Date/Time'].dt.hour
            
            
trainDataSep["taxi_code"] = trainDataSep['Taxi Service'].map({'Uber':1,'Lyft':2})
trainDataSep['Date/Time'] = pd.to_datetime(trainDataSep['Date/Time'], infer_datetime_format =True)
#print(trainDataAug['Date/Time'][1])
trainDataSep['hour'] = trainDataSep['Date/Time'].dt.hour
            
#print(trainDataAug['hour'][1])

trainDataLyft["taxi_code"] = trainDataLyft['Taxi Service'].map({'Uber':1,'Lyft':2})
trainDataLyft['Date/Time'] = pd.to_datetime(trainDataLyft['Date/Time'], infer_datetime_format =True)
#print(trainDataLyft['Date/Time'][1])
trainDataLyft['hour'] = trainDataLyft['Date/Time'].dt.hour
             

#print(trainDataLyft['hour'][1])

uberplot = trainDataAug[["Lat","Lon","taxi_code"]]
lyftplot = trainDataLyft[["Lat","Lon","taxi_code"]]



plt.title('Distribution of traffic in New York City as per Lyft and Uber')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
uberplt = plt.scatter(plttrainData["Lat"][plttrainData['taxi_code'] == 1],plttrainData["Lon"][plttrainData['taxi_code'] == 1],color='r',label="Uber", s=2)
lyftplt = plt.scatter(plttrainData["Lat"][plttrainData['taxi_code'] == 2],plttrainData["Lon"][plttrainData['taxi_code'] == 2],color='b',label="Lyft", s=2)
plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='white', zorder=10)
plt.legend((uberplt, lyftplt), ('Uber','Lyft'), scatterpoints=1,loc='top left', ncol=1,fontsize=8)
text = iter(['Uber', 'Lyft'])



plt.show()

