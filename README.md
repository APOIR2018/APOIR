# APOIR

This is a python implementation of the model which is called APOIR -- a POI recommendation model using adversarial training.


--------------


## Requirements

- python2.7
- tensorflow >= 1.3.0
- numpy >= 1.14.0


------------------



## Datasets

my datasets

| Datasets        |    Users    | POIs  |Check-ins |
| ------------- |:-------------:| :-----:|:--------:|
| Gowalla     |18,737|32,510|1,278,274|
| Foursquare     |24,941|28,593|1,196,248|
| Yelp  |30,887|18,995|860,888|

For Gowalla dataset, we ﬁlter out those users with fewer than 15 check-in POIs and those POIs with fewer than 10 visitors. For Foursquare and Yelp, we discard those users with fewer than 10 check-in POIs and those POIs with fewer than 10 visitors. We partition each dataset into training set and test set. For each user, we use the earlier 75% check-ins as the training data and the most recent 25% check-ins as the test data. All datasets are very sparse (the frequency of most POIs being visited is extremely low).



--------------------------

## Baseline

- USG
- MGMPFM
- LFBCA
- iGSLR
- LORE
- IRenMF
- GeoMF
- RankGeoFM
You can find all the above code at http://spatialkeyword.sce.ntu.edu.sg/eval-vldb17/
- GeoTeaser
You can find this code at https://github.com/shenglin1987/geo_teaser
- PACE
You can find this code at https://github.com/yangji9181/PACE2017



--------------------



## Usage

To run APOIR, first clone the code to your python IDE (eg:Pycharm), then run the main.py.


