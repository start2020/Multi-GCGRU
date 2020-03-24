[toc]

# Raw Datasets

It contains all the raw datasets downloaded from a public financial API (http://tushare.org/). It is composed of two stock indexes CSI300 and CSI500 in Chinese market, of which their version is in 2015. Each stock has features including prices, concepts, industry, shareholders.

The description of each raw data file is as follow:

**all stock prices**: This document contains 13 prices features of all the listed corporations which still exist in 2019-12. The prices features ranges from 2017-05 and end at 2019-12. CSI300 and CSI500 are part of these stocks.

**CSI300 list 2015. xlsx**: It contains the stocks chosen for CSI300 in 2015 version.

**CSI500 list 2015.xlsx**: It contains the stocks chosen for CSI500 in 2015 version.

**CSI300 concepts.xIsx** : It contains the concepts information of each stock in CSI300.

**CSI500 concepts.xIsx**: It contains the concepts information of each stock in CSI500.

**CSI300 shareHolders.xIsx**: It contains the shareholders information of each stock in CSI300.

**CSI500 shareholders.xIsx**: It contains the shareholders information of each stock in CSI500.







# Model Input

This document contains all the datas which have been preprocessed and can be fed into the model.

**CSI300-features.xIsx:** the prices features of CSI300 

**CSI500-features.xIsx:** the prices features of CSI500 

**CSI300-concept-before normalization.xIsx:** the matrix of CSI300 and each entry represents number of shared concept between two stocks.
**CSl300-concept-matrix.xIsx:** the concept matrix of CSI300 which is normalized.

**CSl500-concept-before normalization.xIsx:** the matrix of CSI500 and each entry represents number of shared concept between two stocks.
**CSl500-concept-matrix.xIsx:** the concept matrix of CSI500 which is normalized.

**CSl300-industry-matrix.xIsx:** the industry matrix of CSI300 and each entry represents the industry based influence strength between two stocks.

**CSl500-industry-matrix. xlsx: **the industry matrix of CSI500 and each entry represents the industry based influence strength between two stocks.

**CSl300-shareholder-matrix. xlsx:** the shareholder matrix of CSI300 and each entry represents the shareholder ratio between parent company and subsidiary company.

**CSl500-shareholder-matrix.xlsx:** the shareholder matrix of CSI500 and each entry represents the shareholder ratio between parent company and subsidiary company.




