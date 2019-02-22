# Titanic-Survival-records
Machine learning projects

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

Steps for survival prediction :

To start of with the survival prediction, dataset was downloaded from kaggle site

All the necessary libraries are imported

Data has to be cleaned

Columns with null values has to be taken care of. For example column cabin has maximum null values. By plotting the graph we can find out that cabin feature has least impact on survival rate. Hence we can drop the column.

Name and Passengerid columns can be dropped for its least contribution in prediction

One hot encoder can be used for categorical data(Gender  and  Embarked)

Dataset is divided into train and test set using cross validation

Various classifiers can be applied on train and test data to check for the accuracy score
