# Detecting-ddos-Attacks-using-Machine-Learning

These programs are part of my Thesis: Detecting Distributed Denial of Service Attacks using Machine Learning Algorithms.

The algorithms I have tested so far are: Random Forest, Decision Trees, K-Nearest Neighbour (K=3 and K=5), Logistic Regression.
I am still experimenting with other ones like Hist Gradient Boosting.
All of the above algoritmhs are being tested with and without cross validation.

I am using the python library scikit-learn.
In addition to the main library scikit-learn, to run the programs you will need the libraries: pandas, matplotlib, time.

You will find all the updated code in the master branch!

About the dataset I am using:

I found the dataset at https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset

Modification:
--I balanced it with the oversampling method (balance_dataset.py). 
--I replaced the non-numeric fields in the dataset with numeric fields (config_data1.py)		
  The values of the src field containing the IP addresses of the transmitter in the format 10.0.0.X became X.
  Similarly in the dst field which has the addresses of the receiver.
  The Protocol field contains the protocol used in each case. This will be UDP or TCP or ICMP which became 1,2,3 respectively
--I replaced the empty fields (NaN) of the dataset with 0 using the Imputing method (config_data2.py)

The final dataset is final_dataset.csv
