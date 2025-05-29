# Detecting-ddos-Attacks-using-Machine-Learning

This project is part of my Thesis: Detecting Distributed Denial of Service Attacks using Machine Learning Algorithms.

The algorithms I have tested so far are: Random Forest, Decision Tres, Light Gradient Boosting, Neural Netwroks, TabNet.

All of the above algoritmhs are tested 2 times, with and without feature scaling in order to compare. Also all tests are done with 5 fold cross validation.

I am using the python library scikit-learn and PyTorch.
In addition to the main library scikit-learn, to run the programs you will need the libraries: pandas, matplotlib, numpy.

The dataset being used is: https://www.kaggle.com/datasets/abdussalamahmed/lr-hr-ddos-2024-dataset-for-sdn-based-networks

Modification:
--I balanced it with the oversampling method (balance_dataset.py).	
--I replaced the empty fields (NaN) of the dataset with 0 using the Imputing method (config_data2.py)

The final dataset is final_dataset.csv
