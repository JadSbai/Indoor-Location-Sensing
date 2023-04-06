#Passive Indoor Location Sensing using WiFi

This project is aimed at developing an innovative approach to passive indoor location sensing using WiFi technology. The main objective of the project is to obtain granular level view of customers behavior in the context of the retail industry.

Before running the program, you must have the following installed on your machine:

    Python 3
    Required packages (see requirements.txt)

Installation

    Clone the repo

sh

git clone https://github.com/your_username_/Project-Name.git

    Install the required packages

sh

pip install -r requirements.txt

Usage

To run the program, navigate to the root directory called "Position Predictor" and run the following command:

sh

python main.py

This will execute the program and print the scores of the Baseline random forest regressor as well as the K-means algorithm used as part of the custom model and the Random Forest classifier. The final output will be a plot of the predicted positions by the Random Forest regressor and the custom model against the target positions.
##Experimental Setup

Section 6 of the accompanying paper outlines the experimental setup used in this project, including a critical evaluation and interpretation of the performance results obtained from comparative tests.

###License

This project is licensed under the MIT License - see the LICENSE file for details.