# FishingAI
![image](https://github.com/ManulProgramming/FishingAI/assets/48217245/efb3778a-db28-4820-ad67-3e005ae82e7c)

FishingAI is a program for recognizing fish species from a photo and displaying the real and predicted price on the market for it. The program itself is written in tkinter, which is needed for a "beautiful" user interface. The model is created using TensorFlow/Keras and an internal MobileNetv2 algorithm.

Training is done in a separate script on A Large Scale Fish Dataset made by O. Ulucan, D. Karakaya, M. Turkan: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset. The trained model is saved in .h5 format, which is then used in the program itself.

Price determination is done through another model in the program, LinearRegression (as there may be too little data for now). LinearRegression was trained using the database of Selina Wamucii's website: https://www.selinawamucii.com/insights/prices/kazakhstan/. The data was obtained by parsing with the Requests/BeautifulSoup library. The trained model is saved in .pkl format, which is also further used in the program (the saving process itself is done through the joblib library instead of pickle, because pickle is less secure).

## Project Creation Process:

- **Week 11: Training a model to identify fish from a picture.**

Writing code using tensorflow/keras to load and process pictures of different fish species based on GT (ground truth) blanks. All pictures go through additional augmentation for deeper learning. There will also be data partitioning for training (75%), validation (10%) and testing (15%). The model will be located as a separate .h5 file.

- **Week 12: Collecting fish sales market data.**

Putting together a program of parsing information about different types of fish and their prices. The information is taken from the Selina Wamucii website, which only provides information for the last week and they are on the website as a regular element of the website. For this reason, a separate parsing program needs to be made with requests/BeautifulSoup. The final table will store 3 columns: name (name of fish species), date (date of certain price), avg_price_kg (average price in tg. per 1 kg.) The generated data is saved in a .csv file as a database.

- **Week 13: Training a model to predict the price of fish.**

The process of compiling code based on the scikit-learn library of training a linear regression model on the compiled database from the previous week. The data is also divided into parts for training (80%) and testing (20%).

- **Week 14: Creating a FishingAI program with a graphical interface.**

Using tkinter, a complete program will be written that will combine two trained models for the intended purpose - a user can upload a photo of a newly caught fish, then the initial model will determine in .h5 format what kind of fish it is and send that information to the second model, which in turn will predict the price today and a year from now for that fish. An additional goal is to retrain the model to predict the price based on new data (prior week for when the user uploaded the photo) so that the prediction is more accurate.

- **Week 15: Program validation and testing**

Validation of the program and these two models for bugs or problems. This will be followed by the process of fixing them.

## Installation:

The application requires Python 3.9, other versions were not tested.

It was also programmed and tested on Native Windows 11, it is unknown if this will work with Linux.

Clone the repository from git clone: [https://github.com/ManulProgramming/FishingAI](https://github.com/ManulProgramming/FishingAI).

Go to the /FishingAI and install requirements using pip:

```bash
pip install -r requirements.txt
```

## Usage:

#### - Main program

Run the primary tkinter app using:

```bash
python FishingAI.py
```

This will launch the GUI with an option to upload the image with a fish and then it will predict what is on the picture and give you information about the market price of it.

#### - Parsing market prices

Run the application parsingdata.py from the folder /FishingAI/Data using this command:

```bash
python Data/parsingdata.py
```

This will parse data from Selina Wamucii's website and automatically clear unnecessary information that is considered to be duplicated. The created .csv file is located in Data/train/fish_weekly_prices.csv

#### - Training market prediction model

Run the application fishmarketmodel_train.py from the folder /FishingAI/Data using this command:

```bash
python Data/fishmarketmodel_train.py
```

This will train a model that will predict a price for every possible (in this repository) type of fish based on the database located in Data/train/fish_weekly_prices.csv. Trained models will be created/replaced in Data/models/fishprice.pkl

#### - Training image model

Run the application fishimagemodel_train.py from the folder /FishingAI/Data using this command:

```bash
python Data/fishimagemodel_train.py
```

This will run a console application with the information on what is happening on the screen. The model will be trained on the dataset located in Data/train/Fish_Dataset. The process of training might take a long time depending on your machine. it will create an updated model for the prediction of what type of fish is in the picture. The trained model will be located in the Data/models folder named fishimage.h5

## Notes:

This application is created for educational purposes. It should not be considered as a serious tool for fish classification, but rather as a basic Python project.


Author: Dmitriy Boyarkin

## License:

[MIT](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)
