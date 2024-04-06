import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
import csv

def parsing_training():
    print("Parsing...")
    d = datetime.today() - timedelta(days=7)
    days_of_week={0:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},1:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},2:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},3:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},4:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},5:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},6:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999}}
    fulldatecolumn=[]
    products=['shrimps-prawns','sea-bass','sea-bream','mackerel','trout']
    try:
        with open('Data/train/fish_weekly_prices.csv','r',newline='',encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                fulldatecolumn+=[row[1]]
                if row[2]!='avg_price_kg':
                    days_of_week[datetime(int(row[1].split('/')[2]), int(row[1].split('/')[1]), int(row[1].split('/')[0])).weekday()][row[0]]=float(row[2])
    except:
        fulldatecolumn=['none','none','none']

    with open('Data/train/fish_weekly_prices.csv','a',newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        if fulldatecolumn[0]!='date':
            writer.writerow(['name','date','avg_price_kg'])
        for i in range(1,8):
            bb="20"+d.strftime("%x").split('/')[2]+'-'+d.strftime("%x").split('/')[0]+'-'+d.strftime("%x").split('/')[1]
            if d.strftime("%x").split('/')[1]+'/'+d.strftime("%x").split('/')[0]+'/'+"20"+d.strftime("%x").split('/')[2] in fulldatecolumn:
                d=datetime.today() - timedelta(days=7-i)
                continue
            for product in products:
                a=requests.get(f'https://www.selinawamucii.com/historical-prices/kazakhstan/{product}/{bb}/')
                data=str(BeautifulSoup(a.text,'html.parser').find('div',{"class":"row"})).split('\n')
                min_price,max_price=re.findall('<td>(\d+\.*\d*)</td>',data[14])[0],re.findall('<td>(\d+\.*\d*)</td>',data[19])[0]
                avg_price=(float(min_price)+float(max_price))/2
                name_of_product = 'Shrimp' if product=='shrimps-prawns' else 'Sea Bass' if product=='sea-bass' else 'Sea Bream' if product=='sea-bream' else 'Hourse Mackerel' if product=='mackerel' else 'Trout'
                if days_of_week[d.weekday()][name_of_product]==round(avg_price,3):
                    continue
                writer.writerow([name_of_product,d.strftime("%x").split('/')[1]+'/'+d.strftime("%x").split('/')[0]+'/'+"20"+d.strftime("%x").split('/')[2],avg_price])
            d=datetime.today() - timedelta(days=7-i)
    print("Training...")
    data = pd.read_csv('Data/train/fish_weekly_prices.csv')
    data['date'] = pd.to_datetime(data['date'],dayfirst=True)
    data['date'] = (data['date'] - pd.to_datetime('22/3/2024',dayfirst=True)) / pd.Timedelta(days=1)
    data=pd.get_dummies(data)
    X = data.drop('avg_price_kg',axis=1)
    y = data['avg_price_kg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Test:",r2_score(y_test, predictions))
    predictions = model.predict(X_train)
    print("Train:",r2_score(y_train, predictions))
    joblib.dump(model, 'Data/models/fishprice.pkl')
loaded_model_image = tf.keras.models.load_model("Data/models/fishimage.h5")
loaded_model_market = joblib.load("Data/models/fishprice.pkl")
def upload_image():
    filename = filedialog.askopenfilename()
    if not filename:
        return
    img = Image.open(filename)
    tk_img = ctk.CTkImage(light_image=img,dark_image=img,size=(300,300))
    image_label.configure(image=tk_img, text="")
    print("Image uploaded:", filename)
    test_image_path=filename
    test_image = image.load_img(test_image_path, target_size=(224, 224, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = tf.keras.applications.mobilenet_v2.preprocess_input(test_image)
    prediction = loaded_model_image.predict(test_image)
    predicted_class = np.argmax(prediction)
    labels = {0: 'Gilt-Head Bream', 1: 'Hourse Mackerel', 2: 'Red Sea Bream', 3: 'Sea Bass', 4: 'Shrimp', 5: 'Trout'}
    final_text=labels[predicted_class]+'\n'+str(round(prediction[0][predicted_class]*100,2))+'%'
    if prediction[0][predicted_class]*100<50:
        final_text+='\nPrediction might\nbe wrong!'
    result_label1.configure(text=final_text)
    if checkbox.get()==1:
        messagebox.showinfo(title="AI learning is starting!", message="Please wait! It might take a long time!")
        parsing_training()
    date_today=datetime.today()
    new_data={'date':[date_today.strftime("%x").split('/')[1]+'/'+date_today.strftime("%x").split('/')[0]+'/'+"20"+date_today.strftime("%x").split('/')[2]],'name_Hourse Mackerel':[True if predicted_class==2 else False],'name_Sea Bass':[True if predicted_class==5 else False],'name_Sea Bream':[True if predicted_class==4 or predicted_class==1 else False],'name_Shrimp':[True if predicted_class==6 else False],'name_Trout':[True if predicted_class==8 else False]}
    super_new_data=pd.DataFrame(data=new_data)
    super_new_data['date'] = pd.to_datetime(super_new_data['date'],dayfirst=True)
    super_new_data['date'] = (super_new_data['date'] - pd.to_datetime('22/3/2024',dayfirst=True)) / pd.Timedelta(days=1)
    prediction = loaded_model_market.predict(super_new_data)
    result_label2.configure(text=str(round(prediction[0],2)))
    date_today=datetime.today()+timedelta(days=30)
    new_data={'date':[date_today.strftime("%x").split('/')[1]+'/'+date_today.strftime("%x").split('/')[0]+'/'+"20"+date_today.strftime("%x").split('/')[2]],'name_Hourse Mackerel':[True if predicted_class==2 else False],'name_Sea Bass':[True if predicted_class==5 else False],'name_Sea Bream':[True if predicted_class==4 or predicted_class==1 else False],'name_Shrimp':[True if predicted_class==6 else False],'name_Trout':[True if predicted_class==8 else False]}
    super_new_data=pd.DataFrame(data=new_data)
    super_new_data['date'] = pd.to_datetime(super_new_data['date'],dayfirst=True)
    super_new_data['date'] = (super_new_data['date'] - pd.to_datetime('25/3/2024',dayfirst=True)) / pd.Timedelta(days=1)
    prediction = loaded_model_market.predict(super_new_data)
    result_label3.configure(text=str(round(prediction[0],2)))
app = ctk.CTk()
app.geometry("600x650")
app.title("FishingAI")
frame_top = ctk.CTkFrame(master=app)
frame_top.pack(pady=20, padx=10, fill="both")
image_label = ctk.CTkLabel(master=frame_top, width=300, height=300, text="No image")
image_label.pack(side="top")
upload_button = ctk.CTkButton(master=frame_top, text="UPLOAD IMAGE", command=upload_image)
upload_button.pack(pady=5)
checkbox = ctk.CTkCheckBox(master=frame_top, text="Retrain AI on up-to-date data")
checkbox.pack(pady=10)
frame_bottom = ctk.CTkFrame(master=app)
frame_bottom.pack(pady=20, padx=10, fill="both", expand=True)
labels_text = ["Type of fish:", "Current average price:", "Predicted Price (after 30 days):"]
for i, text in enumerate(labels_text):
    label = ctk.CTkLabel(master=frame_bottom, text=text,width=172)
    label.grid(row=0, column=2*i, padx=10, pady=10, sticky="w")
result_label1 = ctk.CTkLabel(master=frame_bottom, text="", font=('Arial',20,'bold'))
result_label1.grid(row=1, column=0, padx=10, pady=10)
result_label2 = ctk.CTkLabel(master=frame_bottom, text="", font=('Arial',20,'bold'))
result_label2.grid(row=1, column=2, padx=10, pady=10)
result_label3 = ctk.CTkLabel(master=frame_bottom, text="", font=('Arial',20,'bold'))
result_label3.grid(row=1, column=4, padx=10, pady=10)
app.resizable(False, False)
app.mainloop()
