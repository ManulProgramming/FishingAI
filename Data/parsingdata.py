import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
import csv
'''import os
try:
    os.remove('train/fish_weekly_prices.csv')
except:
    None'''
d = datetime.today() - timedelta(days=7)
days_of_week={0:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},1:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},2:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},3:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},4:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},5:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999},6:{"Shrimp":-999,"Sea Bass":-999,"Sea Bream":-999,"Hourse Mackerel":-999,"Trout":-999}}
fulldatecolumn=[]
products=['shrimps-prawns','sea-bass','sea-bream','mackerel','trout']
try:
    with open('train/fish_weekly_prices.csv','r',newline='',encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            fulldatecolumn+=[row[1]]
            if row[2]!='avg_price_kg':
                days_of_week[datetime(int(row[1].split('/')[2]), int(row[1].split('/')[1]), int(row[1].split('/')[0])).weekday()][row[0]]=float(row[2])
        print(days_of_week)
except:
    fulldatecolumn=['none','none','none']

with open('train/fish_weekly_prices.csv','a',newline='',encoding='utf-8') as file:
    writer = csv.writer(file)
    if fulldatecolumn[0]!='date':
        writer.writerow(['name','date','avg_price_kg'])
    for i in range(1,8):
        bb="20"+d.strftime("%x").split('/')[2]+'-'+d.strftime("%x").split('/')[0]+'-'+d.strftime("%x").split('/')[1]
        print(d.strftime("%x").split('/')[1]+'/'+d.strftime("%x").split('/')[0]+'/'+"20"+d.strftime("%x").split('/')[2])
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
                print(days_of_week[d.weekday()][name_of_product],avg_price)
                time.sleep(0.1)
                continue
            writer.writerow([name_of_product,d.strftime("%x").split('/')[1]+'/'+d.strftime("%x").split('/')[0]+'/'+"20"+d.strftime("%x").split('/')[2],avg_price])
            time.sleep(0.1)
        d=datetime.today() - timedelta(days=7-i)
