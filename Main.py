# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def download_climate_data(station_base_add, start_date, end_date):
    for i in range(start_date,end_date,1):
        url="2019-10-0"+ str(i)
        r=requests.get(url)
        r.encoding= r.apparent_encoding
    
        soup=BeautifulSoup(r.text,"lxml")
        tag_table=soup.find(id="MyTable") #用BeautifulSoup找到table位置
        rows=tag_table.findAll("tr") #找到每個
        csvfile="test_2019/10/0"+str(i)+".csv" #開個csv檔案準備寫入
        with open(csvfile,"w+",newline="",encoding="utf-8") as fp:
            writer=csv.writer(fp)
            for row in rows:
                rowList=[]
                for cell in row.findAll(["td","th"]):
                    rowList.append(cell.get_text().replace("\n","").replace("\r",""))
                    writer.writerow(rowList)
                    
def main():
    start_date = datetime(2020, 5, 1)
    end_date = datetime(2020, 5, 30)
    station_base_add = "http://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=C0G620&stname=%25E8%258A%25AC%25E5%259C%2592&datepicker="
    download_climate_data(station_base_add, start_date, end_date)
    print('Finished')
    
if __name__ == "__main__":
    main()