# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import csv
import pandas as pd
import urllib.parse
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_station_info(station_name):
    station_list = pd.read_html("https://e-service.cwb.gov.tw/wdps/obs/state.htm")[0]
    start_index_activated_station = station_list.index[station_list[0] == '站號'].tolist()[0]
    end_index_activated_station = station_list.index[station_list[0] == '二、已撤銷測站：'].tolist()[0]
    activated_station_list = station_list.iloc[start_index_activated_station:end_index_activated_station, 0:7]
    activated_station_list = activated_station_list.dropna()
    index_selected_station = activated_station_list.index[activated_station_list[1] == station_name].tolist()[0]
    selected_station_info = activated_station_list.loc[index_selected_station]
    return (selected_station_info)

def download_climate_data(station_name, start_date, end_date, climate_file_name):
    station_info = get_station_info(station_name)
    station_base_add = "http://e-service.cwb.gov.tw/HistoryDataQuery/"
    station_url = "DayDataController.do?command=viewMain"
    station_num = station_info[0]
    with open(climate_file_name,"w+",newline="",encoding="utf-8") as fp: # for excel visualization, please use 'ANSI', for .txt please use 'utf-8'
        writer=csv.writer(fp)
        date_count = 0
        for i in daterange(start_date,end_date):
            date_str = i.strftime("%Y-%m-%d")
            print("========== Processing %s ============" %date_str)
            url = station_base_add + urllib.parse.quote(station_url +
                                                        "&station=" +
                                                        station_num +
                                                        "&stname=" +
                                                        urllib.parse.quote(station_name) +
                                                        "&datepicker=" +
                                                        date_str, safe = '?&=')
            r=requests.get(url)
            r.encoding= r.apparent_encoding
        
            soup=BeautifulSoup(r.text,"lxml")
            tag_table=soup.find(id="MyTable") #用BeautifulSoup找到table位置
            rows=tag_table.findAll("tr") #找到每個
            for row_i, row in zip(range(len(rows[1:])), rows[1:]):
                if date_count == 0 or row_i > 1:
                    rowList = []
                    for cell_i, cell in zip(range(len(row.findAll(["td","th"]))), row.findAll(["td","th"])):
                        pa_value = cell.get_text().replace("\xa0","").replace(".", "")
                        if row_i > 1 and cell_i == 0:
                            pa_value = date_str.replace('-', '_') + '_' + pa_value
                        rowList.append(pa_value)
                    writer.writerow(rowList)
            date_count = date_count + 1
def first_stage_climate_data_clean(climate_data):
    climate_data_cleaned = climate_data.dropna(axis = 1)
    return climate_data_cleaned

def first_stage_detection_data_clean(detection_data):
    detection_data_cleaned = detection_data.dropna(how = 'all') # remove the nan part
    pd_time = detection_data_cleaned['time'].str.replace('/','_')
    pd_time = pd_time.str.replace(' ','_')
    detection_data_cleaned['time'] = pd_time.str[0:13].str.replace(':', '')
    return detection_data_cleaned
                    
def main():
    start_date = datetime(2019, 10, 1)
    end_date = datetime(2019, 12, 1)
    station_name = "芬園"
    detection_filename = 'database.csv'
    
    
    climate_filename = 'climate_data_' + start_date.strftime("%Y_%m_%d") + '_' + end_date.strftime("%Y_%m_%d") + '.csv'
    if not os.path.exists(climate_filename):
        download_climate_data(station_name, start_date, end_date, climate_filename)
    climate_data = pd.read_csv(climate_filename)
    climate_data_first_cleaned = first_stage_climate_data_clean(climate_data)
    
    
    detection_data = pd.read_csv(detection_filename, encoding = 'ANSI') # the detection file is encoded in "ANSI"
    detection_data_first_cleaned = first_stage_detection_data_clean(detection_data)
    
    print('Finished')
    
if __name__ == "__main__":
    main()