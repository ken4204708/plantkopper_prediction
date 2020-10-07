# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import requests
import csv
import pandas as pd
import numpy as np
import urllib.parse
import os
import matplotlib.pyplot as plt
import math
import pickle
import argparse
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score


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
    climate_data_cleaned = climate_data_cleaned.rename(columns = {"觀測時間(hour)": "time", 
                                           "測站氣壓(hPa)": "air_pressure", 
                                           "氣溫(℃)": "temperature", 
                                           "相對溼度(%)": "humidity", 
                                           "風速(m/s)": "wind_speed", 
                                           "風向(360degree)": "wind_direction", 
                                           "降水量(mm)\t": "rainfall"})
    for column in climate_data_cleaned.columns:
        df_temp = climate_data_cleaned[column].iloc[1:].reset_index()
        df_temp = df_temp[column]
        if (column != 'time'):
            df_temp = df_temp.replace('/', np.NaN)
            df_temp = df_temp.replace('X', np.NaN)
            df_temp = df_temp.astype(float)
            df_temp = df_temp.interpolate(method = 'nearest')
        climate_data_cleaned[column] = df_temp
    return climate_data_cleaned

def get_time_from_data(pd_time):
    pd_time = pd_time.str.replace('/','_')
    pd_time = pd_time.str.replace(' ','_')
    pd_time = pd_time.apply(lambda x: x[:8] + '0' + x[8:] if x[9]=='_' else x) # replace the date number if the date is only one digit    
    time = pd_time.str[0:13].str.replace(':', '')
    return time

def first_stage_detection_data_clean(detection_data):
    detection_data_cleaned = detection_data.dropna(how = 'all') # remove the nan part
    pd_time = detection_data_cleaned['time'].str.replace('/','_')
    pd_time = pd_time.str.replace(' ','_')
    pd_time = pd_time.apply(lambda x: x[:8] + '0' + x[8:] if x[9]=='_' else x) # replace the date number if the date is only one digit
    detection_data_cleaned['time'] = pd_time.str[0:13].str.replace(':', '')
    flag_longitude_exist = ~detection_data_cleaned['longitude'].isna()
    detection_data_GPSExist = detection_data_cleaned.loc[flag_longitude_exist]
    flag_latitude_exist = ~detection_data_GPSExist['latitude'].isna()
    detection_data_GPSExist = detection_data_GPSExist.loc[flag_latitude_exist]
    return detection_data_GPSExist

def second_stage_detection_data_clean(detection_data, station_name): # remove the data far from the object station
    station_info = get_station_info(station_name)
    station_longtitude = float(station_info[3])
    station_latitude = float(station_info[4])
    pd_longitude = detection_data['longitude']
    pd_latitude = detection_data['latitude']
    diff_longtitude = np.array(pd_latitude - station_longtitude)
    diff_latitude = np.array(pd_longitude - station_latitude)
    diff_GPS = np.power(np.power(diff_longtitude, 2)+np.power(diff_latitude, 2), 0.5)
    detection_data['distance'] = pd.DataFrame(diff_GPS)
    detection_data = detection_data.loc[detection_data['distance'] < 0.1]
    detection_data_selected = detection_data.loc[:, ['time', 'amount']]
    detection_data_avg = detection_data_selected.groupby('time').mean()
    return detection_data_avg

def combine_data_climate_detection(detection_data_second_cleaned, climate_data_first_cleaned, predict_date):
    df_temp = climate_data_first_cleaned.merge(detection_data_second_cleaned, on=['time'], how='left')    
    df_amount_interpolated = df_temp['amount'].interpolate(method = 'quadratic')
    df_amount_interpolated = df_amount_interpolated.where(df_amount_interpolated > 0, 0) # Replace the negative value to zero
    df_temp['amount'] = df_amount_interpolated
    df_temp = df_temp.dropna()
    df_temp['predict_amount'] = df_temp['amount'].iloc[predict_date:].reset_index()['amount']
    df_temp = df_temp.dropna()
    return df_temp
    
def data_seperation(data):
    stdsc = StandardScaler()
    X,y = data.iloc[:,1:-1].values,data.iloc[:,-1].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    #X_train_std = stdsc.fit_transform(X_train)
    #X_test_std = stdsc.fit_transform(X_test)
    return X_train, y_train, X_test, y_test


def Plot_feature_weight(Data, X_train, y_train):
    fig = plt.figure(1, figsize=(12, 10))
    ax = plt.subplot(111)
    colors = ['blue','green','red','cyan','yellow','black','pink','lightgreen','lightblue','orange']
    weights, params = [] ,[]
    for c in np.arange(-7,3,0.1):
        enet = ElasticNet(alpha=math.pow(10,c), l1_ratio=1)
        enet.fit(X_train,y_train)
        weights.append(enet.coef_[:])
        params.append(math.pow(10,c))
    weights = np.array(weights)
    
    for column,color in zip(range(weights.shape[1]), colors):
        plt.plot(params,weights[:,column],label=Data.columns[column + 1],color=color)
    plt.axhline(0,color='black',linestyle='--',linewidth=3)
    plt.xlim([math.pow(10,(3)),math.pow(10,(-7))])
    plt.ylabel('weight coefficient')
    plt.xlabel('Alpha')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center',bbox_to_anchor=(0.14,1),ncol=1,fancybox=True)
    plt.show()

def data_preprocessing(data, location):
    start_date = get_time_from_data(data.time)
    start_date = datetime.fromisoformat(start_date.iloc[0].replace('_', '-')[0:10])
    end_date = start_date + timedelta(1)
    climate_filename = 'climate_data_' + start_date.strftime("%Y_%m_%d") + '_' + end_date.strftime("%Y_%m_%d") + '.csv'
    if not os.path.exists(climate_filename):
        download_climate_data(station_name, start_date, end_date, climate_filename)
    climate_data = pd.read_csv(climate_filename) # 24 hours climate data
    climate_data_first_cleaned = first_stage_climate_data_clean(climate_data)
    detection_data_first_cleaned = first_stage_detection_data_clean(data)
    detection_data_second_cleaned = detection_data_first_cleaned.loc[:, ['time', 'amount']]
    merge_data = detection_data_second_cleaned.merge(climate_data_first_cleaned, on=['time'], how='left')
    cols = merge_data.columns.tolist()
    new_cols = cols[:1] + cols[2:6] + cols[1:2]
    merge_data = merge_data[new_cols].iloc[:, 1:].values
    return merge_data

                    
def main(station_name, detection_filename):    
    detection_data = pd.read_csv(detection_filename, encoding = 'ANSI') # the detection file is encoded in "ANSI"
    detection_data_first_cleaned = first_stage_detection_data_clean(detection_data)
    detection_data_second_cleaned = second_stage_detection_data_clean(detection_data_first_cleaned, station_name)
    
    start_date = datetime.fromisoformat(min(detection_data_second_cleaned.index)[0:10].replace('_', '-'))
    end_date = datetime.fromisoformat(max(detection_data_second_cleaned.index)[0:10].replace('_', '-'))
    climate_filename = 'climate_data_' + start_date.strftime("%Y_%m_%d") + '_' + end_date.strftime("%Y_%m_%d") + '.csv'
    if not os.path.exists(climate_filename):
        download_climate_data(station_name, start_date, end_date, climate_filename)
    climate_data = pd.read_csv(climate_filename)
    climate_data_first_cleaned = first_stage_climate_data_clean(climate_data)
    
    for i in range(1, 8):
        model_filename = 'ElasticNet_' + str(i) + '.pickle'
        detection_data_combined_climate = combine_data_climate_detection(detection_data_second_cleaned, climate_data_first_cleaned, i)    
        data = detection_data_combined_climate.iloc[:, [0,1,2,3,4,-2,-1]]
        
        X_train, y_train, X_test, y_test = data_seperation(data)
        
        r2_score_enet_list = []
        non_zero_weights_list = []
        net_list = []
        for a in np.arange(-7,3,1):
            enet = ElasticNet(alpha=math.pow(10,a), l1_ratio=0.5)
            y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
            r2_score_enet = r2_score(y_test, y_pred_enet)
            r2_score_enet_list.append(r2_score_enet)
            num_zero_weights = sum([abs(enet.coef_ - 0.0001) < 1e-3][0])
            non_zero_weights_list.append(num_zero_weights)
            net_list.append(enet)
            print('alpha=',a)
            print(enet.coef_) 
            print("r^2 on test data : %f" % r2_score_enet)
        performance_value = [x + 1e-2*y for x, y in zip(r2_score_enet_list, non_zero_weights_list)]
        index_selected_model = performance_value.index(max(performance_value))
        with open(model_filename, 'wb') as f:
            pickle.dump(net_list[index_selected_model], f)
    print('Training process finished')
    
def run(location, data_filename, model_filename):
    data =pd.read_csv(data_filename)
    x_test = data_preprocessing(data, location)
    with open(model_filename, 'rb') as f:
        enet = pickle.load(f)
    predicted_amount = enet.predict(x_test)[0]
    return predicted_amount
    
if __name__ == "__main__":
    
    # Python Parser
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-mode", help="training mode or inference mode", type=str)
    parser.add_argument("-station_name", help="name of station", type=str)
    parser.add_argument("-data_filename", help="file name of collected data from server", type=str)
    parser.add_argument("-model_filename", help="file name of the saved model", type=str)
    args = parser.parse_args()
    flag_mode = args.mode
    station_name = args.station_name
    data_filename = args.data_filename
    model_filename = args.model_filename
    
    if flag_mode == 'inference':
        # data_filename = 'test_samples.csv'
        # model_filename = 'ElasticNet_1.pickle'
        predicted_value = run(station_name, data_filename, model_filename)
        print('The predicted amount is %s' %(predicted_value))
    else:
        data_filename = 'database.csv'
        main(station_name, data_filename)