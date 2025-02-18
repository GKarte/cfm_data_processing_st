# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:34 2025

@author: Gregor
"""



import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO



st.header("CFM data processing")


csv_file = st.file_uploader("Import IPSE data (.txt-export file)", key="upload_ipse")


# csv_file = "CFM_data_2025-02-17_17-20.csv"

if csv_file is not None:

    df_raw = pd.read_csv(csv_file, sep=",", header=0, index_col=0, engine='python') # read csv
    calib_corr_df = df_raw.loc["calibration correction mbar"]
    sensor_heights_df = df_raw.loc["sensor height"] # store height data
    data_df = df_raw.drop(labels=["sensor number","sensor range","sensor height","calibration correction mbar"], axis=0) # drop non-pressure rows
    
    # adjust timestamps (index of data_df) to start with 0
    time_array_str = list(data_df.index)
    time_array = np.zeros(len(time_array_str))
    for i,timestr in enumerate(time_array_str):
        time_array[i] = sum([a*b for a,b in zip([3600,60,1], map(float,timestr.split(" ")[1].split(':')))])
    time_array = time_array-time_array[0]
    data_df.index = time_array
    
    
    
    
    # t_p_meas = time_array[-1]-time_array[0]
    # # moving average of data
    # data_df_smooth = pd.DataFrame()
    # data_df_smooth.index = data_df.index
    # data_df_smooth = data_df.rolling(window=8, center=True).mean()
    
    # calculate mean values of pressure values and convert to df
    p_mean = data_df.mean().to_frame() # mean values of measured an corrected (auto calib) values
    p_std = data_df.std().to_frame() # mean values of measured an corrected (auto calib) values
    p_mean_not_corr = (data_df.mean()+calib_corr_df).to_frame() # mean values of measured values (not corrected)
    
    # merge dfs
    df_out = pd.merge(sensor_heights_df, p_mean, left_index=True, right_index=True) # add heights
    df_out = pd.merge(df_out, p_std, left_index=True, right_index=True) # add std
    df_out = pd.merge(df_out, p_mean_not_corr, left_index=True, right_index=True) # add calib corr
    df_out.columns = ["h/m", "p_mean/mbar", "p_std/mbar", "p_mean_not_corr/mbar"]
    
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    df_out.to_excel(writer, sheet_name="data", float_format="%.5f", startrow=0, index=True)
    writer.close()
    
    
    
    download = st.download_button(
        label="Export result",
        data=output.getvalue(),
        file_name= f'mean_{csv_file.name}.xlsx'
        )
    
    st.dataframe(df_out)