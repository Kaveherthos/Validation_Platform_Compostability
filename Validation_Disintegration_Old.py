from xgboost import XGBRegressor
import streamlit as st
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,median_absolute_error
from sklearn.metrics import mean_absolute_error as mae

import random

def generate_html_table(df, font_color='#373A36', table_width='70%', font_size='16px'):
    html_table = f'<table border="1" style="border-collapse: collapse; width: {table_width}; color: {font_color}; font-size: {font_size};">'

    # Add header row with background color
    html_table += '<tr style="background-color: #F4F0E8; font-weight: bold;">'
    for col in df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr>'
    # Add data rows with alternating background color
    for i, (_, row) in enumerate(df.iterrows()):
        background_color = '#F4F0E8' if i % 2 == 0 else '#F4F0E8'
        html_table += f'<tr style="background-color: {background_color};">'
        for value in row:
            html_table += f'<td>{value}</td>'
        html_table += '</tr>'

    html_table += '</table>'
    return html_table




with open('./dataframe_disintegration.pickle','rb') as file: 
    data_disintegration_time_dis=pickle.load(file)

file_path = 'species_dic_dis.pickle'

# Open the file in binary read mode
with open(file_path, 'rb') as f:
    # Load the list from the file
    species_dic_dis = pickle.load(f)


data_excel_=data_disintegration_time_dis.copy()



def model_time_surface_volume_ratio(x,b,t_mid):
    time=x
    D=100*(1-1/(1+(time/t_mid)**b))
    return D

Datasets_for_testing={'group1':list(range(2,7)),'group3':list(range(17,22)),'group6':list(range(35,39)),'group7':list(range(40,44)),'group8':list(range(44,49)),'group10':list(range(68,70)),'group11':list(range(83,90)),'group12':list(range(90,92)),'group13':list(range(92,96)),'group14':list(range(97,103)),'group16':list(range(110,115)),'group17':list(range(122,129)),'group18':list(range(141,150)),'group19':list(range(150,153)),'group21':list(range(171,176)),'group22':[187]+list(range(189,190)),'group23':list(range(200,205)),'group24':list(range(207,210))+list(range(211,213)),'group26':list(range(218,223)),'group27':list(range(223,228)),'group28':list(range(229,236)),'group31':list(range(251,255)),'group32':list(range(268,273)),'group33':list(range(274,277)),'group36':list(range(298,304))}
random_dataset_all_testing=[]
# for i_counter_ in range(1):
#     list_selection=[1,3,6,7,8,10,11,12,13,14,16,17,18,19,21,22,23,24,26,27,28,31,32,33,36]
#     random_datasets=[]
#     for r in list_selection:
#         random_datasets_r=random.sample(Datasets_for_testing['group'+str(r)],k=1)
#         random_datasets.append('Data_set'+str(random_datasets_r[0]))
#     random_dataset_all_testing.append(random_datasets)
    # print(random_dataset_all_testing)

dataset_neats_df=pd.read_excel('./Disintegration.xlsx',sheet_name='Neat_Datasets')
dataset_neat=list(dataset_neats_df['Dataset'])


# for i in range(15):
#     random_datasets_r=random.sample(dataset_neat,k=1)
#     # random_datasets.append('Data_set'+str(random_datasets_r[0]))
#     random_dataset_all_testing.append(random_datasets_r)


# random_dataset_all_testing=[["Data_set34","Data_set39","Data_set161","Data_set190","Data_set199","Data_set317"]]

# @use_named_args(space_xgboost)
data_frame_containing_all=pd.read_excel('ComprehensiveModel.xlsx',sheet_name='Neat Commercialized Polymers')
def plotting_validation_curves(random_dataset_all_testing):
    # random_dataset_all_testing
    # st.write(dataset_neat)
    Species_Dictionary_rev={}
    for i_c,name_dic in enumerate(species_dic_dis):
        Species_Dictionary_rev[i_c]=name_dic
    # data_excel_= Data_set_final_unique.copy()
        
    data_excel_sheet=data_excel_[data_excel_['Temperature']>30].iloc[:,:-1].copy()
    # for c_generated in range(5):
    #     random_set=random_dataset_all_testing[c_generated]
    mae_list=[]
    r2_list=[]
    rmse_list=[]
    accuracy_list=[]
    counters=1
    yy_prediction=[]
    true_possitive_rate_denominator=0
    true_possitive_rate_nomerator=0
    false_negative_rate_denominator=0
    false_negative_rate_nomerator=0
    dataset_formulation=pd.read_excel('Materials_Papers_Disintegration_Model_Original.xlsx')
    # len(random_dataset_all_testing)
    df = pd.DataFrame(columns=['Dataset', 'Composite','Prediction of Final Disintegration Value','Experimental Value for the Final Disintegration Data', 'Average Error'])

    for rand_dataset in random_dataset_all_testing:
        formulation=dataset_formulation[dataset_formulation['Dataset']==rand_dataset]['Composition']
        neat_polymer=data_frame_containing_all[data_frame_containing_all['Dataset']==rand_dataset]['Materials']
        # st.write(neat_polymer)
        if len(neat_polymer)==0:
            st.write(rand_dataset+': '+formulation)
        else:
            st.write(rand_dataset+': '+neat_polymer)

        counterw=counters+1
        y_predictions_perrun=[]
        filter_=~data_excel_sheet.isin({'Dataset': [rand_dataset]}).copy()
        Data_training=data_excel_sheet[filter_['Dataset']].copy()
        Data_validation=data_excel_sheet[data_excel_sheet['Dataset']==rand_dataset].copy()
        X_Regression=Data_training.iloc[:,[0]+list(range(3,3+len(species_dic_dis)+2))].copy()
        for column in X_Regression.columns:
            X_Regression[column] = pd.to_numeric(X_Regression[column], errors='coerce')
        y_Regression=Data_training.iloc[:,1].copy()
        X_Regression_test=Data_validation.iloc[:,[0]+list(range(3,3+len(species_dic_dis)+2))].copy()
        for column in X_Regression_test.columns:
            X_Regression_test[column] = pd.to_numeric(X_Regression_test[column], errors='coerce')
        y_Regression_test=Data_validation.iloc[:,1].copy().values
        model_ = XGBRegressor(n_estimators=100, max_depth=20, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)
        model_.fit(X_Regression,y_Regression)

        number_testing=X_Regression_test.shape[0]
        X_Regression_test=X_Regression_test.values
        for counter_regression in range(number_testing):
            X_in=X_Regression_test[counter_regression,:]
            D_1=model_.predict(X_in.reshape((1,-1)))
            if isinstance(D_1, (list, tuple, np.ndarray)) and len(D_1) == 1:
                D_1 = float(D_1[0])
            elif isinstance(D_1, (int, float)):
                D_1 = float(D_1)
            y_predictions_perrun.append(D_1)
            yy_prediction.append(D_1)
        Data_frame_containing_predictions=pd.DataFrame({'Time':Data_validation['Time'],'Disintegration':y_predictions_perrun,'Disintegration_data':Data_validation['Disintegration'],'Dataset':Data_validation['Dataset']})        

        time_i=np.array(Data_frame_containing_predictions.iloc[:,0]).reshape((-1,))
        time_ii=time_i.copy()
        D_sub=np.array(Data_frame_containing_predictions.iloc[:,1]).reshape((-1,))
        for i_t,t in enumerate(time_ii):
            if t<0:
                time_ii[i_t]=0
        time_ii_plotting=np.linspace(0,100,200)
        try:
            parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(time_ii).reshape((-1,)),np.array(D_sub).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
            model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
            y_i_smoothed=np.array(list(map(model_with_embeded_parameters_better,time_ii)))
            Data_frame_containing_predictions_smoothed=pd.DataFrame({'Time':Data_validation['Time'],'Disintegration':y_i_smoothed,'Disintegration_data':Data_validation['Disintegration'],'Dataset':Data_validation['Dataset']})        


            y_i_smoothed_plotting=np.array(list(map(model_with_embeded_parameters_better,time_ii_plotting)))

            Data_frame_containing_predictions_smoothed_more_predictions=pd.DataFrame({'Time':time_ii_plotting,'Disintegration':y_i_smoothed_plotting})        





            mae_=mae(np.array(Data_frame_containing_predictions.iloc[:,2]).reshape((-1,)),y_i_smoothed)
            # st.write(time_i,)
            if len(neat_polymer)==0:
                name_row=rand_dataset+': '+formulation
            else:
                name_row=rand_dataset+': '+neat_polymer
            # st.write(Data_validation)
            new_row = pd.DataFrame([[rand_dataset, name_row,round(y_i_smoothed[-1],2),round(Data_validation.iloc[-1,1],2), round(mae_,2)]],columns=['Dataset', 'Composite','Prediction of Final Disintegration Value','Experimental Value for the Final Disintegration Data', 'Average Error'])
            df = pd.concat([df, new_row], ignore_index=True)



            mae_list.append(mae_)



            fig=   px.line(Data_frame_containing_predictions_smoothed_more_predictions, x='Time', y='Disintegration', title='Disintegration Plot (%) vs Time')
            scatter_fig = px.scatter(Data_frame_containing_predictions, x='Time', y='Disintegration_data',color_discrete_sequence=['red'])

            fig.update_layout(width=1000, height=600,    font=dict(
                family="Arial",
                size=40,  # Adjust the font size as needed
                color="black"  # You can also set the font color if needed
            ))
            for trace in scatter_fig.data:
                # pass
                fig.add_trace(trace)

            fig.update_layout(width=1000,height=600,    font=dict(
                family="Arial",
                size=40,  # Adjust the font size as needed
                color="black"  # You can also set the font color if needed
            ))
            fig.update_xaxes(title_text='<b>Time (day)</b>',tickvals=[0,15,30,45,60,75,90,100],
                                                    ticktext=['<b>0</b>', '<b>15</b>', '<b>30</b>', '<b>45</b>',
                                                        '<b>60</b>', '<b>75</b>', '<b>90</b>', '<b>100</b>'],
                                                        title_font=dict(size=22, family="Arial", color="#373A36"),
                                                        tickfont=dict(size=22, family="Arial", color="#373A36"))
            
            fig.update_yaxes(title_text='<b>Disintegration (%)</b>',tickvals=[0,15,30,45,60,75,90,100],
                                                ticktext=['<b>0</b>', '<b>15</b>', '<b>30</b>', '<b>45</b>', '<b>60</b>',
                                                '<b>75</b>', '<b>90</b>', '<b>100</b>'],range=[0, 100],
                                                title_font=dict(size=22, family="Arial", color="#373A36"),
                                                tickfont=dict(size=22, family="Arial", color="#373A36"))
            
            for trace in scatter_fig.data:
                # pass
                fig.add_trace(trace)



            st.write(fig)

                    
                
                
                
        except RuntimeError as e:
            print('error occured')
    
    html_table=generate_html_table(df, table_width='90%', font_size='25px')
    table=st.markdown(html_table, unsafe_allow_html=True)
    df2 = pd.DataFrame({'Metric':'Mean Absolute Error', 'Value':np.mean(mae_list)},index=[0])
    html_table2=generate_html_table(df2, table_width='90%', font_size='25px')

    table2=st.markdown(html_table2, unsafe_allow_html=True)

    # st.write(np.mean(mae_list))
    # 
    # if counter_composites==1:
    #     table=st.markdown(html_table, unsafe_allow_html=True)
    # else:
    #     table.empty()
    #     table=st.markdown(html_table, unsafe_allow_html=True)



    # st.table(df)
if st.button('Randomly Pick 15 Datasets for Testing!',key='Old Biodegradation Model'):
    random_dataset_all_testing=[]
    for i in range(15):
        random_datasets_r=random.sample(dataset_neat,k=1)
        random_dataset_all_testing.append(random_datasets_r[0])
    plotting_validation_curves(random_dataset_all_testing)