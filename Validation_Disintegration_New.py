import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st

# st.set_page_config(layout="wide")


st.markdown("""
    <style>
        p{
            font-size: 24px;
            font-weight: bold; 
        }
    </style>
""", unsafe_allow_html=True)
from sklearn.metrics import mean_absolute_error as mae
import re
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.express as px
import Disintegration_New_Model
import random
import pickle


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



def model_time_surface_volume_ratio(x,b,t_mid):
    time=x
    D=100*(1-1/(1+(time/t_mid)**b))
    return D
col1,col2=st.columns((2,2))
dataset_neats_df=pd.read_excel('./Disintegration.xlsx',sheet_name='Neat_Datasets')
dataset_neat=list(dataset_neats_df['Dataset'])
data_frame_containing_all=pd.read_excel('ComprehensiveModel.xlsx',sheet_name='Neat Commercialized Polymers')
dataset_formulation=pd.read_excel('Materials_Papers_Disintegration_Model_Original.xlsx')

def run_validation_neat(dataset_neat):
    datasets_neats=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Commercialized Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    datasets_neats_names=list(datasets_neats.loc[:,'Dataset'])
    datasets_neats_natural=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Natural Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    datasets_neats_names_natural=list(datasets_neats.loc[:,'Dataset'])


    datasets_blend_commercialized=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend of Commercialized')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    dataset_blends_names=list(datasets_blend_commercialized.loc[:,'Dataset'])
    datasets_blends_with_organics=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend_of_organic_additives')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    datasets_blends_organic_names=list(datasets_blends_with_organics.loc[:,'Dataset'])

    # dataset_neat=[]
    # while len(dataset_neat)<15:
    #     random_item = random.choice(datasets_neats_names)
    #     dataset_neat.append(random_item)
    # dataset_neat=["Data_set34","Data_set39","Data_set161","Data_set190","Data_set199"]
    # dataset_neat_natural=['Data_set260']
    # st.write(dataset_neat)

    dataset_blend_com=[]
    while len(dataset_blend_com)<3:
        random_item = random.choice(dataset_blends_names)
        dataset_blend_com.append(random_item)
    # st.write(dataset_blend_com)

    datasets_organic=[]
    while len(datasets_organic)<7:
        random_item = random.choice(datasets_blends_organic_names)
        # if random_item not in ['Data_set122','Data_set126','Data_set127','Data_set128','Data_set274','Data_set68','Data_set273','Data_set208','Data_set298','Data_set40','Data_set43','Data_set111','Data_set112','Data_set113','Data_set114']:
        datasets_organic.append(random_item)
    datasets_organic=['Data_set40','Data_set87','Data_set112','Data_set171','Data_set173','Data_set189']
    # st.write(datasets_organic)
    # st.write(datasets_organic)




    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    materials_functional_groups_natural=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    # model_=disintegration_prediction_neat_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)



    with open('./dataframe_disintegration.pickle','rb') as file: 
        data_disintegration_time_dis=pickle.load(file)


    negative_indices = data_disintegration_time_dis['Time'] < 0

    # Convert negative values to zero
    data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

    negative_indices = data_disintegration_time_dis['Disintegration'] < 0

    # Convert negative values to zero
    data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0


    datasets_old=list(data_disintegration_time_dis['Dataset'].unique())

    mae_list_new_model=[]
    unique_dataframe=data_disintegration_time_dis.copy()
    unique_dataframe = unique_dataframe.drop_duplicates(subset=['Time', 'Disintegration'], keep='first')
    mae_list_testing_old=[]
    # col1.write('New Model')
    # col2.write('Old Model')
    # try:

    with st.container():
        df = pd.DataFrame(columns=['Dataset', 'Composite','Prediction of Final Disintegration Value','Experimental Value for the Final Disintegration Data', 'Average Error'])
        mae_list=[]
        for data in dataset_neat:
            formulation=dataset_formulation[dataset_formulation['Dataset']==data]['Composition']

            neat_polymer=data_frame_containing_all[data_frame_containing_all['Dataset']==data]['Materials']

            col1.write(data)            
            # st.write(data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data])
            S_V=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Surface-to-volume (1/mm)'].iloc[0]
            # st.write(S_V)
            data_disintegration=pd.DataFrame(data_disintegration_time_dis)
            data_disintegration=data_disintegration[data_disintegration['Dataset'].isin([data])]
            data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]
            # for key,value in material_selected_and_wt.items():
            try:
                data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==datasets_neats[datasets_neats['Dataset']==data]['Materials'].iloc[0])]
                # st.write(data_row)
                data_row_=data_row.iloc[:,1:]
                data_row_['Time']=0    
                data_row_['Surface-to-volume (1/mm)']=S_V
                # st.write(data_row_.loc[0,'Materials'])
                st.write(data_row)
                material_selected_and_wt={data_row.iloc[0,0]:100}
                fig,parameters_surface_volume=Disintegration_New_Model.neat_commercialized_Validation([],S_V,material_selected_and_wt,data)
                Time=np.linspace(0,100,1000)
                # st.write(X_test_old)
                time_ii=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Time'].values
                Disintegration_ii=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Disintegration'].values

                y_predict=[]
                for t in time_ii:
                    data_row_['Time']=t

                model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
                
                y_i_smoothed_list=list(map(model_with_embeded_parameters_better,Time))
                data_set_predicted = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

                fig = px.line(data_set_predicted, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time')

                scatter_fig = px.scatter(data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data], x='Time', y='Disintegration',color_discrete_sequence=['red'])
                y_data=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Disintegration']



                y_predict=list(map(model_with_embeded_parameters_better,data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Time'].values))

                if len(neat_polymer)==0:
                    name_row=data+': '+formulation
                else:
                    name_row=data+': '+neat_polymer
                # st.write(Data_validation)

                mae_=mae(y_predict,y_data)
                mae_list.append(mae_)
                col1.write(Disintegration_ii)
                new_row = pd.DataFrame([[data, name_row,round(y_i_smoothed_list[-1],2),round(Disintegration_ii[-1],2), round(mae_,2)]],columns=['Dataset', 'Composite','Prediction of Final Disintegration Value','Experimental Value for the Final Disintegration Data', 'Average Error'])
                df = pd.concat([df, new_row], ignore_index=True)




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
                col1.write(fig)
                mae_list_new_model.append(mae(y_predict,y_data))
            except:
                pass
    html_table=generate_html_table(df, table_width='90%', font_size='25px')
    table=st.markdown(html_table, unsafe_allow_html=True)
    df2 = pd.DataFrame({'Metric':'Mean Absolute Error', 'Value':np.mean(mae_list)},index=[0])
    html_table2=generate_html_table(df2, table_width='90%', font_size='25px')

    table2=st.markdown(html_table2, unsafe_allow_html=True)

    return mae_list_new_model
def run_validation_organic():
    datasets_neats=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Commercialized Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    datasets_neats_names=list(datasets_neats.loc[:,'Dataset'])
    datasets_neats_natural=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Natural Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    datasets_neats_names_natural=list(datasets_neats.loc[:,'Dataset'])


    datasets_blend_commercialized=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend of Commercialized')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    dataset_blends_names=list(datasets_blend_commercialized.loc[:,'Dataset'])
    datasets_blends_with_organics=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend_of_organic_additives')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
    datasets_blends_organic_names=list(datasets_blends_with_organics.loc[:,'Dataset'])
    datasets_organic=['Data_set40','Data_set87','Data_set112','Data_set171','Data_set173','Data_set189']

    with open('./dataframe_disintegration.pickle','rb') as file: 
        data_disintegration_time_dis=pickle.load(file)


    negative_indices = data_disintegration_time_dis['Time'] < 0

    data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

    negative_indices = data_disintegration_time_dis['Disintegration'] < 0

    data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0
    datasets_old=list(data_disintegration_time_dis['Dataset'].unique())
    mae_list_new_model=[]
    unique_dataframe=data_disintegration_time_dis.copy()
    unique_dataframe = unique_dataframe.drop_duplicates(subset=['Time', 'Disintegration'], keep='first')
    for data in datasets_organic:
        

        test_old=data
        S_V=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Surface-to-volume (1/mm)'].iloc[0]
        data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        data_disintegration=data_disintegration[data_disintegration['Dataset'].isin([data])]
        data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]
        dataset__=datasets_blends_with_organics[datasets_blends_with_organics['Dataset']==data].copy()
        materials_wt_dictionary={}
        for index,row in dataset__.iterrows():
            materials_wt_dictionary[row['Material Name']]=row['Weight_percentage']
        materials=list(materials_wt_dictionary.keys())

        fig,parameters_surface_volume=Disintegration_New_Model.disintegration_prediction_blend_natural_commercialized_organic_validation(materials,S_V,materials_wt_dictionary,data)
        
        Time=np.linspace(0,100,1000)
        # st.write(X_test_old)
        time_ii=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Time'].values

        model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
        
        y_i_smoothed_list=list(map(model_with_embeded_parameters_better,Time))
        data_set_predicted = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

        fig = px.line(data_set_predicted, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time')

        scatter_fig = px.scatter(data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data], x='Time', y='Disintegration',color_discrete_sequence=['red'])
        y_data=data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Disintegration']
        y_predict=list(map(model_with_embeded_parameters_better,data_disintegration_time_dis[data_disintegration_time_dis['Dataset']==data]['Time'].values))
        for trace in scatter_fig.data:

            fig.add_trace(trace)
        col1.write(fig)
        mae_list_new_model.append(mae(y_predict,y_data))

    return mae_list_new_model









if col1.button('Randomly Pick 15 Datasets for Testing!',key='New Disintegration Model'):
    random_dataset_all_testing=[]
    for i in range(15):
        random_datasets_r=random.sample(dataset_neat,k=1)
        random_dataset_all_testing.append(random_datasets_r[0])

    mae_list_new_model=run_validation_neat(random_dataset_all_testing)

    # st.write('Average MAE is:')
    # st.write(np.mean(mae_list_new_model))

