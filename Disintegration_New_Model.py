import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st
import re
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.express as px
import pickle
def model_time_surface_volume_ratio(x,b,t_mid):
    time=x
    D=100*(1-1/(1+(time/t_mid)**b))
    return D



class disintegration_prediction:
    def __init__(self):
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D
        data_chemical_structure_excel=pd.read_excel('./ChemicalStructureModeling.xlsx')
        data_chemical_grades_excel=pd.read_excel('./Polymer_Grades.xlsx')
        Data_blends_modeling_weight_percentages=pd.read_excel('./Dataset_with_single_polymers.xlsx',sheet_name='Disintegration_blends')
        Data_blends_modeling_weight_percentages=Data_blends_modeling_weight_percentages.iloc[:,0:4]
        datasets=Data_blends_modeling_weight_percentages.iloc[:,0].unique()
        merged_df = pd.merge(data_chemical_grades_excel, data_chemical_structure_excel, on='Materials', how='inner')
        list_columns=list(merged_df.columns)

        # Data_frame_combined=pd.DataFrame(columns=['Dataset']+list_columns[2:-1])
        new_Data_frame_combined=pd.DataFrame(columns=['Dataset']+list_columns[2:-1]) #Final dataframe including all information
        i=0
        data_sets=[data for data in datasets if data != 'Data_set58']
        # data_sets
        for dataset in data_sets:
            data_selected_data_i=Data_blends_modeling_weight_percentages[Data_blends_modeling_weight_percentages['Datasets']==dataset]
            # st.write(data_selected_data_i)
            new_Data_frame_combined.loc[i,'Dataset']=dataset
            # st.write(new_Data_frame_combined)
            # new_row=[0]*(len(list_columns)-3)
            counter=0
            for index,row in data_selected_data_i.iterrows():
                grade=row[-1]
                new_data_frame=merged_df[merged_df['Polymer Grade']==grade].iloc[:,2:-1].copy()
                new_data_frame['Dataset']=dataset
                new_data_frame=new_data_frame[['Dataset']+list_columns[2:-1]]
                if counter==0:
                    new_Data_frame_combined.iloc[i,1:]=(row['Weight_percentage']*new_data_frame.iloc[0,1:].values)/100
                else:
                    new_Data_frame_combined.iloc[i,1:]=new_Data_frame_combined.iloc[i,1:]+(row['Weight_percentage']*new_data_frame.iloc[0,1:].values)/100

                counter+=1
            i+=1
        import pickle

        if 'selected_disintegration_data' not in st.session_state.keys():
            with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
                data = pickle.load(pickle_file)
        else:
            with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
                data = pickle.load(pickle_file)



        # with open('./dataframe_disintegration_old.pkl','rb') as file: 
        #     data=pickle.load(file)


        negative_indices = data['Time'] < 0

        # Convert negative values to zero
        data.loc[negative_indices, 'Time'] = 0

        negative_indices = data['Disintegration'] < 0

        # Convert negative values to zero
        data.loc[negative_indices, 'Disintegration'] = 0



        data_disintegration=pd.DataFrame(data)


        data_selected_part_single=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]

        data_selected_part_single


        data_disintegration_chemical_structure = pd.merge(new_Data_frame_combined, data_selected_part_single, on='Dataset', how='inner')
        # st.write(data_disintegration_chemical_structure)
        data_disintegration_chemical_structure_filtered = data_disintegration_chemical_structure.loc[:, (data_disintegration_chemical_structure != 0).any(axis=0)]
        # st.write(data_disintegration_chemical_structure_filtered)
        X_train=data_disintegration_chemical_structure_filtered.iloc[:,1:-1].values
        Y_train=data_disintegration_chemical_structure_filtered.iloc[:,-1].values
        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)
        
        self.model.fit(X_train,Y_train)
    def predict(self,X_in):
        y=self.model.predict(X_in)
        return y


class disintegration_prediction_neat_commercialized:
    def __init__(self,datasets_excluded):
        self.datasets_excluded=datasets_excluded
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D
        # st.write(datasets_excluded)
        if datasets_excluded==None:        
            materials_data_sets=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Commercialized Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
            materials_functional_groups_neat=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials, Mw,PDI, ester_wt_percentage and other wt_percentage values
            datasets=materials_data_sets.iloc[:,0].unique()
        else:
            materials_data_sets = pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Commercialized Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
            materials_data_sets = materials_data_sets[~materials_data_sets['Dataset'].isin([datasets_excluded])]
            materials_functional_groups_neat = pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials, Mw,PDI, ester_wt_percentage and other wt_percentage values
            datasets=materials_data_sets.iloc[:,0].unique()
            datasets=[data for data in datasets if data not in  datasets_excluded]


        merged_df = pd.merge(materials_data_sets.iloc[:,[0,1]], materials_functional_groups_neat, on='Materials', how='inner')
        list_columns=list(merged_df.columns)


        # st.write(new_Data_frame_combined)
        with open('./dataframe_disintegration.pickle','rb') as file: 
            data_disintegration_time_dis=pickle.load(file)


        # if 'selected_disintegration_data' not in st.session_state.keys():
        #     with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)
        # else:
        #     with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)


        negative_indices = data_disintegration_time_dis['Time'] < 0

        # Convert negative values to zero
        data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

        negative_indices = data_disintegration_time_dis['Disintegration'] < 0

        # Convert negative values to zero
        data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0
        data_disintegration_time_dis=data_disintegration_time_dis[~data_disintegration_time_dis['Dataset'].isin([datasets_excluded])]

        data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets)]
        data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]
        Data_set_training=pd.merge(merged_df,data_time_dis_training,on='Dataset',how='inner')
        # st.write(Data_set_training.iloc[:,2:-1])
        X_train=Data_set_training.iloc[:,2:-1].values
        # st.write(X_train)
        # st.write(Data_set_training)
        Y_train=Data_set_training.iloc[:,-1].values

        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)
        
        self.model.fit(X_train,Y_train)
        # with open('disintegration_prediction_neat_commercialized.pkl', 'wb') as f:
        #     pickle.dump(self.model, f)


    def predict(self,X_in):
        y=self.model.predict(X_in)
        return y


class disintegration_prediction_neat_natural:
    def __init__(self,datasets_excluded):
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D


        # if datasets_excluded==None:        
        #     materials_data_sets=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Natural Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
        #     materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
        # else:
        #     materials_data_sets=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Neat Natural Polymers')# the columns of this dataset are: Datasets, Materials, wt_percentage and Type (commercialized, natural,etc)
        #     materials_data_sets = materials_data_sets[~materials_data_sets['Dataset'].isin([datasets_excluded])]
        #     materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
        # datasets=materials_data_sets.iloc[:,0].unique()
        # merged_df = pd.merge(materials_data_sets.iloc[:,[0,1]], materials_functional_groups_neat.iloc[:,:-2], on='Materials', how='inner')
        # list_columns=list(merged_df.columns)


        # import pickle
        # # st.write(new_Data_frame_combined)
        # if 'selected_disintegration_data' not in st.session_state.keys():
        #     with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)
        # else:
        #     with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)



        # negative_indices = data_disintegration_time_dis['Time'] < 0

        # # Convert negative values to zero
        # data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

        # negative_indices = data_disintegration_time_dis['Disintegration'] < 0

        # # Convert negative values to zero
        # data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0

        # data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        # if datasets_excluded is None:
        #     data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets)]
    
        # else:
        #     datasets_included=[data for data in datasets if data not in  datasets_excluded]

        #     data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets_included)]
    
        # data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]
        # Data_set_training=pd.merge(merged_df,data_time_dis_training,on='Dataset',how='inner')
        # # st.write('Training dataset is:')
        # # st.write(Data_set_training.iloc[:,2:-1])
        # X_train=Data_set_training.iloc[:,2:-1].values
        # Y_train=Data_set_training.iloc[:,-1].values
        with open('disintegration_prediction_neat_natural.pkl', 'rb') as f:
            self.model = pickle.load(f)
# disintegration_prediction_neat_natural
#         self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)
        
#         self.model.fit(X_train,Y_train)
#         with open('disintegration_prediction_neat_natural.pkl', 'wb') as f:
#             pickle.dump(self.model, f)

    def predict(self,X_in):
        y=self.model.predict(X_in)
        return y


class disintegration_prediction_blend_commercialized:
    def __init__(self):
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D
        data=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend of Commercialized')
        data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
        # data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
        datasets=data.Dataset.unique()
        data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
        # data_natural_polymer=pd.DataFrame({'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
        data_set_features=pd.DataFrame({'Dataset':'','Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0},index=[0])
        data_set_features_all=pd.DataFrame({'Dataset':'','Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0},index=[0])
        list_all=data_set_features_all.keys()
        for data_i in datasets:
            data_=data[data['Dataset']==data_i].copy()
            data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0},index=[0])
            list_columns_commercialized_=data_commercialized_polymer.columns
            for data_index,data_row in data_.iterrows():
                if data_row['Type']=='Commercialized':
                    for col_names in list_columns_commercialized:
                        data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_commercialized_pool[data_commercialized_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                
                
            for col_ in list_all:
                if col_=='Dataset':
                    data_set_features.loc[0,col_]=data_i
                elif col_ in list_columns_commercialized_:
                    data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
                
            if data_i==datasets[0]:
                data_set_all_features=data_set_features.copy()
            else:
                data_set_all_features=pd.concat([data_set_all_features,data_set_features],ignore_index=True)
            # st.write('Disintegration data of features for blend of commercialized is:')  
            # st.write(data_set_all_features)
        if 'selected_disintegration_data' not in st.session_state.keys():
            with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
                data_disintegration_time_dis = pickle.load(pickle_file)
        else:
            with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
                data_disintegration_time_dis = pickle.load(pickle_file)



        negative_indices = data_disintegration_time_dis['Time'] < 0

        # Convert negative values to zero
        data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

        negative_indices = data_disintegration_time_dis['Disintegration'] < 0

        # Convert negative values to zero
        data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0


        data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets)]
        data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]

        Data_set_training=pd.merge(data_set_all_features,data_time_dis_training,on='Dataset',how='inner')
        # st.write('Training dataset is:')
        # st.write(Data_set_training)
        X_train=Data_set_training.iloc[:,1:-1].values
        Y_train=Data_set_training.iloc[:,-1].values

        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)

        self.model.fit(X_train,Y_train)
        with open('disintegration_prediction_blend_commercialized.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self,X_in):
        y=self.model.predict(X_in)
        return y













class disintegration_prediction_blend_nat_com:
    def __init__(self):
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D
        
        # data=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend_Commercialized__Natural')


        # data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
        # data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')



        # datasets=data.Dataset.unique()
        # data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
        # data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

        # list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
        # list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
        # list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
        # 'Weight_fraction_ether_Natural',
        # 'Weight_fraction_carbonyl_Natural',
        # 'Weight_fraction_hydroxyl_Natural',
        # 'Weight_fraction_ketal_Natural',
        # 'Weight_fraction_BenzeneCycle_Natural',
        # 'Weight_fraction_carboxylic_acid_Natural']
        # data_set_features=pd.DataFrame({'Dataset':'','Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        # data_set_features_all=pd.DataFrame({'Dataset':'','Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        # list_all=data_set_features_all.keys()
        # for data_i in datasets:

                
        #     data_=data[data['Dataset']==data_i].copy()
        #     data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
        #     data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        #     wt_commercialized=0
        #     for data_index,data_row in data_.iterrows():
        #         # print(data_row)
        #         if data_row['Type']=='Commercialized':
        #             # st.write(data_row)
        #             # st.write(list_columns_commercialized)
        #             for col_names in list_columns_commercialized:
        #                 # st.write(col_names)
        #                 if col_names in ['Mw','PDI']:
        #                     data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(data_row['Wt%'])*data_commercialized_pool[data_commercialized_pool['Materials']==data_row['Materials']].loc[:,col_names].values
        #                 else:
        #                     data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(data_row['Wt%']/100)*data_commercialized_pool[data_commercialized_pool['Materials']==data_row['Materials']].loc[:,col_names].values
        #             wt_commercialized+=data_row['Wt%']
        #         elif data_row['Type']=='Natural':
        #             for col_names in list_columns_natural:
        #                 data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(data_row['Wt%']/100)*data_natural_pool[data_natural_pool['Materials']==data_row['Materials']].loc[:,col_names].values
            
                    
        #         data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=data_[data_['Type']=='Commercialized']['Wt%'].sum()
        #         # st.write(data_commercialized_polymer)
        #         if wt_commercialized>0:
        #             data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/wt_commercialized
        #             data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/wt_commercialized

                
        #     for col_ in list_all:
        #         if col_=='Dataset':
        #             data_set_features.loc[0,col_]=data_i
        #         elif col_ in list_columns_commercialized_:
        #             data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        #         elif col_ in list_columns_natural:
        #             data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
                
        #     if data_i==datasets[0]:
        #         data_set_all_features=data_set_features.copy()
        #     else:
        #         data_set_all_features=pd.concat([data_set_all_features,data_set_features],ignore_index=True)
                

        # if 'selected_disintegration_data' not in st.session_state.keys():
        #     with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)
        # else:
        #     with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)



        # negative_indices = data_disintegration_time_dis['Time'] < 0

        # # Convert negative values to zero
        # data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

        # negative_indices = data_disintegration_time_dis['Disintegration'] < 0

        # # Convert negative values to zero
        # data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0


        # data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        # data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets)]
        # data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]

        # Data_set_training=pd.merge(data_set_all_features,data_time_dis_training,on='Dataset',how='inner')
        # # st.write('Training dataset is:')
        # # st.write(Data_set_training)
        # X_train=Data_set_training.iloc[:,1:-1].values
        # Y_train=Data_set_training.iloc[:,-1].values

        # self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)

        # self.model.fit(X_train,Y_train)
        with open('disintegration_prediction_blend_nat_com.pkl', 'rb') as f:
            self.model=pickle.load( f)

    def predict(self,X_in):
        y=self.model.predict(X_in)
        return y



class disintegration_prediction_blend_nat_com_org:
    def __init__(self,datasets_excluded):
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D
        
        data=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend_of_organic_additives')


        data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
        data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
        data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')






        if datasets_excluded==None:        
            datasets=data.Dataset.unique()
            # datasets=[data for data in datasets if data not in datasets_excluded]# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
        else:
            datasets=data.Dataset.unique()
            datasets=[data for data in datasets if data not in datasets_excluded]


        # st.write('datasets are')
        # st.write(datasets)












        data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
        data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
        list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']

        list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
        list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
        list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                    'Weight_fraction_of_maleinized_group_organic',
                    'Weight_fraction_of_carboxilic_acid_groups_organic',
                    'Triazinane_organic',
                    'Isocyanate_organic',
                    'Weight_fraction_aldehide_organic']


        data_set_features=pd.DataFrame({
            'Dataset': '',
            'Mw': 0,
            'PDI': 0,
            'Weight_fraction_ester': 0,
            'Weight_fraction_ether': 0,
            'Weight_fraction_BenzeneCycle': 0,
            'Weight_fraction_urethane': 0,
            'Weight_fraction_hydroxyl':0,
            'Weight_fraction_phthalate':0,
            'Weight_percentage_Commercialized Polymers': 0,
            'Weight_fraction_ester_Natural':0,
            'Weight_fraction_amine_Natural': 0,
            'Weight_fraction_ether_Natural': 0,
            'Weight_fraction_carbonyl_Natural': 0,
            'Weight_fraction_hydroxyl_Natural': 0,
            'Weight_fraction_ketal_Natural': 0,
            'Weight_fraction_BenzeneCycle_Natural': 0,
            'Weight_fraction_carboxylic_acid_Natural': 0,
            'MW_organic': 0,
            'Weight_fraction_esters_organic': 0,
            'Weight_fraction_hydroxyl_organic': 0,
            'Weight_fraction_phenyl_organic': 0,
            'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
            'Weight_fraction_of_carbons_in_double_bond_organic': 0,
            'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
            'Weight_fraction_of_maleinized_group_organic': 0,
            'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
            'Triazinane_organic': 0,
            'Isocyanate_organic': 0,
            'Weight_fraction_aldehide_organic': 0},index=[0])
        data_set_features_all=pd.DataFrame({
            'Dataset': '',
            'Mw': 0,
            'PDI': 0,
            'Weight_fraction_ester': 0,
            'Weight_fraction_ether': 0,
            'Weight_fraction_BenzeneCycle': 0,
            'Weight_fraction_urethane': 0,
            'Weight_fraction_hydroxyl':0,
            'Weight_fraction_phthalate':0,
            'Weight_percentage_Commercialized Polymers': 0,
            'Weight_fraction_ester_Natural':0,
            'Weight_fraction_amine_Natural': 0,
            'Weight_fraction_ether_Natural': 0,
            'Weight_fraction_carbonyl_Natural': 0,
            'Weight_fraction_hydroxyl_Natural': 0,
            'Weight_fraction_ketal_Natural': 0,
            'Weight_fraction_BenzeneCycle_Natural': 0,
            'Weight_fraction_carboxylic_acid_Natural': 0,
            'MW_organic': 0,
            'Weight_fraction_esters_organic': 0,
            'Weight_fraction_hydroxyl_organic': 0,
            'Weight_fraction_phenyl_organic': 0,
            'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
            'Weight_fraction_of_carbons_in_double_bond_organic': 0,
            'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
            'Weight_fraction_of_maleinized_group_organic': 0,
            'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
            'Triazinane_organic': 0,
            'Isocyanate_organic': 0,
            'Weight_fraction_aldehide_organic': 0},index=[0])
        list_all=data_set_features_all.keys()
        for data_i in datasets:   
            # st.write(data_i)
            # st.write(data_i)             
            data_=data[data['Dataset']==data_i].copy()
            data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
            data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
            data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
            wt_percentage_commercialized=0
            wt_percentage_organic=0
            # st.write(data_)
            for data_index,data_row in data_.iterrows():
                # st.write(data_row)
                # st.write(data_row)
                # st.write('data row is')
                # st.write(data_row)
                # st.write('data row is')
                # st.write(data_row)
                if data_row['Type']=='Commercialized':
                    # st.write(data_row)
                    # st.write(list_columns_commercialized)
                    # st.write(list_columns_commercialized)
                    for col_names in list_columns_commercialized:
                        # st.write({'col name is':col_names})
                        
                        # st.write(col_names)
                        # st.write(col_names)
                        # st.write('error is')
                        # st.write(len(col_names))
                        # st.write(data_row)
                        # st.write(col_names)
                        # st.write(data_row)
                        if col_names in ['Mw','PDI']:

                            data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(data_row['Weight_percentage'])*data_commercialized_pool[data_commercialized_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                        else:    
                            data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_commercialized_pool[data_commercialized_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                    wt_percentage_commercialized+=data_row['Weight_percentage']

                elif data_row['Type']=='Natural':
                    for col_names in list_columns_natural:
                        # st.write('here is the error')
                        # st.write(col_names)
                        # st.write(data_natural_polymer.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']==data_row['Material Name']].loc[:,col_names].values)
                        # st.write(data_natural_pool[data_natural_pool['Materials']==data_row['Material Name']])
                        data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                elif data_row['Type']=='Organic Additive':
                    for col_names in list_columns_organic_additive:
                        if col_names=='MW_organic':
                            data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(data_row['Weight_percentage'])*data_organic_pool[data_organic_pool['Materials']==data_row['Material Name']].loc[:,col_names].values

                        else:
                            data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                        wt_percentage_organic+=data_row['Weight_percentage']
                elif data_row['Material Name']=='Thermoplastic Starch (TPS)':
                    for col_names in list_columns_organic_additive:
                        data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+0.2*(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']=='Glycerol'].loc[:,col_names].values
                    
                    for col_names in list_columns_natural:
                        data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+0.8*(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']=='Starch'].loc[:,col_names].values

                elif data_row['Material Name']=='Renol':
                    for col_names in list_columns_organic_additive:
                        data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(1/8)*(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']=='Methyl 9,10-epoxysterate'].loc[:,col_names].values
                    
                    for col_names in list_columns_natural:
                        data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(7/8)*(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']=='Lignin'].loc[:,col_names].values

                elif data_row['Material Name']=='Thermoplastic Wheat Flour':
                    for col_names in list_columns_organic_additive:
                        data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(0.23)*(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']=='Glycerol'].loc[:,col_names].values
                    
                    for col_names in list_columns_natural:
                        data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+0.77*(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']=='Mantegna Wheat Flour'].loc[:,col_names].values


                    
                data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=data_[data_['Type']=='Commercialized']['Weight_percentage'].sum()
                if wt_percentage_commercialized>0:
                    data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
                    data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/wt_percentage_commercialized
                if wt_percentage_organic>0:
                    data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic
            for col_ in list_all:
                if col_=='Dataset':
                    data_set_features.loc[0,col_]=data_i
                elif col_ in list_columns_commercialized_:
                    data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
                elif col_ in list_columns_natural:
                    data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
                elif col_ in list_columns_organic_additive:
                    data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]
                
            if data_i==datasets[0]:
                data_set_all_features=data_set_features.copy()
            else:
                data_set_all_features=pd.concat([data_set_all_features,data_set_features],ignore_index=True)
                
        with open('./dataframe_disintegration.pickle','rb') as file: 
            data_disintegration_time_dis=pickle.load(file)

        # if 'selected_disintegration_data' not in st.session_state.keys():
        #     with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)
        # else:
        #     with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)



        negative_indices = data_disintegration_time_dis['Time'] < 0

        # Convert negative values to zero
        data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

        negative_indices = data_disintegration_time_dis['Disintegration'] < 0

        # Convert negative values to zero
        data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0


        data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets)]
        data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]
        Data_set_training=pd.merge(data_set_all_features,data_time_dis_training,on='Dataset',how='inner')
        # st.write(Data_set_training)
        X_train=Data_set_training.iloc[:,1:-1].values
        Y_train=Data_set_training.iloc[:,-1].values

        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)

        self.model.fit(X_train,Y_train)
        # with open('disintegration_prediction_blend_nat_com_org.pkl', 'rb') as f:
        #     self.model=pickle.load(f)

    def predict(self,X_in):
        # st.write('X_in is')
        # st.write(X_in)
        y=self.model.predict(X_in)
        return y





class disintegration_prediction_blend_nat_com_org_inorg:
    def __init__(self,datasets_excluded):
        def model_time_surface_volume_ratio(x,b,t_mid):
            time=x
            D=100*(1-1/(1+(time/t_mid)**b))
            return D
        
        # data=pd.read_excel('./Materials_Information/ComprehensiveModel.xlsx',sheet_name='Blend_of_inorganic_Additives')


        # data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
        # data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
        # data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')
        # data_inorganic_pool=pd.read_excel('./Materials_Information/Inorganic_Materials_not_clay_additive.xlsx')
        # data_clay_pool=pd.read_excel('./Materials_Information/Clay_Additive.xlsx')




        # if datasets_excluded==None:        
        #     datasets=data.Dataset.unique()
        #     # datasets=[data for data in datasets if data not in datasets_excluded]# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
        # else:
        #     datasets=data.Dataset.unique()
        #     datasets=[data for data in datasets if data not in datasets_excluded]

        # data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
        # data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        # data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
        # data_inorganic_additive=pd.DataFrame({'Silver': 0,'CalciumPhosphate': 0,'HydroxyApatite': 0,'CalciumCarbonate': 0,'SiO2': 0,'Titanium Oxide':0,'Graphene':0,'Volcanic Ash': 0,'d50 (mm)': 0},index=[0])
        # data_clay_additive=pd.DataFrame({'Cation_Exchange_Capacity':0,'Water Contact Angle':0,'d50 (micron)':0},index=[0])
        
        # list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
        # list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
        # list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
        # list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
        #             'Weight_fraction_of_maleinized_group_organic',
        #             'Weight_fraction_of_carboxilic_acid_groups_organic',
        #             'Triazinane_organic',
        #             'Isocyanate_organic',
        #             'Weight_fraction_aldehide_organic']
        # list_columns_inorganic=['Silver','CalciumPhosphate','HydroxyApatite','CalciumCarbonate','SiO2','Titanium Oxide','Graphene','Volcanic Ash','d50 (mm)']
        # list_columns_clay=['Cation_Exchange_Capacity','Water Contact Angle',	'd50 (micron)']

        # data_set_features=pd.DataFrame({
        #     'Dataset': '',
        #     'Mw': 0,
        #     'PDI': 0,
        #     'Weight_fraction_ester': 0,
        #     'Weight_fraction_ether': 0,
        #     'Weight_fraction_BenzeneCycle': 0,
        #     'Weight_fraction_urethane': 0,
        #     'Weight_fraction_hydroxyl':0,
        #     'Weight_fraction_phthalate':0,
        #     'Weight_percentage_Commercialized Polymers': 0,
        #     'Weight_fraction_ester_Natural':0,
        #     'Weight_fraction_amine_Natural': 0,
        #     'Weight_fraction_ether_Natural': 0,
        #     'Weight_fraction_carbonyl_Natural': 0,
        #     'Weight_fraction_hydroxyl_Natural': 0,
        #     'Weight_fraction_ketal_Natural': 0,
        #     'Weight_fraction_BenzeneCycle_Natural': 0,
        #     'Weight_fraction_carboxylic_acid_Natural': 0,
        #     'MW_organic': 0,
        #     'Weight_fraction_esters_organic': 0,
        #     'Weight_fraction_hydroxyl_organic': 0,
        #     'Weight_fraction_phenyl_organic': 0,
        #     'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        #     'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        #     'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        #     'Weight_fraction_of_maleinized_group_organic': 0,
        #     'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        #     'Triazinane_organic': 0,
        #     'Isocyanate_organic': 0,
        #     'Weight_fraction_aldehide_organic': 0,
        #     'Silver': 0,
        #     'CalciumPhosphate': 0,
        #     'HydroxyApatite': 0,
        #     'CalciumCarbonate': 0,
        #     'SiO2': 0,
        #     'Titanium Oxide':0,
        #     'Graphene':0,
        #     'Volcanic Ash': 0,
        #     'd50 (mm)': 0,
        #     'Cation_Exchange_Capacity':0,
        #     'Water Contact Angle':0,	'd50 (micron)':0},index=[0])
        # data_set_features_all=pd.DataFrame({
        #     'Dataset': '',
        #     'Mw': 0,
        #     'PDI': 0,
        #     'Weight_fraction_ester': 0,
        #     'Weight_fraction_ether': 0,
        #     'Weight_fraction_BenzeneCycle': 0,
        #     'Weight_fraction_urethane': 0,
        #     'Weight_fraction_hydroxyl':0,
        #     'Weight_fraction_phthalate':0,
        #     'Weight_percentage_Commercialized Polymers': 0,
        #     'Weight_fraction_ester_Natural':0,
        #     'Weight_fraction_amine_Natural': 0,
        #     'Weight_fraction_ether_Natural': 0,
        #     'Weight_fraction_carbonyl_Natural': 0,
        #     'Weight_fraction_hydroxyl_Natural': 0,
        #     'Weight_fraction_ketal_Natural': 0,
        #     'Weight_fraction_BenzeneCycle_Natural': 0,
        #     'Weight_fraction_carboxylic_acid_Natural': 0,
        #     'MW_organic': 0,
        #     'Weight_fraction_esters_organic': 0,
        #     'Weight_fraction_hydroxyl_organic': 0,
        #     'Weight_fraction_phenyl_organic': 0,
        #     'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        #     'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        #     'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        #     'Weight_fraction_of_maleinized_group_organic': 0,
        #     'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        #     'Triazinane_organic': 0,
        #     'Isocyanate_organic': 0,
        #     'Weight_fraction_aldehide_organic': 0,
        #     'Silver': 0,
        #     'CalciumPhosphate': 0,
        #     'HydroxyApatite': 0,
        #     'CalciumCarbonate': 0,
        #     'SiO2': 0,
        #     'Titanium Oxide':0,
        #     'Graphene':0,
        #     'Volcanic Ash': 0,
        #     'd50 (mm)': 0,
        #     'Cation_Exchange_Capacity':0,
        #     'Water Contact Angle':0,	'd50 (micron)':0},index=[0])
        # list_all=data_set_features_all.keys()
        # st.write(data)
        # for data_i in datasets:                
        #     data_=data[data['Dataset']==data_i].copy()
        #     data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
        #     data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
        #     data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
        #     data_inorganic_additive=pd.DataFrame({'Silver':0,'CalciumPhosphate':0,'HydroxyApatite':0,'CalciumCarbonate':0,'SiO2':0,'Titanium Oxide':0,'Graphene':0,'Volcanic Ash':0,'d50 (mm)':0},index=[0])
        #     data_clay_additive=pd.DataFrame({'Cation_Exchange_Capacity':0,'Water Contact Angle':0,'d50 (micron)':0},index=[0])
        #     wt_percentage_clay=0
        #     wt_percentage_inorganic=0
        #     wt_percentage_commercialized=0
            
        #     for data_index,data_row in data_.iterrows():
        #         # st.write('data row is')
        #         st.write(data_row)
        #         if data_row['Material Name']=='P(BS82-co-Pripol18) Lab Made':
        #             data_row['Material Name']='P(BS82-co-Pripol18)'

        #         if data_row['Type']=='Commercialized':
        #             wt_percentage_commercialized+=data_row['Weight_percentage']

        #             for col_names in list_columns_commercialized:
        #                 st.write(col_names)
        #                 data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_commercialized_pool[data_commercialized_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #         elif data_row['Type']=='Natural':
        #             for col_names in list_columns_natural:
        #                 data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #         elif data_row['Type']=='Organic Additive':
        #             for col_names in list_columns_organic_additive:
        #                 data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #         elif data_row['Type']=='Inorganic Additive':
        #             wt_percentage_inorganic+=data_row['Weight_percentage']

        #             for col_names in list_columns_inorganic:

        #                 data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_inorganic_pool[data_inorganic_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #                 if col_names=='d50 (mm)':
        #                     data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(data_row['Weight_percentage'])*data_inorganic_pool[data_inorganic_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #                 else:
        #                     data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(data_row['Weight_percentage'])*data_inorganic_pool[data_inorganic_pool['Materials']==data_row['Material Name']].loc[:,col_names].values

        #         elif data_row['Type']=='Clay Additive':
        #             wt_percentage_clay+=data_row['Weight_percentage']

        #             for col_names in list_columns_clay:
        #                 #st.write(col_names)
        #                 if col_names=='d50 (micron)':
        #                     data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(data_row['Weight_percentage'])*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #                 elif col_names=='Water Contact Angle':
        #                     data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(data_row['Weight_percentage'])*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #                 elif col_names=='Cation_Exchange_Capacity':
        #                     data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(data_row['Weight_percentage'])*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
        #                 else:
        #                     data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(data_row['Weight_percentage']/100)*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values

        #         elif data_row['Material Name']=='Thermoplastic Starch (TPS)':
        #             for col_names in list_columns_organic_additive:
        #                 data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+0.2*(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']=='Glycerol'].loc[:,col_names].values
                    
        #             for col_names in list_columns_natural:
        #                 data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+0.8*(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']=='Starch'].loc[:,col_names].values
        #         elif data_row['Material Name']=='Renol':
        #             for col_names in list_columns_organic_additive:
        #                 data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(1/8)*(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']=='Methyl 9,10-epoxysterate'].loc[:,col_names].values
                    
        #             for col_names in list_columns_natural:
        #                 data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(7/8)*(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']=='Lignin'].loc[:,col_names].values


        #         elif data_row['Material Name']=='Thermoplastic Wheat Flour':
        #             for col_names in list_columns_organic_additive:
        #                 data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(0.23)*(data_row['Weight_percentage']/100)*data_organic_pool[data_organic_pool['Materials']=='Glycerol'].loc[:,col_names].values
                    
        #             for col_names in list_columns_natural:
        #                 data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+0.77*(data_row['Weight_percentage']/100)*data_natural_pool[data_natural_pool['Materials']=='Mantegna Wheat Flour'].loc[:,col_names].values


        #     if wt_percentage_inorganic>0:
        #         data_inorganic_additive.loc[0,'d50 (mm)']=data_inorganic_additive.loc[0,'d50 (mm)']/wt_percentage_inorganic
        #     if wt_percentage_clay>0:
        #         data_clay_additive.loc[0,'Cation_Exchange_Capacity']=data_clay_additive.loc[0,'Cation_Exchange_Capacity']/wt_percentage_clay
        #         data_clay_additive.loc[0,'Water Contact Angle']=data_clay_additive.loc[0,'Water Contact Angle']/wt_percentage_clay
        #         data_clay_additive.loc[0,'d50 (micron)']=data_clay_additive.loc[0,'d50 (micron)']/wt_percentage_clay

        #     if wt_percentage_commercialized>0:
        #         data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']/wt_percentage_commercialized
                
                
        #     for col_ in list_all:
        #         if col_=='Dataset':
        #             data_set_features.loc[0,col_]=data_i
        #         elif col_ in list_columns_commercialized_:
        #             data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        #         elif col_ in list_columns_natural:
        #             data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        #         elif col_ in list_columns_organic_additive:
        #             data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

        #         elif col_ in list_columns_inorganic:
        #             data_set_features.loc[0,col_]=data_inorganic_additive.loc[0,col_]
        #         elif col_ in list_columns_clay:
        #             data_set_features.loc[0,col_]=data_clay_additive.loc[0,col_]


        #     if data_i==datasets[0]:
        #         data_set_all_features=data_set_features.copy()
        #     else:
        #         data_set_all_features=pd.concat([data_set_all_features,data_set_features],ignore_index=True)
                
        # # st.write(data_set_all_features)
        # if 'selected_disintegration_data' not in st.session_state.keys():
        #     with open('./'+st.session_state['username']+'/dataframe_disintegration_old.pkl', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)
        # else:
        #     with open('./'+st.session_state['username']+'/'+st.session_state['selected_disintegration_data']+'.pickle', 'rb') as pickle_file:
        #         data_disintegration_time_dis = pickle.load(pickle_file)



        # negative_indices = data_disintegration_time_dis['Time'] < 0

        # # Convert negative values to zero
        # data_disintegration_time_dis.loc[negative_indices, 'Time'] = 0

        # negative_indices = data_disintegration_time_dis['Disintegration'] < 0

        # # Convert negative values to zero
        # data_disintegration_time_dis.loc[negative_indices, 'Disintegration'] = 0


        # data_disintegration=pd.DataFrame(data_disintegration_time_dis)
        # data_disintegration=data_disintegration[data_disintegration['Dataset'].isin(datasets)]
        # data_time_dis_training=data_disintegration.loc[:,['Time','Surface-to-volume (1/mm)','Disintegration','Dataset']]
        # Data_set_training=pd.merge(data_set_all_features,data_time_dis_training,on='Dataset',how='inner')
        # # st.write(Data_set_training)
        # X_train=Data_set_training.iloc[:,1:-1].values
        # Y_train=Data_set_training.iloc[:,-1].values

        # self.model = xgb.XGBRegressor(n_estimators=150, max_depth=30, eta=.1, subsample=0.6, colsample_bytree=0.8,reg_lambda=10)#,monotone_constraints=string_constraint)#,monotone_constraints=monotone_dic)

        # self.model.fit(X_train,Y_train)
        with open('disintegration_prediction_blend_nat_com_org_inorg.pkl', 'rb') as f:
            self.model=pickle.load( f)

    def predict(self,X_in):
        # st.write('X_in is')
        # st.write(X_in)
        y=self.model.predict(X_in)
        return y




















# data=pd.read_excel('Polymer_Grades.xlsx')
# data_features=pd.read_excel('dataframe_grades_material_information.xlsx')
# materials_list=list(data.Materials)
# names=[]
# for index,row in data.iterrows():
#     # st.write(row)
#     if row['Materials']=='PP':
#         # pass
#         new_name='Polypropylene'
#     elif row['Polymer Grade']=='Not Specified':
#         pass
#     elif row['Materials']=='LDPE':
#         new_name='LDPE'
#     else:
#         new_name=row['Materials']+' '+row['Polymer Grade']
#     names.append(new_name)
# materials=st.multiselect('Select the Materials',options=sorted(names))
# st.write(names)
# st.write(materials)
# Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']


# material_selected_and_wt={}


def natural_blend(materials,Thickness,material_selected_and_wt):
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']

    # st.write('Natural Blend')
    model_=disintegration_prediction_blend_natural_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness

    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])

    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)



    y_i_smoothed=model_with_embeded_parameters_better(90.0)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume

def natural_blend_constraint(materials,Thickness,material_selected_and_wt):
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']

    # st.write('Natural Blend')
    model_=disintegration_prediction_blend_natural_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness

    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])

    y_90=model_with_embeded_parameters_better(90)

    return y_90



def natural_blend_first(materials,Thickness,time_dis,material_selected_and_wt):
    # st.write('Natural Blend')
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']

    model_=disintegration_prediction_blend_natural_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=time_dis
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness

    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])

    y_90=model_with_embeded_parameters_better(time_dis)
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)

    return y_90,y_i_smoothed_list


def commercialized_blend(materials,Thickness,material_selected_and_wt):
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']

    # st.write('commercialized_blend')
    model_=disintegration_prediction_blend_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        # st.write(data_row)

        # data_row=data_features[(data_features['Materials']==first_word) & (data_features['Polymer Grade']==rest_of_string) ]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness

    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])

    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)




    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume



def commercialized_blend_first(materials,Thickness,Time_dis,material_selected_and_wt):
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']

    # st.write('commercialized_blend')
    model_=disintegration_prediction_blend_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        # st.write(data_row)

        # data_row=data_features[(data_features['Materials']==first_word) & (data_features['Polymer Grade']==rest_of_string) ]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=Time_dis
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness

    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])

    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)

    y_90=model_with_embeded_parameters_better(Time_dis)


    # data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

    # Plot the curve using Plotly Express
    # fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return y_90,np.array(list(y_i_smoothed_list))



def commercialized_blend_constraint(materials,Thickness,material_selected_and_wt):
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']

    # st.write('commercialized_blend')
    model_=disintegration_prediction_blend_commercialized()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known

    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        # st.write(data_row)

        # data_row=data_features[(data_features['Materials']==first_word) & (data_features['Polymer Grade']==rest_of_string) ]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness

    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])

    y_90=model_with_embeded_parameters_better(90)
    return y_90

def neat_commercialized(materials,Thickness,material_selected_and_wt):
    # st.write('neat_commercialized')
    # st.write(material_selected_and_wt)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    model_=disintegration_prediction_neat_commercialized(None)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        # st.write('data row is')
        # st.write(data_row)
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_i_smoothed=model_with_embeded_parameters_better(90.0)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    fig.update_layout(
        height=600,  # Adjust the height of the plot as needed
        width=1200    # Adjust the width of the plot as needed
    )

    # tick_values_x=[f'<b>{label}</b>' for label in lca_data_frame['Composite names']]



    return fig,parameters_surface_volume

def neat_commercialized_Validation(materials,S_V,material_selected_and_wt,data_exlude):
    # st.write('neat_commercialized')
    # st.write(material_selected_and_wt)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    model_=disintegration_prediction_neat_commercialized(data_exlude)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=S_V
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_i_smoothed=model_with_embeded_parameters_better(90.0)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    fig.update_layout(
        height=600,  # Adjust the height of the plot as needed
        width=1200    # Adjust the width of the plot as needed
    )

    # tick_values_x=[f'<b>{label}</b>' for label in lca_data_frame['Composite names']]



    return fig,parameters_surface_volume



def neat_commercialized_first(materials,Thickness,Time_dis,material_selected_and_wt):
    # st.write('neat_commercialized')
    # st.write(material_selected_and_wt)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    model_=disintegration_prediction_neat_commercialized(None)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        # st.write('data row is')
        # st.write(data_row)
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(Time_dis)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})

    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)

    y_90=model_with_embeded_parameters_better(Time_dis)


    # tick_values_x=[f'<b>{label}</b>' for label in lca_data_frame['Composite names']]



    return y_90,y_i_smoothed_list




def neat_commercialized_constraint(materials,Thickness,material_selected_and_wt):
    # st.write('neat_commercialized')
    # st.write(material_selected_and_wt)
    materials_functional_groups_commercialized=pd.read_excel('./Materials_information/Commercialized_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Mw','PDI',	'Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    model_=disintegration_prediction_neat_commercialized(None)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_commercialized[(materials_functional_groups_commercialized['Materials']==key)]
        # st.write('data row is')
        # st.write(data_row)
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_90=model_with_embeded_parameters_better(90)
    return y_90









def neat_natural(materials,S_V,material_selected_and_wt,datasets_excluded):
    # st.write(material_selected_and_wt)
    materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']
    model_=disintegration_prediction_neat_natural(datasets_excluded)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    # st.write(material_selected_and_wt)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_neat[(materials_functional_groups_neat['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:-2]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=S_V
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        # if t==0:
        #     st.write(DataFrame)
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_i_smoothed=model_with_embeded_parameters_better(90.0)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume



def neat_natural_first(materials,S_V,Time_dis,material_selected_and_wt):
    # st.write(material_selected_and_wt)
    materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']
    model_=disintegration_prediction_neat_natural(None)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    # st.write(material_selected_and_wt)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_neat[(materials_functional_groups_neat['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:-2]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=S_V
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        # if t==0:
        #     st.write(DataFrame)
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    # y_i_smoothed=model_with_embeded_parameters_better(90.0)
    # y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)

    y_90=model_with_embeded_parameters_better(Time_dis)


    return y_90,y_i_smoothed_list


def neat_natural_constraint(materials,S_V,material_selected_and_wt):
    # st.write(material_selected_and_wt)
    materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']
    model_=disintegration_prediction_neat_natural(None)
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    # st.write(material_selected_and_wt)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_neat[(materials_functional_groups_neat['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:-2]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=S_V
    Time=np.linspace(0,100,1000)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        # if t==0:
        #     st.write(DataFrame)
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    # y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(90.0)

    return y_90

def blend_natural(materials,Thickness,material_selected_and_wt):
    # st.write('blend_natural')
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']

    materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']
    model_=disintegration_prediction_blend_natural()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_neat[(materials_functional_groups_neat['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:-1]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness
    # st.write(DataFrame)
    Time=np.linspace(0,100,1000)
    # st.write(DataFrame)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(DataFrame)
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_i_smoothed=model_with_embeded_parameters_better(90.0)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='<b>Disintegration (%)</b>', title='Disintegration Plot (%) vs Time (day)')
    fig.update_layout(
        height=600,  # Adjust the height of the plot as needed
        width=800    # Adjust the width of the plot as needed
    )
    return fig,parameters_surface_volume

def blend_natural_first(materials,Thickness,Time_dis,material_selected_and_wt):
    # st.write('blend_natural')
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']

    materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']
    model_=disintegration_prediction_blend_natural()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_neat[(materials_functional_groups_neat['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:-1]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness
    # st.write(DataFrame)
    Time=np.linspace(0,100,1000)
    # st.write(DataFrame)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(DataFrame)
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(Time_dis)
    
    return y_90,y_i_smoothed_list
def blend_natural_constraint(materials,Thickness,material_selected_and_wt):
    # st.write('blend_natural')

    materials_functional_groups_neat=pd.read_excel('./Materials_information/Natural_Polymer.xlsx')# the columns in this dataset are: Materials,Weight_fraction_amine_Natural	Weight_fraction_ether_Natural	Weight_fraction_carbonyl_Natural	Weight_fraction_hydroxyl_Natural	Weight_fraction_ketal_Natural	Weight_fraction_BenzeneCycle_Natural	Weight_fraction_carboxylic_acid_Natural Part of structure polymer and known
    Columns_names=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural','Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural','Weight_fraction_carboxylic_acid_Natural']
    model_=disintegration_prediction_blend_natural()
    DataFrame=pd.DataFrame(columns=Columns_names)
    DataFrame.loc[0] = [0] * len(Columns_names)
    for key,value in material_selected_and_wt.items():
        data_row=materials_functional_groups_neat[(materials_functional_groups_neat['Materials']==key)]
        DataFrame.iloc[0,:]=DataFrame.iloc[0,:]+data_row.iloc[:,1:-1]*value/100
    DataFrame['Time']=90
    DataFrame['Surface_to_volume (1/mm)']=2/Thickness
    # st.write(DataFrame)
    Time=np.linspace(0,100,1000)
    # st.write(DataFrame)
    y_predict=[]
    for t in Time:
        DataFrame['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(DataFrame)
        y_predict.append(model_.predict(DataFrame))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(90.0)
    return y_90


#### modify from here. just copied from above

def disintegration_prediction_blend_natural_commercialized(materials,Thickness,material_selected_and_wt):
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    materials_type=pd.read_excel('./Materials_List_All.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_set_features=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_set_features_all=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    list_all=data_set_features_all.keys()

    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    sum_commercialized=0
    for materials_i in materials:
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                # st.write(col_names)
                if col_names in ['Mw','PDI']:                    
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
    
            
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]


    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=2/Thickness
    model_=disintegration_prediction_blend_nat_com()
    # y=class_blend.predict(data_set_features)
    # st.write(DataFrame)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume




def disintegration_prediction_blend_natural_commercialized_first(materials,Thickness,Time_dis,material_selected_and_wt):
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_set_features=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_set_features_all=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    list_all=data_set_features_all.keys()

    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    sum_commercialized=0
    for materials_i in materials:
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                # st.write(col_names)
                if col_names in ['Mw','PDI']:                    
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
    
            
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]


    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=2/Thickness
    model_=disintegration_prediction_blend_nat_com()
    # y=class_blend.predict(data_set_features)
    # st.write(DataFrame)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(Time_dis)
    # data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    # fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return y_90,y_i_smoothed_list






def disintegration_prediction_blend_natural_commercialized_constraint(materials,Thickness,material_selected_and_wt):
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    materials_type=pd.read_excel('./Materials_List_All.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw_polymer_commercialized_Phase':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_set_features=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_set_features_all=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0,'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    list_all=data_set_features_all.keys()

    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    sum_commercialized=0
    for materials_i in materials:
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                # st.write(col_names)
                if col_names in ['Mw','PDI']:                    
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
    
            
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]


    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=2/Thickness
    model_=disintegration_prediction_blend_nat_com()
    # y=class_blend.predict(data_set_features)
    # st.write(DataFrame)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_90=model_with_embeded_parameters_better(90)
    return y_90





def disintegration_prediction_blend_natural_commercialized_organic(materials,S_V,material_selected_and_wt,dataset_excluded):
    
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']


    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']


    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2_Organic']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']



    





        # st.write(material_selected_and_wt)


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']


    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_organic=0
    for materials_i in materials:
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                if col_names in ['Mw','PDI']:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

                # st.write(col_names)
                # st.write('Col name is')
                # st.write(col_names)
                # st.write(data_commercialized_polymer)
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
                else:    
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values
            wt_percentage_organic+=material_selected_and_wt[materials_i]
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized

    # st.write('datasets com and natu')
    # st.table(data_commercialized_polymer)
    # st.table(data_natural_polymer)
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org(dataset_excluded)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume


def disintegration_prediction_blend_natural_commercialized_organic_validation(materials,S_V,material_selected_and_wt,dataset_excluded):
    
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']


    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']


    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2_Organic']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']



    





        # st.write(material_selected_and_wt)


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']


    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_organic=0
    for materials_i in materials:
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                if col_names in ['Mw','PDI']:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

                # st.write(col_names)
                # st.write('Col name is')
                # st.write(col_names)
                # st.write(data_commercialized_polymer)
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
                else:    
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values
            wt_percentage_organic+=material_selected_and_wt[materials_i]
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized

    # st.write('datasets com and natu')
    # st.table(data_commercialized_polymer)
    # st.table(data_natural_polymer)
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org(dataset_excluded)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume



def disintegration_prediction_blend_natural_commercialized_organic_first(materials,S_V,Time_dis,material_selected_and_wt):
    # st.write(material_selected_and_wt)
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']


    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']

        # st.write(material_selected_and_wt)
    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2_Organic']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']


    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_organic=0
    for materials_i in materials:
        # st.write(materials_i)
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                # st.write(col_names)
                if col_names in ['Mw','PDI']:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

                # st.write(col_names)
                # st.write('Col name is')
                # st.write(col_names)
                # st.write(data_commercialized_polymer)
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
                else:    
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values
            wt_percentage_organic+=material_selected_and_wt[materials_i]
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized

    # st.write('datasets com and natu')
    # st.table(data_commercialized_polymer)
    # st.table(data_natural_polymer)
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org(None)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(Time_dis)
    # data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # # Plot the curve using Plotly Express
    # fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return y_90,y_i_smoothed_list









def disintegration_prediction_blend_natural_commercialized_organic_constraint(materials,S_V,material_selected_and_wt):
    
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']


    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']



    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']



        # st.write(material_selected_and_wt)


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']


    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_organic=0
    for materials_i in materials:
        # st.write('type is')
        # st.write(list(materials_type[materials_type['Materials']==materials_i]['Type'])[0])
        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            for col_names in list_columns_commercialized:
                if col_names in ['Mw','PDI']:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values

                # st.write(col_names)
                # st.write('Col name is')
                # st.write(col_names)
                # st.write(data_commercialized_polymer)
                else:
                    data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
            sum_commercialized+=material_selected_and_wt[materials_i]

        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
                else:    
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values
            wt_percentage_organic+=material_selected_and_wt[materials_i]
    data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic
    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Mw']=data_commercialized_polymer.loc[0,'Mw']/100
        data_commercialized_polymer.loc[0,'PDI']=data_commercialized_polymer.loc[0,'PDI']/sum_commercialized

    # st.write('datasets com and natu')
    # st.table(data_commercialized_polymer)
    # st.table(data_natural_polymer)
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org(None)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        # if t==0:
        #     st.write('DataFrame is')
        #     st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_90=model_with_embeded_parameters_better(90)
    # data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # # Plot the curve using Plotly Express
    # fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return y_90





def disintegration_prediction_blend_natural_commercialized_organic_inorganic(materials,S_V,material_selected_and_wt,dataset_excluded):
    
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']
        # st.write(material_selected_and_wt)

    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']
    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']
    if 'E-Material-1' in materials_list:
        if 'E-Material-1-Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Material-1-Commercialized_Polymer']=material_selected_and_wt['E-Material-1-Commercialized_Polymer']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Material-1-Commercialized_Polymer']=material_selected_and_wt['E-Material-1']*1

        if 'E-Material-1-Organic_Additive' in materials_list:
            material_selected_and_wt['E-Material-1-Organic_Additive']=material_selected_and_wt['E-Material-1-Organic_Additive']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Material-1-Organic_Additive']=material_selected_and_wt['E-Material-1']*1
 

        if 'E-Mateiral-1-Inorganic_Materials_not_clay_additive' in materials_list:
            material_selected_and_wt['E-Mateiral-1-Inorganic_Materials_not_clay_additive']=material_selected_and_wt['E-Mateiral-1-Inorganic_Materials_not_clay_additive']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Material-1']*1
 
        if 'E-Mateirial-1-Natural_Polymer' in materials_list:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Mateirial-1-Natural_Polymer']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Material-1']*1
        del material_selected_and_wt['E-Material-1']


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')
    data_inorganic_pool=pd.read_excel('./Materials_Information/Inorganic_Materials_not_clay_additive.xlsx')
    data_clay_pool=pd.read_excel('./Materials_Information/Clay_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])

    data_inorganic_additive=pd.DataFrame({'Silver':0,'CalciumPhosphate':0,'HydroxyApatite':0,'CalciumCarbonate':0,'SiO2':0,'Titanium Oxide':0,'Graphene':0,'Volcanic Ash':0,'d50 (mm)':0},index=[0])
    data_clay_additive=pd.DataFrame({'Cation_Exchange_Capacity':0,'Water Contact Angle':0,'d50 (micron)':0},index=[0])







    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']

    list_columns_inorganic=['Silver','CalciumPhosphate','HydroxyApatite','CalciumCarbonate','SiO2','Titanium Oxide','Graphene','Volcanic Ash','d50 (mm)']
    list_columns_clay=['Cation_Exchange_Capacity','Water Contact Angle',	'd50 (micron)']

    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0,
        'Silver': 0,
        'CalciumPhosphate': 0,
        'HydroxyApatite': 0,
        'CalciumCarbonate': 0,
        'SiO2': 0,
        'Titanium Oxide':0,
        'Graphene':0,
        'Volcanic Ash': 0,
        'd50 (mm)': 0,
        'Cation_Exchange_Capacity':0,
        'Water Contact Angle':0,	'd50 (micron)':0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0,
        'Silver': 0,
        'CalciumPhosphate': 0,
        'HydroxyApatite': 0,
        'CalciumCarbonate': 0,
        'SiO2': 0,
        'Titanium Oxide':0,
        'Graphene':0,
        'Volcanic Ash': 0,
        'd50 (mm)': 0,
        'Cation_Exchange_Capacity':0,
        'Water Contact Angle':0,	'd50 (micron)':0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_clay=0
    wt_percentage_organic=0
    wt_percentage_inorganic=0
    for materials_i in materials:
        # st.write(materials_i)

        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            sum_commercialized+=material_selected_and_wt[materials_i]

            for col_names in list_columns_commercialized:
                data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        

                else:
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
            wt_percentage_organic+=material_selected_and_wt[materials_i]
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Inorganic Additive'):
            wt_percentage_inorganic+=material_selected_and_wt[materials_i]

            for col_names in list_columns_inorganic:
                # st.write(col_names)
                # st.write(materials_i)
                # st.write(data_inorganic_additive)

                if col_names=='d50 (mm)':
                    data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_inorganic_pool[data_inorganic_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_inorganic_pool[data_inorganic_pool['Materials']==materials_i].loc[:,col_names].values
            
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Clay Additive'):
            wt_percentage_clay+=material_selected_and_wt[materials_i]

            for col_names in list_columns_clay:
                # data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values


                if col_names=='d50 (micron)':

                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                    # data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                elif col_names=='Water Contact Angle':
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                elif col_names=='Cation_Exchange_Capacity':
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values

            
    if wt_percentage_clay>0:
        data_clay_additive.loc[0,'Cation_Exchange_Capacity']=data_clay_additive.loc[0,'Cation_Exchange_Capacity']/wt_percentage_clay
        data_clay_additive.loc[0,'Water Contact Angle']=data_clay_additive.loc[0,'Water Contact Angle']/wt_percentage_clay
        data_clay_additive.loc[0,'d50 (micron)']=data_clay_additive.loc[0,'d50 (micron)']/wt_percentage_clay

    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized


    if wt_percentage_inorganic>0:
        data_inorganic_additive.loc[0,'d50 (mm)']=data_inorganic_additive.loc[0,'d50 (mm)']/wt_percentage_inorganic

    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic






    # data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

        elif col_ in list_columns_inorganic:
            data_set_features.loc[0,col_]=data_inorganic_additive.loc[0,col_]

        elif col_ in list_columns_clay:
            data_set_features.loc[0,col_]=data_clay_additive.loc[0,col_]





            
    # st.write(data_set_features)
    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org_inorg(dataset_excluded)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        if t==0:
            pass
        #     st.write('DataFrame is')
            # st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # Plot the curve using Plotly Express
    fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return fig,parameters_surface_volume









def disintegration_prediction_blend_natural_commercialized_organic_inorganic_first(materials,S_V,Time_dis,material_selected_and_wt):
    
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']
        # st.write(material_selected_and_wt)

    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']
    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']
    if 'E-Material-1' in materials_list:
        if 'E-Material-1-Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Material-1-Commercialized_Polymer']=material_selected_and_wt['E-Material-1-Commercialized_Polymer']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Material-1-Commercialized_Polymer']=material_selected_and_wt['E-Material-1']*1

        if 'E-Material-1-Organic_Additive' in materials_list:
            material_selected_and_wt['E-Material-1-Organic_Additive']=material_selected_and_wt['E-Material-1-Organic_Additive']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Material-1-Organic_Additive']=material_selected_and_wt['E-Material-1']*1
 

        if 'E-Mateiral-1-Inorganic_Materials_not_clay_additive' in materials_list:
            material_selected_and_wt['E-Mateiral-1-Inorganic_Materials_not_clay_additive']=material_selected_and_wt['E-Mateiral-1-Inorganic_Materials_not_clay_additive']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Material-1']*1
 
        if 'E-Mateirial-1-Natural_Polymer' in materials_list:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Mateirial-1-Natural_Polymer']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Material-1']*1
        del material_selected_and_wt['E-Material-1']


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')
    data_inorganic_pool=pd.read_excel('./Materials_Information/Inorganic_Materials_not_clay_additive.xlsx')
    data_clay_pool=pd.read_excel('./Materials_Information/Clay_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])

    data_inorganic_additive=pd.DataFrame({'Silver':0,'CalciumPhosphate':0,'HydroxyApatite':0,'CalciumCarbonate':0,'SiO2':0,'Titanium Oxide':0,'Graphene':0,'Volcanic Ash':0,'d50 (mm)':0},index=[0])
    data_clay_additive=pd.DataFrame({'Cation_Exchange_Capacity':0,'Water Contact Angle':0,'d50 (micron)':0},index=[0])







    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']

    list_columns_inorganic=['Silver','CalciumPhosphate','HydroxyApatite','CalciumCarbonate','SiO2','Titanium Oxide','Graphene','Volcanic Ash','d50 (mm)']
    list_columns_clay=['Cation_Exchange_Capacity','Water Contact Angle',	'd50 (micron)']

    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0,
        'Silver': 0,
        'CalciumPhosphate': 0,
        'HydroxyApatite': 0,
        'CalciumCarbonate': 0,
        'SiO2': 0,
        'Titanium Oxide':0,
        'Graphene':0,
        'Volcanic Ash': 0,
        'd50 (mm)': 0,
        'Cation_Exchange_Capacity':0,
        'Water Contact Angle':0,	'd50 (micron)':0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0,
        'Silver': 0,
        'CalciumPhosphate': 0,
        'HydroxyApatite': 0,
        'CalciumCarbonate': 0,
        'SiO2': 0,
        'Titanium Oxide':0,
        'Graphene':0,
        'Volcanic Ash': 0,
        'd50 (mm)': 0,
        'Cation_Exchange_Capacity':0,
        'Water Contact Angle':0,	'd50 (micron)':0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_clay=0
    wt_percentage_organic=0
    wt_percentage_inorganic=0
    for materials_i in materials:
        # st.write(materials_i)

        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            sum_commercialized+=material_selected_and_wt[materials_i]

            for col_names in list_columns_commercialized:
                data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        

                else:
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
            wt_percentage_organic+=material_selected_and_wt[materials_i]
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Inorganic Additive'):
            wt_percentage_inorganic+=material_selected_and_wt[materials_i]

            for col_names in list_columns_inorganic:
                # st.write(col_names)
                # st.write(materials_i)
                # st.write(data_inorganic_additive)

                if col_names=='d50 (mm)':
                    data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_inorganic_pool[data_inorganic_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_inorganic_pool[data_inorganic_pool['Materials']==materials_i].loc[:,col_names].values
            
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Clay Additive'):
            wt_percentage_clay+=material_selected_and_wt[materials_i]

            for col_names in list_columns_clay:
                # data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values


                if col_names=='d50 (micron)':

                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                    # data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                elif col_names=='Water Contact Angle':
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                elif col_names=='Cation_Exchange_Capacity':
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values

            
    if wt_percentage_clay>0:
        data_clay_additive.loc[0,'Cation_Exchange_Capacity']=data_clay_additive.loc[0,'Cation_Exchange_Capacity']/wt_percentage_clay
        data_clay_additive.loc[0,'Water Contact Angle']=data_clay_additive.loc[0,'Water Contact Angle']/wt_percentage_clay
        data_clay_additive.loc[0,'d50 (micron)']=data_clay_additive.loc[0,'d50 (micron)']/wt_percentage_clay

    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized


    if wt_percentage_inorganic>0:
        data_inorganic_additive.loc[0,'d50 (mm)']=data_inorganic_additive.loc[0,'d50 (mm)']/wt_percentage_inorganic

    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic






    # data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

        elif col_ in list_columns_inorganic:
            data_set_features.loc[0,col_]=data_inorganic_additive.loc[0,col_]

        elif col_ in list_columns_clay:
            data_set_features.loc[0,col_]=data_clay_additive.loc[0,col_]





            
    # st.write(data_set_features)
    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org_inorg(None)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        if t==0:
            pass
        #     st.write('DataFrame is')
            # st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_i_smoothed_list=map(model_with_embeded_parameters_better,Time)
    y_90=model_with_embeded_parameters_better(Time_dis)
    # data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # # Plot the curve using Plotly Express
    # fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return y_90,y_i_smoothed_list




def disintegration_prediction_blend_natural_commercialized_organic_inorganic_constraint(materials,S_V,material_selected_and_wt):
    
    materials_list=list(material_selected_and_wt.keys())
    if 'Thermoplastic Starch (TPS)' in materials_list:
        if 'Starch' in materials_list:
            material_selected_and_wt['Starch']=material_selected_and_wt['Starch']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8
        else:
            material_selected_and_wt['Starch']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.8

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Starch (TPS)']*0.2
        del material_selected_and_wt['Thermoplastic Starch (TPS)']
        # st.write(material_selected_and_wt)

    if 'Renol' in materials_list:
        if 'Lignin' in materials_list:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Lignin']+material_selected_and_wt['Renol']*7/8
        else:
            material_selected_and_wt['Lignin']=material_selected_and_wt['Renol']*7/8

        if 'Methyl 9,10-epoxysterate' in materials_list:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Methyl 9,10-epoxysterate']+material_selected_and_wt['Renol']*1/8
        else:
            material_selected_and_wt['Methyl 9,10-epoxysterate']=material_selected_and_wt['Renol']*1/8
        del material_selected_and_wt['Renol']

    if 'Thermoplastic Wheat Flour' in materials_list:
        if 'Mantegna Wheat Flour' in materials_list:
            material_selected_and_wt['Mantegna Wheat Flour']=material_selected_and_wt['Mantegna Wheat Flour']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.77
        else:
            material_selected_and_wt['Thermoplastic Wheat Flour']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.77

        if 'Glycerol' in materials_list:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Glycerol']+material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        else:
            material_selected_and_wt['Glycerol']=material_selected_and_wt['Thermoplastic Wheat Flour']*0.23
        del material_selected_and_wt['Thermoplastic Wheat Flour']
    if 'E-Elastomer-2' in materials_list:
        if 'E-Elastomer-2_Natural_Polymers' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2_Natural_Polymers']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Natural_Polymers']=material_selected_and_wt['E-Elastomer-2']

        if 'E-Elastomer-2_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2_Organic']+material_selected_and_wt['E-Elastomer-2']*1
        else:
            material_selected_and_wt['E-Elastomer-2_Organic']=material_selected_and_wt['E-Elastomer-2']*1
        del material_selected_and_wt['E-Elastomer-2_Organic']

    if 'E-Elastomer-3' in materials_list:
        if 'E-Elastomer-3_Organic' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3_Organic']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Organic']=material_selected_and_wt['E-Elastomer-3']

        if 'E-Elastomer-3_Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']+material_selected_and_wt['E-Elastomer-3']*1
        else:
            material_selected_and_wt['E-Elastomer-3_Commercialized_Polymer']=material_selected_and_wt['E-Elastomer-3']*1
        del material_selected_and_wt['E-Elastomer-3']
    if 'E-Material-1' in materials_list:
        if 'E-Material-1-Commercialized_Polymer' in materials_list:
            material_selected_and_wt['E-Material-1-Commercialized_Polymer']=material_selected_and_wt['E-Material-1-Commercialized_Polymer']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Material-1-Commercialized_Polymer']=material_selected_and_wt['E-Material-1']*1

        if 'E-Material-1-Organic_Additive' in materials_list:
            material_selected_and_wt['E-Material-1-Organic_Additive']=material_selected_and_wt['E-Material-1-Organic_Additive']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Material-1-Organic_Additive']=material_selected_and_wt['E-Material-1']*1
 

        if 'E-Mateiral-1-Inorganic_Materials_not_clay_additive' in materials_list:
            material_selected_and_wt['E-Mateiral-1-Inorganic_Materials_not_clay_additive']=material_selected_and_wt['E-Mateiral-1-Inorganic_Materials_not_clay_additive']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Material-1']*1
 
        if 'E-Mateirial-1-Natural_Polymer' in materials_list:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Mateirial-1-Natural_Polymer']+material_selected_and_wt['E-Material-1']*1
        else:
            material_selected_and_wt['E-Mateirial-1-Natural_Polymer']=material_selected_and_wt['E-Material-1']*1
        del material_selected_and_wt['E-Material-1']


    materials=list(material_selected_and_wt.keys())
    materials_type=pd.read_excel('./Materials_Information/Materials_info_disintegration.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])

    list_columns_commercialized=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate']
    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural',
    'Weight_fraction_ether_Natural',
    'Weight_fraction_carbonyl_Natural',
    'Weight_fraction_hydroxyl_Natural',
    'Weight_fraction_ketal_Natural',
    'Weight_fraction_BenzeneCycle_Natural',
    'Weight_fraction_carboxylic_acid_Natural']
    data_commercialized_pool=pd.read_excel('./Materials_Information/Commercialized_Polymer.xlsx')
    data_natural_pool=pd.read_excel('./Materials_Information/Natural_Polymer.xlsx')
    data_organic_pool=pd.read_excel('./Materials_Information/Organic_Additive.xlsx')
    data_inorganic_pool=pd.read_excel('./Materials_Information/Inorganic_Materials_not_clay_additive.xlsx')
    data_clay_pool=pd.read_excel('./Materials_Information/Clay_Additive.xlsx')


    data_commercialized_polymer=pd.DataFrame({'Mw':0,'PDI':0,'Weight_fraction_ester':0,'Weight_fraction_ether':0,'Weight_fraction_BenzeneCycle':0,'Weight_fraction_urethane':0,'Weight_fraction_hydroxyl':0,'Weight_fraction_phthalate':0,'Weight_percentage_Commercialized Polymers':0},index=[0])
    data_natural_polymer=pd.DataFrame({'Weight_fraction_ester_Natural':0,'Weight_fraction_amine_Natural':0,'Weight_fraction_ether_Natural':0,'Weight_fraction_carbonyl_Natural':0,'Weight_fraction_hydroxyl_Natural':0,'Weight_fraction_ketal_Natural':0,'Weight_fraction_BenzeneCycle_Natural':0,'Weight_fraction_carboxylic_acid_Natural':0},index=[0])
    data_organic_additive=pd.DataFrame({'MW_organic': 0,'Weight_fraction_esters_organic': 0,'Weight_fraction_hydroxyl_organic': 0,'Weight_fraction_phenyl_organic': 0,'Weight_fraction_of_oxygens_in_ether_form_organic': 0,'Weight_fraction_of_carbons_in_double_bond_organic': 0,'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,'Weight_fraction_of_maleinized_group_organic': 0,'Weight_fraction_of_carboxilic_acid_groups_organic': 0,'Triazinane_organic': 0,'Isocyanate_organic': 0,'Weight_fraction_aldehide_organic': 0},index=[0])

    data_inorganic_additive=pd.DataFrame({'Silver':0,'CalciumPhosphate':0,'HydroxyApatite':0,'CalciumCarbonate':0,'SiO2':0,'Titanium Oxide':0,'Graphene':0,'Volcanic Ash':0,'d50 (mm)':0},index=[0])
    data_clay_additive=pd.DataFrame({'Cation_Exchange_Capacity':0,'Water Contact Angle':0,'d50 (micron)':0},index=[0])







    list_columns_commercialized_=['Mw','PDI','Weight_fraction_ester','Weight_fraction_ether','Weight_fraction_BenzeneCycle','Weight_fraction_urethane','Weight_fraction_hydroxyl','Weight_fraction_phthalate','Weight_percentage_Commercialized Polymers']
    list_columns_natural=['Weight_fraction_ester_Natural','Weight_fraction_amine_Natural','Weight_fraction_ether_Natural','Weight_fraction_carbonyl_Natural','Weight_fraction_hydroxyl_Natural', 'Weight_fraction_ketal_Natural','Weight_fraction_BenzeneCycle_Natural', 'Weight_fraction_carboxylic_acid_Natural']
    list_columns_organic_additive=['MW_organic', 'Weight_fraction_esters_organic','Weight_fraction_hydroxyl_organic','Weight_fraction_phenyl_organic','Weight_fraction_of_oxygens_in_ether_form_organic','Weight_fraction_of_carbons_in_double_bond_organic','Weight_fraction_of_oxygen_in_epoxy_form_organic',
                'Weight_fraction_of_maleinized_group_organic',
                'Weight_fraction_of_carboxilic_acid_groups_organic',
                'Triazinane_organic',
                'Isocyanate_organic',
                'Weight_fraction_aldehide_organic']

    list_columns_inorganic=['Silver','CalciumPhosphate','HydroxyApatite','CalciumCarbonate','SiO2','Titanium Oxide','Graphene','Volcanic Ash','d50 (mm)']
    list_columns_clay=['Cation_Exchange_Capacity','Water Contact Angle',	'd50 (micron)']

    data_set_features=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0,
        'Silver': 0,
        'CalciumPhosphate': 0,
        'HydroxyApatite': 0,
        'CalciumCarbonate': 0,
        'SiO2': 0,
        'Titanium Oxide':0,
        'Graphene':0,
        'Volcanic Ash': 0,
        'd50 (mm)': 0,
        'Cation_Exchange_Capacity':0,
        'Water Contact Angle':0,	'd50 (micron)':0},index=[0])
    data_set_features_all=pd.DataFrame({
        'Mw': 0,
        'PDI': 0,
        'Weight_fraction_ester': 0,
        'Weight_fraction_ether': 0,
        'Weight_fraction_BenzeneCycle': 0,
        'Weight_fraction_urethane': 0,
        'Weight_fraction_hydroxyl':0,
        'Weight_fraction_phthalate':0,
        'Weight_percentage_Commercialized Polymers': 0,
        'Weight_fraction_ester_Natural':0,
        'Weight_fraction_amine_Natural': 0,
        'Weight_fraction_ether_Natural': 0,
        'Weight_fraction_carbonyl_Natural': 0,
        'Weight_fraction_hydroxyl_Natural': 0,
        'Weight_fraction_ketal_Natural': 0,
        'Weight_fraction_BenzeneCycle_Natural': 0,
        'Weight_fraction_carboxylic_acid_Natural': 0,
        'MW_organic': 0,
        'Weight_fraction_esters_organic': 0,
        'Weight_fraction_hydroxyl_organic': 0,
        'Weight_fraction_phenyl_organic': 0,
        'Weight_fraction_of_oxygens_in_ether_form_organic': 0,
        'Weight_fraction_of_carbons_in_double_bond_organic': 0,
        'Weight_fraction_of_oxygen_in_epoxy_form_organic': 0,
        'Weight_fraction_of_maleinized_group_organic': 0,
        'Weight_fraction_of_carboxilic_acid_groups_organic': 0,
        'Triazinane_organic': 0,
        'Isocyanate_organic': 0,
        'Weight_fraction_aldehide_organic': 0,
        'Silver': 0,
        'CalciumPhosphate': 0,
        'HydroxyApatite': 0,
        'CalciumCarbonate': 0,
        'SiO2': 0,
        'Titanium Oxide':0,
        'Graphene':0,
        'Volcanic Ash': 0,
        'd50 (mm)': 0,
        'Cation_Exchange_Capacity':0,
        'Water Contact Angle':0,	'd50 (micron)':0},index=[0])




    list_all=data_set_features_all.keys()











    sum_commercialized=0
    wt_percentage_clay=0
    wt_percentage_organic=0
    wt_percentage_inorganic=0
    for materials_i in materials:
        # st.write(materials_i)

        if (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Commercialized'):
            sum_commercialized+=material_selected_and_wt[materials_i]

            for col_names in list_columns_commercialized:
                data_commercialized_polymer.loc[0,col_names]=data_commercialized_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_commercialized_pool[data_commercialized_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Natural'):
            for col_names in list_columns_natural:
                data_natural_polymer.loc[0,col_names]=data_natural_polymer.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_natural_pool[data_natural_pool['Materials']==materials_i].loc[:,col_names].values
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Organic Additive'):
            for col_names in list_columns_organic_additive:
                if col_names=='MW_organic':
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        

                else:
                    data_organic_additive.loc[0,col_names]=data_organic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_organic_pool[data_organic_pool['Materials']==materials_i].loc[:,col_names].values        
            wt_percentage_organic+=material_selected_and_wt[materials_i]
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Inorganic Additive'):
            wt_percentage_inorganic+=material_selected_and_wt[materials_i]

            for col_names in list_columns_inorganic:
                # st.write(col_names)
                # st.write(materials_i)
                # st.write(data_inorganic_additive)

                if col_names=='d50 (mm)':
                    data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_inorganic_pool[data_inorganic_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_inorganic_additive.loc[0,col_names]=data_inorganic_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_inorganic_pool[data_inorganic_pool['Materials']==materials_i].loc[:,col_names].values
            
        elif (list(materials_type[materials_type['Materials']==materials_i]['Type'])[0]=='Clay Additive'):
            wt_percentage_clay+=material_selected_and_wt[materials_i]

            for col_names in list_columns_clay:
                # data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values


                if col_names=='d50 (micron)':

                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                    # data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]*data_clay_pool[data_clay_pool['Materials']==data_row['Material Name']].loc[:,col_names].values
                elif col_names=='Water Contact Angle':
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                elif col_names=='Cation_Exchange_Capacity':
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i])*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values
                else:
                    data_clay_additive.loc[0,col_names]=data_clay_additive.loc[0,col_names]+(material_selected_and_wt[materials_i]/100)*data_clay_pool[data_clay_pool['Materials']==materials_i].loc[:,col_names].values

            
    if wt_percentage_clay>0:
        data_clay_additive.loc[0,'Cation_Exchange_Capacity']=data_clay_additive.loc[0,'Cation_Exchange_Capacity']/wt_percentage_clay
        data_clay_additive.loc[0,'Water Contact Angle']=data_clay_additive.loc[0,'Water Contact Angle']/wt_percentage_clay
        data_clay_additive.loc[0,'d50 (micron)']=data_clay_additive.loc[0,'d50 (micron)']/wt_percentage_clay

    if sum_commercialized>0:
        data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized


    if wt_percentage_inorganic>0:
        data_inorganic_additive.loc[0,'d50 (mm)']=data_inorganic_additive.loc[0,'d50 (mm)']/wt_percentage_inorganic

    if wt_percentage_organic>0:
        data_organic_additive.loc[0,'MW_organic']=data_organic_additive.loc[0,'MW_organic']/wt_percentage_organic






    # data_commercialized_polymer.loc[0,'Weight_percentage_Commercialized Polymers']=sum_commercialized
    for col_ in list_all:
        if col_ in list_columns_commercialized_:
            data_set_features.loc[0,col_]=data_commercialized_polymer.loc[0,col_]
        elif col_ in list_columns_natural:
            data_set_features.loc[0,col_]=data_natural_polymer.loc[0,col_]
        elif col_ in list_columns_organic_additive:
            data_set_features.loc[0,col_]=data_organic_additive.loc[0,col_]

        elif col_ in list_columns_inorganic:
            data_set_features.loc[0,col_]=data_inorganic_additive.loc[0,col_]

        elif col_ in list_columns_clay:
            data_set_features.loc[0,col_]=data_clay_additive.loc[0,col_]





            
    # st.write(data_set_features)
    data_set_features['Time']=90
    data_set_features['Surface_to_volume (1/mm)']=S_V
    # st.write('Dataset features are')
    # st.write(data_set_features)
    # st.write(data_set_features)
    model_=disintegration_prediction_blend_nat_com_org_inorg(None)
    Time=np.linspace(0,100,1000)
    # st.write(data_set_features)
    y_predict=[]
    for t in Time:
        data_set_features['Time']=t
        if t==0:
            pass
        #     st.write('DataFrame is')
            # st.write(data_set_features)
        y_predict.append(model_.predict(data_set_features))
    parameters_surface_volume,pcov=curve_fit(model_time_surface_volume_ratio,np.array(Time).reshape((-1,)),np.array(y_predict).reshape((-1,)),p0=np.array([3.46,50]),bounds=([0,0], [50, 10000]),maxfev=10000)
    model_with_embeded_parameters_better=lambda x_time:model_time_surface_volume_ratio(x_time,parameters_surface_volume[0],parameters_surface_volume[1])
    y_90=model_with_embeded_parameters_better(90)
    # data = pd.DataFrame({'Time (day)': Time, 'Disintegration (%)': y_i_smoothed_list})
    # # Plot the curve using Plotly Express
    # fig = px.line(data, x='Time (day)', y='Disintegration (%)', title='Disintegration Plot (%) vs Time (day)')
    return y_90

