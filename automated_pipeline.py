# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:52:58 2021

@author: Sahil
"""
import os
import base64
import evalml
import streamlit as st
import pandas as pd
from evalml.automl import AutoMLSearch
import time
from autoscraper import AutoScraper




col1,col2,col3=st.beta_columns([1,3,1])
col2.title("Automated ML Pipeline")
st.subheader("Single solution for all of your problem with hyperparameter optimization")

def about():
    st.sidebar.subheader("Automl ! A Shortcut for ML Engineers")
    with st.sidebar:
        '''Automated Machine Learning provides methods and processes to make Machine Learning available for non-Machine Learning experts, to improve efficiency of Machine Learning and to accelerate research on Machine Learning.'''
        '''Making a science of model search argues that the performance of a given technique depends on both the fundamental quality of the algorithm and the details of its tuning and that it is sometimes difficult to know whether a given technique is genuinely better, or simply better tuned. To improve the situation, Bergstra et al. propose reporting results obtained by tuning all algorithms with the same hyperparameter optimization toolkit. Sculley et al.’s recent ICLR workshop paper Winner’s Curse argues in the same direction and gives recent examples in which correct hyperperameter optimization of baselines improved over the latest state-of-the-art results and newly proposed methods.

Hyperparameter optimization and algorithm configuration provide methods to automate the tedious, time-consuming and error-prone process of tuning hyperparameters to new tasks at hand and provide software packages implement the suggestion from Bergstra et al.’s Making a science of model search. '''
#Autoscapper
#url data input method

def url_data():
    about()
    st.info("This feature has limited functionality")
    url=st.text_input("Webpage URL",help="Enter a url where your data is placed")
    if url=="":
        st.info("Please enter a valid input to get started")
        st.stop()
    
    #getting data Column names as user input
    column_name=st.text_input("enter candidadte",key="value")
    value_list=column_name.split(",")
    
    #getting data example for refferances
    candidate=st.text_input("Candidate example value",key="candidates",help="use ; as seperator to enter another value")
    items_list=candidate.split(";")
    #st.write(items)
    
# create object
    scraper = AutoScraper()
# feeding for scraping
    final_result = scraper.build(url,items_list)
# display result
    #st.write(final_result)
    
    results=scraper.get_result_similar("https://trak.in/india-startup-funding-investment-2015/",grouped=True,keep_order=True)
    result={}
    for key,value in results.items():
        if value not in result.values():
            result[key]=value
            
    orient_df=pd.DataFrame.from_dict(result,orient="index")
    df=orient_df.transpose()
    st.write(df)
    df.columns=value_list
    df.fillna(value=pd.np.nan,inplace=True)
    st.write(df)
    
    cols=df.columns.tolist()
    col1,col2=st.beta_columns(2)
    target=col1.selectbox("Select Target", cols,key="target")
    x=df.drop(columns=target)
    y=df[target]

    
    typelist=['binary','multiclass','regression','time series regression','time series multiclass','time series binary']
    p_type=col2.selectbox("Select problem type",typelist,key="p_type")
        

    x_train,x_test,y_train,y_test=evalml.preprocessing.split_data(x,y,problem_type=p_type)

    #its training time
# =============================================================================
#             st.spinner()
#             with st.spinner(text='In progress'):
#                 st.info("Wait while we are selecting a best algoritham for your problem..Hold your breath.")
#                 time.sleep(20)
# =============================================================================
    automl = AutoMLSearch(X_train=x_train, y_train=y_train, problem_type=p_type)
    automl.search()


    rank=automl.rankings
# to see the rankings
# =============================================================================
# if st.button("Check rankings",key="rank"):
#     st.write(rank)
# =============================================================================

#checking best pipeline     ###############################################################

    best_pipeline=automl.best_pipeline
    description=automl.describe_pipeline(automl.rankings.iloc[0]["id"])


### Evaluate on hold out data
    problem_list=['binary','time series binary']
    problem_list2=['multiclass','time series multiclass']
            
    if p_type in problem_list:
        best_pipeline.score(x_test, y_test, objectives=["auc","f1","Precision","Recall"])

        automl_tunned = AutoMLSearch(X_train=x_train, y_train=y_train,
                                         problem_type=p_type,
                                         objective='auc',
                                         additional_objectives=['f1', 'precision'],
                                         max_batches=1,
                                         optimize_thresholds=True)

        automl_tunned.search()

        tunned_rankings=automl_tunned.rankings

        automl_tunned.describe_pipeline(automl_tunned.rankings.iloc[0]["id"])

        tunned_pipeline= automl_tunned.best_pipeline

        tunned_pipeline.score(x_test, y_test,  objectives=["auc"])

        pred=tunned_pipeline.predict_proba(x_test).to_dataframe()


# for multiclass type problem
    elif p_type in problem_list2:
        best_pipeline.score(x_test, y_test, objectives=["log loss multiclass","MCC multiclass","accuracy multiclass"])

        automl_tunned = AutoMLSearch(X_train=x_train, y_train=y_train,
                                         problem_type=p_type,
                                         objective="log loss multiclass",
                                         additional_objectives=['MCC multiclass', 'accuracy multiclass'],
                                         max_batches=1,
                                         optimize_thresholds=True)

        automl_tunned.search()

        tunned_rankings=automl_tunned.rankings

        automl_tunned.describe_pipeline(automl_tunned.rankings.iloc[0]["id"])

        tunned_pipeline= automl_tunned.best_pipeline

        tunned_pipeline.score(x_test, y_test,  objectives=["log loss multiclass"])

        pred=tunned_pipeline.predict(x_test).to_series()

    
# for regression type problems
    else:
                best_pipeline.score(x_test, y_test, objectives=["r2","MSE","MAE","Root Mean Squared Error"])
                automl_tunned = AutoMLSearch(X_train=x_train, y_train=y_train,
                                         problem_type=p_type,
                                         objective='r2',
                                         additional_objectives=['Root Mean Squared Error', 'MSE','MAE'],
                                         max_batches=1,
                                         optimize_thresholds=True)

                automl_tunned.search()

                tunned_rankings=automl_tunned.rankings

                automl_tunned.describe_pipeline(automl_tunned.rankings.iloc[0]["id"])

                tunned_pipeline= automl_tunned.best_pipeline

                tunned_pipeline.score(x_test, y_test,  objectives=["r2"])

                tunned_pipeline.fit(x_train,y_train)
                    
                pred=tunned_pipeline.predict(x_test).to_series()
    col1,col2,col3=st.beta_columns([1,1,1])        
    if col2.button("Predict Results",key="output",help="shows results"):
            st.spinner()
            with st.spinner(text='In progress'):
                 st.info("Wait while we are selecting a best algoritham for your problem..Hold your breath.")
                 time.sleep(20)
            st.info("Done. Here you go.")
            st.write(pred)

    col11,col12=st.beta_columns([3,1])
    with col11:
        with st.beta_expander("Compare Models"):
                st.write(rank)
        
    with col12:
        with st.beta_expander("Best Pipeline"):
                st.success(best_pipeline)
                
        
    
    
 

# =============================================================================
#direct file upload method
    
def file_data():


    uploaded_file = st.file_uploader("Upload Files",type=['csv','xls','xlxs','json','png','jpeg'])
# =============================================================================
# if uploaded_file is not None:
#     file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
#     st.write(file_details)
# =============================================================================

#geeting the file type to read the file as a dataframe

    if uploaded_file is None:
        st.info("Upload a dataset first to start making predictions")
    else:
        
        filename=str(uploaded_file.name)
        file_type=filename.split('.',2)

        if file_type[1]=='csv':
            df=pd.read_csv(filename)
    
        elif file_type[1]=='xls' or 'xlsx':
            df=pd.read_excel(filename)
            st.write(df.head())    
#wait for json type in new update 


    

        cols=df.columns.tolist()
        col1,col2=st.beta_columns(2)
        target=col1.selectbox("Select Target", cols,key="target")
        x=df.drop(columns=target)
        y=df[target]

    
        typelist=['binary','multiclass','regression','time series regression','time series multiclass','time series binary']
        p_type=col2.selectbox("Select problem type",typelist,key="p_type")
        

        x_train,x_test,y_train,y_test=evalml.preprocessing.split_data(x,y,problem_type=p_type)

    #its training time
# =============================================================================
#             st.spinner()
#             with st.spinner(text='In progress'):
#                 st.info("Wait while we are selecting a best algoritham for your problem..Hold your breath.")
#                 time.sleep(20)
# =============================================================================
        automl = AutoMLSearch(X_train=x_train, y_train=y_train, problem_type=p_type)
        automl.search()


        rank=automl.rankings
# to see the rankings
# =============================================================================
# if st.button("Check rankings",key="rank"):
#     st.write(rank)
# =============================================================================

#checking best pipeline     ###############################################################

        best_pipeline=automl.best_pipeline
        
        
        description=automl.describe_pipeline(automl.rankings.iloc[0]["id"],return_dict=True)
        
        file=open("pipeline_details.txt","w")
        str_dict=repr(description)
        file.write(str_dict)
        file.close()
        def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Click Here To Download {file_label}</a>'
                return href

        


### Evaluate on hold out data
        problem_list=['binary','time series binary']
        problem_list2=['multiclass','time series multiclass']
            
        if p_type in problem_list:
            best_pipeline.score(x_test, y_test, objectives=["auc","f1","Precision","Recall"])

            automl_tunned = AutoMLSearch(X_train=x_train, y_train=y_train,
                                         problem_type=p_type,
                                         objective='auc',
                                         additional_objectives=['f1', 'precision'],
                                         max_batches=1,
                                         optimize_thresholds=True)

            automl_tunned.search()

            tunned_rankings=automl_tunned.rankings

            automl_tunned.describe_pipeline(automl_tunned.rankings.iloc[0]["id"])

            tunned_pipeline= automl_tunned.best_pipeline

            tunned_pipeline.score(x_test, y_test,  objectives=["auc"])

            pred=tunned_pipeline.predict_proba(x_test).to_dataframe()


# for multiclass type problem
        elif p_type in problem_list2:
                best_pipeline.score(x_test, y_test, objectives=["log loss multiclass","MCC multiclass","accuracy multiclass"])

                automl_tunned = AutoMLSearch(X_train=x_train, y_train=y_train,
                                         problem_type=p_type,
                                         objective="log loss multiclass",
                                         additional_objectives=['MCC multiclass', 'accuracy multiclass'],
                                         max_batches=1,
                                         optimize_thresholds=True)

                automl_tunned.search()

                tunned_rankings=automl_tunned.rankings

                automl_tunned.describe_pipeline(automl_tunned.rankings.iloc[0]["id"])

                tunned_pipeline= automl_tunned.best_pipeline

                tunned_pipeline.score(x_test, y_test,  objectives=["log loss multiclass"])

                pred=tunned_pipeline.predict(x_test).to_series()

    
# for regression type problems
        else:
                best_pipeline.score(x_test, y_test, objectives=["r2","MSE","MAE","Root Mean Squared Error"])
                automl_tunned = AutoMLSearch(X_train=x_train, y_train=y_train,
                                         problem_type=p_type,
                                         objective='r2',
                                         additional_objectives=['Root Mean Squared Error', 'MSE','MAE'],
                                         max_batches=1,
                                         optimize_thresholds=True)

                automl_tunned.search()

                tunned_rankings=automl_tunned.rankings

                automl_tunned.describe_pipeline(automl_tunned.rankings.iloc[0]["id"])

                tunned_pipeline= automl_tunned.best_pipeline

                tunned_pipeline.score(x_test, y_test,  objectives=["r2"])

                tunned_pipeline.fit(x_train,y_train)
                    
                pred=tunned_pipeline.predict(x_test).to_series()
                
        col1,col2,col3=st.beta_columns([1,1,1])

        
        if col2.button("Predict Results",key="output",help="shows results"):
            st.spinner()
            with st.spinner(text='In progress'):
                 st.info("Wait while we are selecting a best algoritham for your problem..Hold your breath.")
                 time.sleep(20)
            st.info("Done. Here you go.")
            st.write(pred)

            col11,col12=st.beta_columns([3,1])
            with col11:
                with st.beta_expander("Compare Models"):
                    st.write(rank)
            with col12:
                with st.beta_expander("Pipeline Details"):
                    st.success(best_pipeline)
                    st.markdown(get_binary_file_downloader_html('pipeline_details.txt', 'Pipeline Details'), unsafe_allow_html=True)
# =============================================================================
#         with col13:
#             with st.beta_expander("Model Deatils"):
#                 st.success(description)                
# =============================================================================
        
        
 
# =============================================================================
def main():
    search_option=st.sidebar.selectbox("Data Input Method",["Local File","Webpage"])
    if search_option=="Webpage":
        url_data()
                  
    else:
        file_data()
        about()                
                
            
        
if __name__=="__main__":

    main()


