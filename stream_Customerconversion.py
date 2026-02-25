import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd


with st.sidebar :
    selected=option_menu("ML Tasks",["Classification","Regression","Clustering"],
                         icons=["bar-chart","graph-up","diagram-3"],
                         menu_icon="cast",default_index=0)
# classification module

if selected=="Classification":
    st.title("Purchase Prediction App ")
    st.markdown("Predict whether a user will make a purchase based on page and product features.")

    #Load the saved classification pipeline

    try:
        with open ("/Users/saranya/Documents/Projects/customer_conversion_project/xgboost_classifier_pipeline.pkl",'rb') as file:
            classification_model=pickle.load(file)
            
    except FileNotFoundError:
        st.error("Model file not found .Please make sure 'xgboost_classifier_pipeline.pkl is in app adirectory")

    #User inputs

    st.subheader("Enter Feature values")

    country=st.selectbox("country",['India','USA','Uk','Germany','France'])
    page1_main_category=st.selectbox("Main Category",['Men','Women','Kids','Accessories'])
    page2_clothing_model=st.selectbox("clothing Model",['T-Shirt','Jeans','Dress','Shoes'])
    colour=st.selectbox("Colour",['Red','Blue','Green','Black','White'])
    location=st.selectbox("Location",['Homepage','Product Page','Checkout'])
    model_photography=st.selectbox("Model Photography",['Studio','Outdoor','None'])
    
    
    order=st.number_input("Order Count",min_value=0,value=1)
    revenue=st.number_input("Revenue",min_value=0.0,value=100.0)
    page=st.number_input("Page Number",min_value=1,value=1)

    year=st.selectbox("Year",[2022,2023,2024,2025])
    month=st.selectbox("Month",list(range(1,13)))
    day=st.selectbox("Day",list(range(1,32)))

    #Predict button 
    if st.button("Predict Purchase"):
        input_df=pd.DataFrame([{
            'year':year,
            'month':month,
            'day':day,
            'order':order,
            'country':country,
            'page1_main_category':page1_main_category,
            'page2_clothing_model':page2_clothing_model,
            'colour':colour,
            'location': location,
            'model_photography': model_photography,
            'page':page,
            'revenue':revenue
        }])

        
            
        prediction = classification_model.predict(input_df)
        result="Will Purchase" if prediction[0]== 1 else "Will Not Purchase"
        st.success(f"prediction :{result}")
    # Regression module

elif selected=="Regression":
        st.title("Revenue Prediction App")
        st.markdown("Estimate expected revenue for a user based on their behavior and attributes.")

        try:
            with open('/Users/saranya/Documents/Projects/customer_conversion_project/revenue_regression_pipeline.pkl','rb') as file:
                reg_model=pickle.load(file)
        except FileNotFoundError:
            st.error("Regression model file not fond")
        else:
            st.subheader("Enter feature values")

            #Categorical inputs
            country=st.selectbox("country",['India','USA','Uk','Germany','France'])
            page1_main_category=st.selectbox("Main Category",['Men','Women','Kids','Accessories'])
            page2_clothing_model=st.selectbox("clothing Model",[ 'T-Shirt','Jeans','Dress','Shoes'])
            colour=st.selectbox("Colour",['Red','Blue','Green','Black','White'])
            location=st.selectbox("Location",['Homepage','Product Page','Checkout'])
            model_photography=st.selectbox("Model Photography",['Studio','Outdoor','None'])
    
    
            order=st.number_input("Order Count",min_value=0,value=1)
            revenue=st.number_input("Revenue",min_value=0.0,value=100.0)
            page=st.number_input("Page Number",min_value=1,value=1)

            year=st.selectbox("Year",[2022,2023,2024,2025])
            month=st.selectbox("Month",list(range(1,13)))
            day=st.selectbox("Day",list(range(1,32)))
            
            if st.button("Predict Revenue"):
                import pandas as pd 

                input_df=pd.DataFrame([{
               'year':year,
                'month':month,
                'day':day,
                'order':order,
                'country':country,
                'page1_main_category':page1_main_category,
                'page2_clothing_model':page2_clothing_model,
                'colour':colour,
                'location': location,
                'model_photography': model_photography,
                'page':page
            
             }])

                try:
                    prediction_rev = reg_model.predict(input_df)[0]
                    st.success(f" predicted revenue: {prediction_rev:.2f}")
                except Exception as e:
                    st.error(f"prediction failed: {e}")
                    st.write("Input Dataframe:",input_df)
    # clustering 
elif selected == "Clustering":
        st.title("User Segmentation app")
        st.markdown("segment users based on browsing and product interaction feature")


        try:
            with open ('/Users/saranya/Documents/Projects/customer_conversion_project/kmeans_model.pkl','rb') as file:
                cluster_model=pickle.load(file)

            with open('/Users/saranya/Documents/Projects/customer_conversion_project/scaler2.pkl','rb') as f:
                scaler=pickle.load(f)
        except FileNotFoundError:
            st.error("clustering model or scaler not found")
        else:
            st.subheader("enter user behaviour features")

            # input from clustering features
            country=st.selectbox("country",['India','USA','Uk','Germany','France'])
            page1_main_category=st.selectbox("Main Category",['Men','Women','Kids','Accessories'])
            page2_clothing_model=st.selectbox("clothing Model",[ 'T-Shirt','Jeans','Dress','Shoes'])
            colour=st.selectbox("Colour",['Red','Blue','Green','Black','White'])
            location=st.selectbox("Location",['Homepage','Product Page','Checkout'])
            model_photography=st.selectbox("Model Photography",['Studio','Outdoor','None'])
    
    
            order=st.number_input("Order Count",min_value=0,value=1)
            page=st.number_input("Page Number",min_value=1,value=1)
            # Mapping categorical values to match training
            country_map = {'India': 0, 'USA': 1, 'UK': 2, 'Germany': 3, 'France': 4}
            colour_map = {'Red': 0, 'Blue': 1, 'Green': 2, 'Black': 3, 'White': 4}
            location_map = {'Homepage': 0, 'Product Page': 1, 'Checkout': 2}
            model_photography_map = {'Studio': 0, 'Outdoor': 1, 'None': 2}
            main_category_map = {'Men': 0, 'Women': 1, 'Kids': 2, 'Accessories': 3}
            clothing_model_map = {'T-Shirt': 0, 'Jeans': 1, 'Dress': 2, 'Shoes': 3}
            
            # create input dataframe

            input_df=pd.DataFrame([{
               'order':order,
                'country':country_map[country],
                'page1_main_category':main_category_map[page1_main_category],
                'page2_clothing_model':clothing_model_map[page2_clothing_model],
                'colour':colour_map[colour],
                'location': location_map[location],
                'model_photography': model_photography_map[model_photography],
                'page':page
            
             }])
            
            input_df=input_df[['order', 'country', 
       'page1_main_category','colour', 'location', 'model_photography',
        'page', 'page2_clothing_model']]
            if st.button("Find cluster"):
                try:
                    #scaler input 
                    scaled_input=scaler.transform(input_df)

                    # predict cluster label
                    cluster_label = cluster_model.predict(scaled_input)[0]
                    
                    if cluster_label==-1:
                        st.warning(f" This user consider as noise (not part of any cluster)")
                    else:
                        st.success(f"User belongs to cluster:{cluster_label}")
                except Exception as e:
                    st.error("clustering failed due to error")
                    st.write(f"error details :{e}")
                    st.write("Input dataframe :",input_df)
    
    
    
     




