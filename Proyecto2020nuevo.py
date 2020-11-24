# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import datetime
#from bokeh.plotting import Histogram, output_file, show
import altair as alt
import streamlit as st
import os 
from PIL import Image
import pandas as pd
import numpy as np
#import cv2
from vega_datasets import data
import plotly.graph_objects as go
import plotly.express as px
import json
import joypy
import matplotlib.cm as cm
import pickle
from joblib import dump, load
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import time
from fpdf import FPDF
import base64
import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText

def main():
    
    df = load_data()
    df_predict = load_data_predict()
    categorical_feature_mask = df_predict.dtypes==object
    categorical_cols = df_predict.columns[categorical_feature_mask].tolist()
    #st.info("Ha seleccionado {}".format(archivo_csv))
    #df = st.cache(pd.read_csv)(archivo_csv,encoding='iso-8859-1')
    #st.text("Total Registros:")
    #st.write(df.shape[0])
    logo = Image.open('imagenes/logo.png')
    #logo = logo.resize((100, 100))
    st.sidebar.image(logo,use_column_width=True)
    
    page = st.sidebar.selectbox("Elija una Opción", ["Inicio", "Exploración","Serie de Tiempo","GeoVisualización","Predicción"])
    
    
    if page == "Inicio":
        #st.header("This is your data explorer.")
        #st.icon('face')
        video_file = open('imagenes/formulatam2020.mp4', 'rb')
        video_bytes = video_file.read()
        html_tmp="""
        <div style="background-color:#035105;"><p style="font-size:40px;color:white;text-align:center">Formulatam Z</p></div>"""
        st.markdown(html_tmp,unsafe_allow_html=True)
        #gif('https://giphy.com/embed/cj85ivIMTwqsULwBt2')
        st.sidebar.markdown('Equipo')
        st.sidebar.image('imagenes/TE.png',use_column_width=True)
        st.sidebar.markdown('Tania Leal	- tania.estay.leal@gmail.com')
        st.sidebar.image('imagenes/Cv.jpg',use_column_width=True)
        st.sidebar.markdown('Camila Vera	- cam.veravilla@gmail.com')
        st.sidebar.image('imagenes/CC.png',use_column_width=True)
        st.sidebar.markdown('Cesar Hernández	cesar.f50@gmail.com')
        st.sidebar.image('imagenes/PV.jpeg',use_column_width=True)
        st.sidebar.markdown('Pablo Veloz	pvelozm@gmail.com')
        st.sidebar.image('imagenes/HR.png',use_column_width=True)
        st.sidebar.markdown('Héctor Rojas	hector.rojas.p@usach.cl')
        
        st.video(video_bytes)
        
    elif page == "Exploración":
        html_tmp="""
        <div style="background-color:#035105;"><p style="font-size:40px;color:white;text-align:center">Exploración</p></div>
        """
        st.markdown(html_tmp,unsafe_allow_html=True)
        
        
        number =st.number_input("Mostrar número de filas",value=15)
        st.dataframe(df.head(number))
        if st.checkbox("Sumario Total"):
          st.write(df.describe())
        st.write("Columnas: ")
        st.table(df_predict.columns)
        html_tmp="""
        <div style="background-color:powderblue;"><p style="font-size:20px;color:black;text-align:center">Exploración Univariada</p></div>
        """
        st.markdown(html_tmp,unsafe_allow_html=True) 
        
        #for i in df.columns:
          #st.write(df[i].dtype)
        col= st.selectbox("Selecciona columna",["variable"]+df_predict.columns.tolist())
        if col != "variable":
          if col == "Price":
            g=quantitative_price_data(df,"Price")
            st.pyplot()
          elif col== "Mileage":
            g=quantitative_mileage_data(df,"Price")
            st.pyplot()
            #g2=mileage_tooltip(df)
            #st.altair_chart(g2)
          elif col== "Year":
            g=cont_registros_year(df,"Year")
            st.altair_chart(g)
          elif col== "City":
            g=cont_registros_city(df,"City")
            st.altair_chart(g)
          elif col== "State":
            g=cont_registros_state(df,"State")
            st.altair_chart(g)
          elif col== "Make":
            make_count = Image.open('imagenes/squarify_marcas.png')
            st.image(make_count,use_column_width=True)
          elif col== "Model":
            g=cont_registros_model(df,"Model")
            st.altair_chart(g)
          else:
            if col!="DisplacementL":
              g=cont_registros(df_predict,col)
              st.altair_chart(g)
            else:
              q = df_predict["DisplacementL"].quantile(0.99)
              df_99=df_predict[["DisplacementL"]][df_predict["DisplacementL"] < q]
              g=cont_registros(df_99,col)
              st.altair_chart(g)
        html_tmp="""
        <div style="background-color:powderblue;"><p style="font-size:20px;color:black;text-align:center">Exploración respecto al Precio</p></div>
        """
        st.markdown(html_tmp,unsafe_allow_html=True) 
        col=st.radio("Selecciona columna",('Kilometraje', 'Ciudad', 'Estado','Marcas','Model'))
        if col == 'Kilometraje':
          g=dosd_mileage2(df,'Mileage')
          st.altair_chart(g)
          if st.checkbox("Filtro por Marca"):
            g2=mileage_tooltip(df)
            st.altair_chart(g2) 
        elif col == 'Ciudad':
          state=st.selectbox("Selecciona Estado",df["State"].unique().tolist())
          g=dosd_city(df,state,'City')
          st.altair_chart(g)
          if st.checkbox("Desea indicar solo una ciudad"):
            solo_city=st.text_input("Ingrese Ciudad","City")
            if solo_city!="City":
              g=dosd_solo_city(df,solo_city)
            #df_solo_state=df[df["City"]==]
              st.altair_chart(g)
        elif col == 'Model':
          make= st.selectbox("Selecciona Marca",df["Make"].unique().tolist())
          g=dosd_model(df,make,col)
          st.altair_chart(g)
          if st.checkbox("Desea indicar solo un Modelo"):
            solo_model=st.text_input("Ingrese Model","Model")
            if solo_model!="Model":
              g=dosd_solo_model(df,solo_model)
            #df_solo_state=df[df["City"]==]
              st.altair_chart(g)      
        elif col == 'Estado':
          g = map_tooltip()
          st.altair_chart(g)
        elif col == "Marcas":
          dosd_marcas = Image.open('imagenes/dosd_marcas.png')
          #logo = logo.resize((100, 100))
          st.image(dosd_marcas,use_column_width=True)
          
        #multi= st.multiselect("Select variable",["variable"]+df.columns.tolist())
        #elif col== "Year":
        #  g=nominal_year(df,"Year")
        #  st.pyplot()
    elif page == "Serie de Tiempo":
        html_tmp="""
        <div style="background-color:#035105;"><p style="font-size:40px;color:white;text-align:center">Serie de Tiempo</p></div>"""
        st.markdown(html_tmp,unsafe_allow_html=True)  
        #st.markdown('Board __Formulatam Z__ Time-Series by Year')
        
        
        mark = st.selectbox('Serie de Tiempo Marca-Año',sorted(df['Make'].unique().tolist()))
        df_make_year_price=df[["Make","Price","Year"]]
        df_Make=df[df["Make"]==mark]
        df_make_year_price=df_Make.groupby(["Make","Year"])['Price'].mean().reset_index()
        c = alt.Chart(df_make_year_price).mark_line().encode(x='Year', y='Price')
        st.write(c)
        
        #df_Make=df[df["Make"]==mark]
        #model=st.selectbox('Time Serie Model in Year',sorted(df_Make['Model'].unique().tolist()))
        #marca = st.text_input("Consultar por Modelo","marca")
        if st.checkbox("Consultar por modelo"):
          #df_Make=df[df["Make"]==marca]
          model = st.selectbox('Time Serie Model in Year',sorted(df_Make["Model"].unique().tolist()))
          df_model=df_Make[df_Make["Model"]==model]
          df_model_year_price=df_model[["Model","Price","Year"]]
          df_model_year_price=df_model_year_price.groupby(["Model","Year"])['Price'].mean().reset_index()
          c = alt.Chart(df_model_year_price).mark_line().encode(x='Year', y='Price')
          st.write(c)
        
        if st.checkbox("Comparar Marcas"):
          #df_Make=df[df["Make"]==marca]
          marcas = st.multiselect('Seleccione Marcas',sorted(df['Make'].unique().tolist()))
          if len(marcas)!=0:
            df_makes=df[df["Make"].isin(marcas)]
            if st.button("Serie de tiempo por Marcas"):
            #df_model=df_Make[df_Make["Model"]==model]
            #df_model_year_price=df_model[["Model","Price","Year"]]
              df_makes=df_makes.groupby(["Make","Year"])['Price'].mean().reset_index()
            #c = alt.Chart(df_model_year_price).mark_line().encode(x='Year', y='Price')
              c=alt.Chart(df_makes).mark_line().encode(x='Year',y='Price',color='Make')
              st.write(c)
              #st.line_chart(df_makes)
        
        
    elif page == "GeoVisualización":
        states = alt.topo_feature(data.us_10m.url, 'states')
        html_tmp="""
        <div style="background-color:#035105;"><p style="font-size:40px;color:white;text-align:center">GeoVisualización</p></div>"""
        st.markdown(html_tmp,unsafe_allow_html=True)  
        #st.markdown('Board __Formulatam Z__ GeoVisualization by Year')   
        slider_ph = st.empty()
        value = slider_ph.slider("Año", 1997, 2018, 2000, 1)
    
        map_df2=pd.read_csv(r'datasets/geopricefull.csv',delimiter=",")
        map_df2['Price']=map_df2['Price'].astype(int)
        make_selectors = st.selectbox('Marca',["Marca"]+sorted(map_df2['Make'].unique().tolist()))
        if make_selectors == "Marca":
          query=map_df2[map_df2["Year"]==int(value)]
        else:
          query=map_df2[(map_df2["Make"]==make_selectors) & (map_df2["Year"]==int(value))]

        st.write(query)
        
        
        #states = alt.topo_feature(data.us_10m.url, 'states')
        #map_df=pd.read_csv(r'datasets/geoprice.csv',delimiter=",")
        mapa_df2=alt.Chart(states,title="Precio de Autos por Años y Estados").mark_geoshape().encode(
        color='Price:Q',tooltip=['State:N','Price:Q']).transform_lookup(lookup='id',
        from_=alt.LookupData(query, 'STATE_FIPS',['Price','State'])
        ).properties(width=700,height=500
        ).project('albersUsa').properties(width=700,height=700
        )
        mapa_df2.height=600
        #return mapa_df2+mapa_df2
        
        st.altair_chart((mapa_df2+mapa_df2))
        
    if page =="Predicción":
      html_tmp="""
      <div style="background-color:#035105;"><p style="font-size:40px;color:white;text-align:center">Predicción de Precios</p></div>"""
      st.markdown(html_tmp,unsafe_allow_html=True)  
      #st.markdown('Board __Formulatam Z__ Predict Price')
      var_predict = False
     
      #k=0
      #code=st.selectbox("Make",df["Make"].unique().tolist())
      grilla="""
      <hr style="border: 5px solid darkgreen;"></hr>"""
      st.sidebar.markdown(grilla,unsafe_allow_html=True)
      code2 = "Características de vehículo"
      var = """<p><h3>{code}</h3></p>""".format(code=code2)
      st.sidebar.markdown(var,unsafe_allow_html=True)
      pickle_in = open("datasets/label_ok3.pickle","rb")
      model = load('datasets/xgb_full_model4.joblib',"rb")
      df=df_predict
      #y = model.predict(df_test_2)
      d = pickle.load(pickle_in)
      for i in df.columns:
        if i!="Price":
          if i== "Year":
            year = st.sidebar.slider("Año", int(df.Year.min()), int(df.Year.max()), 2014, 1)
          if i=="Mileage":
            mileage = st.sidebar.number_input("Kilometraje")
          if i=="State":
            state = st.sidebar.selectbox("Estado",sorted(df[i].unique().tolist()))
          if i=="Make":
            make = st.sidebar.selectbox("Marca",sorted(df[i].unique().tolist()))
          if i=="AirBagLocSide":
            airbag = st.sidebar.selectbox("Airbags",df[i].unique().tolist())
          if i=="BodyClass":
            bodyClass = st.sidebar.selectbox("Clase",df[i].unique().tolist())
          if i=="DisplacementL":
            displacementL = st.sidebar.selectbox("CC",sorted(df[i].unique().tolist()))
          if i=="Doors":
            doors = st.sidebar.selectbox("Puertas",df[i].unique().tolist())
            doors = int(doors)
          if i=="DriveType":
            drivetype = st.sidebar.selectbox("Tracción",df[i].unique().tolist())
          if i=="EngineCylinders":
            engineCylinders = st.sidebar.selectbox("Cilindrada",df[i].unique().tolist())
            engineCylinders = int(engineCylinders) 
          if i=="FuelTypePrimary":
            fuel = st.sidebar.selectbox("Combustible",df[i].unique().tolist())
          #if i=="Manufacturer":
          #  manufacturer = st.sidebar.selectbox(i,df[i].unique().tolist())
          if i=="PlantCountry":
            plantCountry = st.sidebar.selectbox("Planta",df[i].unique().tolist())
          if i=="TPMS":
            tpms = st.sidebar.selectbox("TPMS",df[i].unique().tolist())
          if i=="VehicleType":
            vehicleType = st.sidebar.selectbox("Tipo de Vehículo",df[i].unique().tolist())
      boton="""<p style="border: border: 2px solid #4CAF50;">Predecir</p>"""
      df_test_predict = pd.DataFrame({'Year':[year],'Mileage':[mileage],'State':[state],'Make':[make],'AirBagLocSide':[airbag],'BodyClass':[bodyClass],'DisplacementL':[displacementL],'Doors':[doors],'DriveType':[drivetype],'EngineCylinders':[engineCylinders],'FuelTypePrimary':[fuel],'PlantCountry':[plantCountry],'TPMS':[tpms],'VehicleType':[vehicleType]})
      latest_iteration = st.empty()
      df_test_predict2=df_test_predict.copy()
      df_test_predict[categorical_cols] = df_test_predict[categorical_cols].apply(lambda x: d[x.name].transform(x))
      y = model.predict(df_test_predict)
      valor="${:,.2f}".format(y[0])
      v=int(y[0])
      ax=predict_model3(df,make,year,bodyClass,v)
      #st.altair_chart(g)
      #st.pyplot()
      ax.figure.savefig('chart.png')
      calcular_precio2=calcular_precio(v,0.903,0.11)
      df_precio = pd.DataFrame(np.array([["${:,.1f}".format(calcular_precio2[0]), "${:,.1f}".format(calcular_precio2[1]), "${:,.1f}".format(calcular_precio2[2]),"{:,.2f}".format(calcular_precio2[3],2)]]),columns=['Precio Compra', 'Precio Venta', 'Utilidad Moneda','Utilidad Final'])
      pdf2=pdf(df_test_predict2,valor,'chart.png',calcular_precio2[0],calcular_precio2[1],calcular_precio2[2],calcular_precio2[3])
      boton_predict = st.sidebar.button("Predicción")
      if boton_predict:
        bar = st.progress(0)
        for i in range(100):
          latest_iteration.text(f'Cargando... {i+1}%')
          val=bar.progress(i + 1)
          time.sleep(0.02)
        bar = st.empty()
        st.write(df_test_predict2)
        st.success(f'El precio del vehículo es: {valor}')
        
        st.table(df_precio)
        st.balloons()
        
      if st.sidebar.button("Resumen"):
        st.write(df_test_predict2)
        st.success(f'El precio del vehículo es: {valor}')
        st.table(df_precio)
)
      if st.sidebar.checkbox("Desea Enviar Reporte a e-mail"):
        mail=st.sidebar.text_input("Ingrese e-Mail")
        #enviar_mail(mail)
        if mail!="":
          msg=EmailMessage() 
          msg['Subject']='Reporte'
          msg['From']='formulatamz@gmail.com'
          msg['To']=mail
          msg.set_content('Se adjunta reporte de vehículo Marca: '+str(df_test_predict2.Make.iloc[0]))
          files = ['simple_pdf.pdf']
          for file in files:
            with open(file,'rb') as f:
              file_data = f.read()
              file_name = f.name
            msg.add_attachment(file_data,maintype='application', subtype='octet-stream',filename=file_name)
          with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
            smtp.login("email","password")
            smtp.send_message(msg)
            st.sidebar.success('Envío Exitoso')


@st.cache
def load_data():
    df=st.cache(pd.read_csv)('datasets/full_vin_limpio_ok2.csv',delimiter=",",encoding='iso-8859-1')
    df=df.drop('Unnamed: 0',axis=1)
    return df

  
def load_data_predict():
    #archivo_csv=file_selector()
    #st.info("Ha seleccionado {}".format(archivo_csv))
    df=st.cache(pd.read_csv)('datasets/full_vin_limpio_ok3.csv',delimiter=",",encoding='iso-8859-1')
    #df=pd.read_csv(r'datasets/true_car_listings.csv',delimiter=",")
    df=df.drop('Unnamed: 0',axis=1)
    return df  
  
  
def map_tooltip():
    states = alt.topo_feature(data.us_10m.url, 'states')
    map_df=pd.read_csv(r'datasets/geoprice.csv',delimiter=",")
    mapa_df2=alt.Chart(states,title="Price cars Mean per States").mark_geoshape().encode(
    color='Price:Q',tooltip=['STATE_NAME:N','Price:Q']
    ).transform_lookup(
    lookup='id',
    from_=alt.LookupData(map_df, 'STATE_FIPS',['Price','STATE_NAME'])
    ).properties(width=700,height=500
    ).project('albersUsa').properties(width=700,height=700
    )
    mapa_df2.height=600
    return mapa_df2+mapa_df2

def mileage_tooltip(df):
    df_make= df.groupby('Make').size().reset_index(name='counts')
    df_make_ord=df_make.sort_values(by='counts',ascending=False)
    #df_make_ord2["Make"].head(25).tolist()
    filter1=df["Make"].isin(df_make_ord["Make"].head(25).tolist())
    df_25marcas=df[filter1]
    df_25marcas=df_25marcas.sample(frac=0.05, replace=True, random_state=1)
    
    
    color = alt.Color('Make:N')
    # We create two selections:
    # - a brush that is active on the top panel
    # - a multi-click that is active on the bottom panel
    brush = alt.selection_interval(encodings=['x'])
    click = alt.selection_multi(encodings=['color'])

    # Top panel is scatter plot of temperature vs time
    points = alt.Chart().mark_circle().encode(
        alt.X('Mileage:Q', axis=alt.Axis(title='Mileage')),
        alt.Y('Price:Q',
            axis=alt.Axis(title='Price'),
        ),
        color=alt.condition(brush, color, alt.value('lightgray')),
        tooltip=['Make:N'],
    ).properties(
        width=600,
        height=300
    ).add_selection(
        brush
    ).transform_filter(
        click
    )
    
    bars = alt.Chart().mark_bar().encode(
    alt.Y('count()', scale=alt.Scale(type='log')),
    alt.X('Make:N'),
    color=alt.condition(click, color, alt.value('lightgray')),
    ).transform_filter(
        brush
    ).properties(
        width=600,
    ).add_selection(
        click
    )
    
    
    g3=alt.vconcat(points,bars,data=df_25marcas,title="Price vs Mileage")
    return g3
def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

    st.write(graph)
    
def comparison_by_bodyClass(df):
    df=df.sample(frac=0.1, replace=True, random_state=1)
    brush = alt.selection_interval()
    points = alt.Chart(df).mark_point().encode(
    x='Year:Q',
    y='Price:Q',
    color=alt.condition(brush, 'BodyClass:N', alt.value('lightgray'))
    ).add_selection(brush)
    bars = alt.Chart(df).mark_bar().encode(
        y='BodyClass:N',
        color='BodyClass:N',
        x='count(BodyClass):Q'
    ).transform_filter(brush)
    
    return points & bars
    
    
    
    
def quantitative_price_data(df,col):
  sns.set_style("whitegrid")
  kwargs = dict(hist_kws={'alpha':.8}, kde_kws={'linewidth':2})
  plt.figure(figsize=(15, 8))
  q = df["Price"].quantile(0.99)
  df_99=df[["Price"]][df["Price"] < q]
  axes=sns.distplot(df["Price"][df["Price"] < q],color="#959a9f",**kwargs,bins=58)
  axes.set_xlim([0,55000])
  axes.set_ylim([0,0.000065])
  plt.title("Distribución Price Cars",fontsize=18)
  plt.axvline(df_99["Price"].mean(),lw=5,color="#4f575f")
  mean_99=df_99["Price"].mean()
  plt.axvline(df_99["Price"].median(),lw=5,color="#4f575f")
  plt.text(df_99["Price"].median()+26, 0.00005, f"({df_99.Price.median()}) Mediana",
           bbox={'facecolor':'green', 'edgecolor': 'green', 'pad':5, 
                 'alpha': 0.4}, zorder=15)
  plt.text(mean_99+30, 0.00006, f"({round(mean_99,1)}) Media",
           bbox={'facecolor':"green", 'edgecolor': "green", 'pad':5, 
                 'alpha':  0.4}, zorder=12)
  plt.xlabel('')
  #plt.ylabel('Densidad')
  plt.xticks(rotation=30,fontsize=13)
  plt.yticks(rotation=30,fontsize=13)
  axes.get_yaxis().set_visible(True)
  return axes

def quantitative_mileage_data(df,col):
  sns.set_style("whitegrid")
  kwargs = dict(hist_kws={'alpha':.8}, kde_kws={'linewidth':2})
  plt.figure(figsize=(15, 8))
  q = df["Mileage"].quantile(0.99)
  df_99=df[["Mileage"]][df["Mileage"] < q]
  axes=sns.distplot(df["Mileage"][df["Mileage"] < q],color="#959a9f",**kwargs,bins=58)
  axes.set_xlim([0,175000])
  #axes.set_ylim([0,0.000065])
  plt.title("Distribución Mileage Cars",fontsize=18)
  plt.axvline(df_99["Mileage"].mean(),lw=5,color="#4f575f")
  mean_99=df_99["Mileage"].mean()
  plt.axvline(df_99["Mileage"].median(),lw=5,color="#4f575f")
  plt.text(df_99["Mileage"].median()+26, 0.000008, f"({df_99.Mileage.median()}) Mediana",
           bbox={'facecolor':'green', 'edgecolor': 'green', 'pad':5, 
                 'alpha': 0.4}, zorder=15)
  plt.text(mean_99+30, 0.000009, f"({round(mean_99,1)}) Media",
           bbox={'facecolor':"green", 'edgecolor': "green", 'pad':5, 
                 'alpha':  0.4}, zorder=12)
  plt.xlabel('')
  #plt.ylabel('Densidad')
  plt.xticks(rotation=30,fontsize=13)
  plt.yticks(rotation=30,fontsize=13)
  axes.get_yaxis().set_visible(True)
  return axes

def cont_registros_year(df,col):
  source = df[col].value_counts().reset_index()
  source.rename(columns = {'index':'Year','Year':'count'}, inplace = True)
  g=alt.Chart(source).mark_bar().encode(
    x='Year',
    y='count',
    color='Year'
  )
  return g

def cont_registros(df,col):
  source = df[col].value_counts().reset_index()
  source.rename(columns = {'index':col,col:'count'}, inplace = True)
  g=alt.Chart(source).mark_bar().encode(
    x=col,
    y='count',
    color=col
  ).properties(
      width=600,
      height=800
  )
  return g


def cont_registros_city(df,col):
  source = df[col].value_counts().reset_index().head(30)
  source.rename(columns = {'index':'City','City':'count'}, inplace = True)
  #source=source.sort_values(by=["count"],ascending=False).head(20)
  g=alt.Chart(source).mark_bar().encode(
   alt.X('City:N', sort=alt.EncodingSortField(field="City", op="count", order='ascending')),
   alt.Y('count:Q'))
  return g

def cont_registros_state(df,col):
  source = df[col].value_counts().reset_index().head(30)
  source.rename(columns = {'index':'State','State':'count'}, inplace = True)
  #source=source.sort_values(by=["count"],ascending=False).head(20)
  g=alt.Chart(source).mark_bar().encode(
   alt.X('State:N',sort=alt.EncodingSortField(field="State", op="count", order='ascending')),
   alt.Y('count:Q'),color='State')
  return g

def cont_registros_model(df,col):
  source = df[col].value_counts().reset_index().head(30)
  source.rename(columns = {'index':'Model','Model':'count'}, inplace = True)
  #source=source.sort_values(by=["count"],ascending=False).head(20)
  g=alt.Chart(source).mark_bar().encode(
   alt.X('Model:N', sort=alt.EncodingSortField(field="Model", op="count", order='ascending')),
   alt.Y('count:Q'))
  return g

def nominal_year(df,col):
  plt.figure(figsize=(25,15), dpi= 80)
  fig, axes = joypy.joyplot(df, column=['Price'], by=col, ylim='own',x_range=[-10000,70000], figsize=(14,10),colormap=cm.Blues_r)
  return axes

def dosd_mileage(df,col):
  c = alt.Chart(df).mark_circle().encode(
    x='Mileage', y='Price',color='red',opacity=0.3)
  return c
  #st.altair_chart(c, use_container_width=True)
def dosd_mileage2(df,col):
  df=df.sample(frac=0.008, replace=True, random_state=1)
  g=alt.Chart(df).mark_circle(
    color='red',
    opacity=0.3
  ).encode(
    x='Mileage:Q',
    y='Price:Q'
  )
  return g

def dosd_city(df,state,col):
  df_state=df[df['State']==state]
  df_state=df_state.sample(frac=0.1, replace=True, random_state=1)
  g=alt.Chart(df_state).mark_boxplot().encode(
    x='City:O',
    y='Price:Q',
    color='City'
  )
  return g

def dosd_solo_city(df,city):
  df_city=df[df['City']==city]
  #df_state=df_state.sample(frac=0.1, replace=True, random_state=1)
  g=alt.Chart(df_city).mark_boxplot().encode(
    y='Price:Q'
  ).properties(
      width=200,
      height=300
  )
  return g

def dosd_model(df,make,col):
  df_make=df[df['Make']==make]
  df_make=df_make.sample(frac=0.1, replace=True, random_state=1)
  g=alt.Chart(df_make).mark_boxplot().encode(
    x='Model:O',
    y='Price:Q',
    color='Model'
  )
  return g

def predict_model(df,make,year,clase):
  df_make=df[df['Make']==make]
  df_year=df_make[df_make['Year']==year]
  df_clase=df_year[df_year['BodyClass']==clase]
  #df_clase=df_clase.sample(frac=0.5, replace=True, random_state=1)
  g=alt.Chart(df_clase).mark_circle(
    color='red',
    opacity=0.3
  ).encode(
    x='Year:O',
    y='Price:Q'
  )
  return g

def predict_model2(df,make,year,clase,price):
  df_make=df[df['Make']==make]
  #df_year=df_make[df_make['Year']==year]
  df_clase=df_make[df_make['BodyClass']==clase]
  g=alt.Chart(df_clase).mark_bar().encode(
      alt.X("Price:Q", bin=True),
      y='count()',
  )
  overlay = pd.DataFrame({'x': [price]})
  vline = alt.Chart(overlay).mark_rule(color='red', strokeWidth=3).encode(x='x:Q')
  return (g+vline)
def predict_model3(df,make,year,clase,price):
  df_make=df[df['Make']==make]
  #df_year=df_make[df_make['Year']==year]
  #df_clase=df_make[df_make['BodyClass']==clase]
  q = df_make["Price"].quantile(0.99)
  df_99=df_make[["Price"]][df_make["Price"] < q]
  plt.figure(figsize=(10,5))
  #g=plt.hist(df_clase["Price"], bins=30)
  g=sns.distplot(df_99["Price"],kde=False, color="b",axlabel="Precio",bins=30)
  g.axvline(x=price,linewidth=4, color='r')
  g.axvline(df_99["Price"].median(),lw=4,color="#4f575f")
  mediana = "${:,.0f}".format(int(df_99["Price"].median()))
  prediccion = "${:,.0f}".format(int(price))
  plt.legend(['Prediccion:'+str(prediccion),'Mediana:'+str(mediana)])
  plt.title("Histograma Precio")
  return g

def dosd_solo_model(df,model):
  df_model=df[df['Model']==model]
  #df_state=df_state.sample(frac=0.1, replace=True, random_state=1)
  g=alt.Chart(df_model).mark_boxplot().encode(
    y='Price:Q'
  ).properties(
      width=200,
      height=300
  )
  return g

def encode_categorical(df):
    le = LabelEncoder()
    categorical_feature_mask = df.dtypes==object
    categorical_cols = df.columns[categorical_feature_mask].tolist()

    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    
    return df
  
def file_selector(pdf,carpeta):
  archivos_csv=os.listdir(carpeta)
  archivo_seleccionado=pdf
  return os.chdir(carpeta,archivo_seleccionado)
    
  
  
def pdf(df,valor,imagen,precio_compra,precio_venta,utilidad_moneda,utilidad_final):

  precio_compra="${:,.1f}".format(precio_compra)
  precio_venta="${:,.1f}".format(precio_venta)
  utilidad_moneda="${:,.1f}".format(utilidad_moneda)
  utilidad_final="{:,.2f}".format(utilidad_final)
  timbre='imagenes/timbre.png'
  pdf = FPDF() 
  pdf.add_page()
  pdf.set_font("Arial", size = 15) 
  pdf.cell(200, 10, txt = "FORMULATAM Z", ln = 1, align = 'C') 
  pdf.cell(200, 10, txt = "BEST TIME - BEST PRICE ", ln = 2, align = 'C')
  pdf.image('imagenes/logo.png',10, 8, 33)
  pdf.set_font('arial', '', 11)
  pdf.cell(280, 10, "Precio Compra: " '%s' % (str(precio_compra)), 0, 1, 'C')
  pdf.cell(276, 1, "Precio Venta: " '%s' % (str(precio_venta)), 0, 1, 'C')
  pdf.cell(279, 10, "Utilidad Moneda: " '%s' % (str(utilidad_moneda)), 0, 1, 'C')
  pdf.cell(266, 1, "Utilidad Final: " '%s' % (str(utilidad_final)), 0, 1, 'C')
  pdf.cell(15, 5, " ", 7, 4, 'C')
  pdf.cell(10)
  pdf.set_font('arial', '', 13)

  pdf.cell(-10)
  pdf.cell(150, 10, 'Características del vehículo', 1, 0, 'C')
  #pdf.cell(100, 10, 'Valores', 1, 1, 'C')
  #pdf.cell(40, 10, 'Mike', 1, 2, 'C')
  pdf.cell(90, 10, " ", 0, 2, 'C')
  pdf.cell(-150)
  pdf.set_font('arial', '', 11)
  for i in df.columns:
    pdf.cell(50, 10, '%s' % (i), 1, 0, 'C')
    pdf.cell(100, 10, '%s' % (str(df[i].iloc[0])), 1, 1, 'C')
    #pdf.cell(40, 10, '%s' % (str(df.Charles.iloc[i])), 1, 2, 'C')
    #pdf.cell(-90)
  pdf.cell(90, 2, " ", 0, 2, 'C')
  #pdf.cell(30)
  pdf.cell(150, 10, 'PRECIO PREDICT:'+str(valor), 1, 0, 'C')

  pdf.cell(-200)
  pdf.image(imagen, x = 0, y = 220, w = 98, h = 75, type = '', link = '')
 
  pdf.image(timbre, x = 115, y = 240, w = 35, h = 35, type = '', link = '')
 
  #'D','filename.pdf'
  #return pdf.output('simple_pdf.pdf','F')
  return pdf.output('simple_pdf.pdf','F')
  

def get_table_download_link(pdf):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    pdf=pdf
    #csv = df.to_csv(index=False)
    b64 = base64.b64encode(pdf.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/pdf;base64,{b64}">Download pdf file</a>'

def gif(srcs, width="100%", height=500):
    st.write(
        f'<iframe width="{width}" height="{height}" src="{srcs}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',unsafe_allow_html=True,)

    
def calcular_precio(precio_estimado,r2,margen_utilidad_base) :
    factor_error = 1 - r2
    
    #A cuanto debe comprar de acuerdo a margen utilidad
    precio_compra = precio_estimado*(1 - margen_utilidad_base)
    
    #Aplicar factor de error del modelo
    precio_compra = precio_compra-(precio_compra * factor_error)
    
    #Genera el precio venta cargando la desviación del modelo
    precio_venta = precio_estimado*(1 + factor_error)
    
    #Utilidad en monenda
    utilidad_moneda = precio_venta - precio_compra
    
    #Utilidad real si vende al valor propuesto
    utilidad_final = ((precio_venta/precio_estimado)- 1) + margen_utilidad_base
    
    return round(precio_compra,1), round(precio_venta,1), round(utilidad_moneda,1), round(utilidad_final,2)    
def enviar_mail(mail):
    msg=EmailMessage() 
    msg['Subject']='Reporte'
    msg['From']='pvelozm@gmail.com'
    msg['To']=mail
    msg.set_content('image attached...')
    files = ['simple_pdf.pdf']
    for file in files:
      with open(file,'rb') as f:
        file_data = f.read()
        file_name = f.name
      msg.add_attachment(file_data,maintype='application', subtype='octet-stream',filename=file_name)
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
      smtp.login("email","password")
      smtp.send_message(msg)
if __name__ == "__main__":
    main()
