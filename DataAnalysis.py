import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


file_name= 'kc_house_data_NaN.csv'
Date_Time = 'date'

st.write("""
	# Price Prediction 
	""")
st.write("""
	# Machine Learning Web App
	""")

st.write ("""
	This application predicts **House Sales price in King Country, USA**.

	Dataset contains house sale prices for King County, which includes Seattle. 
	It includes homes sold between May 2014 and May 2015.
	"""
)

@st.cache(allow_output_mutation=True)
def load_data(nrows):
	df=pd.read_csv(file_name, nrows=nrows)
	df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)
	lowercase = lambda x: str(x).lower()
	df.rename(lowercase, axis='columns', inplace = True)
	df[Date_Time] = pd.to_datetime(df[Date_Time])
	df.rename(columns = {"long" : "lon"}, inplace = True)
	mean=df['bedrooms'].mean()
	df['bedrooms'].replace(np.nan,mean, inplace=True)
	mean=df['bathrooms'].mean()
	df['bathrooms'].replace(np.nan,mean, inplace=True)
	return df


data = load_data(100000)
data_frame = data


st.sidebar.header('User parameters')
select = st.sidebar.selectbox("Choose Machine Learning Algorithm", ['Home','Linear Regression', 'KNeighborsClassifier'])
y = data_frame[('price')]
if select == 'Home':
	if( st.checkbox('Show Raw Data', False)):
		st.subheader('Raw Data')
		st.write(data)

	st.write('Locations of Houses sold between May 2014 and May 2015.')
	data_map = data_frame[["lat", "lon"]]
	st.map(data_map)

	st.header("Correlation with price")
	choose = st.selectbox('Choose one whether a feature is negatively or positively correlated with price.', [ 'Features','sqft_above', 'sqft_basement', 'waterfront', 'grade', 'floors'])
	if choose == ('sqft_above'):
		sns.regplot(x = "sqft_above", y = "price", data = data)
		st.pyplot()
	elif choose == ('sqft_basement'):
		sns.regplot(x = "sqft_basement", y = "price", data = data)
		st.pyplot()
	elif choose == ('waterfront'):
		sns.regplot(x = "waterfront", y = "price", data = data)
		st.pyplot()
	elif choose == ('grade'):
		sns.regplot(x = "grade", y = "price", data = data)
		st.pyplot()
	elif choose == ('floors'):
		sns.regplot(x = "floors", y = "price", data = data)
		st.pyplot()

	image = Image.open('data.jpeg')
	st.image(image, caption='Coded with Python ', use_column_width=True)


elif select == 'Linear Regression':
	random_state_slider = st.sidebar.slider('Random State for train/test samples', 1, 5)
	test_slider = st.sidebar.slider('Test Size ( 0.1 = 10 %  of test samples)', 0.1, 0.7)
	st.header('Predicting price value of a house with features')
	st.subheader('Best possible score is 1.0 that means a feature is more correlated with the price.')
	ft = st.selectbox('Choose a feature',['Features', 'sqft_living' ,'floors', 'waterfront','lat','bedrooms','sqft_basement','view' ,'bathrooms','sqft_living15','sqft_above','grade'])
	if ft == 'sqft_living':
		Z = data_frame[['sqft_living']]
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:',lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'floors':
		Z = data_frame[['floors']]
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'waterfront':
		Z = data_frame[['waterfront']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])
		
	elif ft == 'lat':
		Z = data_frame[['lat']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'bedrooms':
		Z = data_frame[['bedrooms']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'sqft_basement':
		Z = data_frame[['sqft_basement']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'view':
		Z = data_frame[['view']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'bathrooms':
		Z = data_frame[['bathrooms']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'sqft_living15':
		Z = data_frame[['sqft_living15']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'sqft_above':
		Z = data_frame[['sqft_above']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'grade':
		Z = data_frame[['grade']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		lm = LinearRegression()
		lm.fit(x_train, y_train)
		st.write('Training score:', lm.score(x_train, y_train))
		st.write('')
		st.write('Prediction')
		st.dataframe(lm.predict(x_train), 600, 200)
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

elif select == 'KNeighborsClassifier':
	random_state_slider = st.sidebar.slider('Random State for train/test samples', 1, 5)
	test_slider = st.sidebar.slider('Test Size ( 0.1 = 10 %  of test samples)', 0.1, 0.7)
	st.header('Predicting price value of a house with features')
	st.subheader('Best possible score is 1.0 that means a feature is more correlated with the price.')
	ft = st.selectbox('Choose a feature',['Features', 'sqft_living' ,'floors', 'waterfront','lat','bedrooms','sqft_basement','view' ,'bathrooms','sqft_living15','sqft_above','grade'])
	if ft == 'sqft_living':
		Z = data_frame[['sqft_living']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M*10)
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'floors':
		Z = data_frame[['floors']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'waterfront':
		Z = data_frame[['waterfront']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'lat':
		Z = data_frame[['lat']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'bedrooms':
		Z = data_frame[['bedrooms']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'bathrooms':
		Z = data_frame[['bathrooms']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'sqft_above':
		Z = data_frame[['sqft_above']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'sqft_living15':
		Z = data_frame[['sqft_living15']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'grade':
		Z = data_frame[['grade']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'sqft_basement':
		Z = data_frame[['sqft_basement']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])

	elif ft == 'view':
		Z = data_frame[['view']]
		
		
		x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size= test_slider, random_state=random_state_slider)
		
		st.write('Prediction with Training samples')
		k = st.slider('K value', 1, 10)
		knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
		T_metrics = metrics.accuracy_score(y_train, knn_model.predict(x_train))
		st.write('Training Set Accuracy of Model: ', T_metrics*1000, "%")

		M = f1_score(y_train, knn_model.predict(x_train), average='weighted')
		st.write('Weighted F1 score:', M * 10 )
		
		st.write('Number of test samples:',x_test.shape[0])
		st.write('Number of training samples:',x_train.shape[0])



st.balloons()
