import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
#function for getting objects and hot one encoding
def hot_one_encoding(df):
    #first we need to use remove column function
    df=remove_column(df)
    df_objects=df.select_dtypes(include='object')
    df_encoded=pd.get_dummies(df_objects,drop_first=True,dtype=int)
    return df_encoded

#function for getting the integer and float columns some dataframe
def get_ints(df):
    #first we need to use remove column function
    df=remove_column(df)
    df_ints=df.select_dtypes(include=['int64','float64'])
    return df_ints

#function for removing id and sale price if exists
def remove_column(df):
    #remove id if exists and remove sale if exists
    id_column='Id'
    sale_column='SalePrice'
    if id_column in df.columns:
        df=df.drop(columns=[id_column])
    if sale_column in df.columns:
        df=df.drop(columns=[sale_column])
    return df

#function for normalizing integers
def z_score_normalization(df):
    #define scalar
    scaler=StandardScaler()
    df_scaled=scaler.fit_transform(df)
    #convert back to dataframe
    return pd.DataFrame(df_scaled,columns=df.columns)

#function for min-max scaling
def min_max(df):
    scaler=MinMaxScaler()
    df_scaled=scaler.fit_transform(df)
    return pd.DataFrame(df_scaled,columns=df.columns)

#function for concatanation
def concat(df_encoded,df_z_score):
    combined=pd.concat([df_encoded,df_z_score],axis=1)
    return combined

#function for finding missing values
def align_columns(train_set,test_set):
    #find the missing columns
    missing_columns=set(train_set)-set(test_set)
    for column in missing_columns:
        test_set[column]=0
    test_set=test_set[train_set.columns]
    return test_set

#function for debugging
def debug(train_set,test_set):
    print(train_set.info())
    print(test_set.info())

#function to get all object columns
def get_object_columns(df):
    df=df.select_dtypes(include='object')
    return df

#function for saving both to csv
def debug_view(train_set,test_set):
    train_set.to_csv('train_set.csv',index=False)
    test_set.to_csv('test_set.csv',index=False)

#read csv's for test and train
df=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

#now we should probably hot one encode
df_encoded=hot_one_encoding(df)
df_test_encoded=hot_one_encoding(df_test)

#min max them and get ints first
df_minmax=min_max(get_ints(df))
df_test_minmax=min_max(get_ints(df_test))

#okay now we concat
train_set=concat(df_encoded,df_minmax)
test_set=concat(df_test_encoded,df_test_minmax)

#now we have to align
test_set=align_columns(train_set,test_set)

#get sale price column
df_y=df['SalePrice']
if test_set.shape[1] == train_set.shape[1]:
    #split data
    x_train, x_test, y_train, y_test =train_test_split(train_set,df_y,test_size=0.2,random_state=42)
    #declare model
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    #fit model
    model.fit(x_train, y_train)
    #predict
    predicitons=model.predict(x_test)
    mse=mean_squared_error(y_test,predicitons)
    print(mse)
    #make predictions
    predictions_2=model.predict(test_set)
    #create a dataframe for predictions
    predict=pd.DataFrame(predictions_2,columns=['SalePrice'])
    #create id column starting at 1461
    predict['Id']=range(1461,1461+len(predict))
    #reorder
    predict=predict[['Id','SalePrice']]
    #salve to csv
    predict.to_csv('predict.csv',index=False)


else:
    debug(train_set,test_set)