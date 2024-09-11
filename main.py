import pandas as ps
import numpy as ny


housing = ps.read_csv('housing.csv')
# prices = housing['median_house_value']
# print(housing.head(2))

housing_df = housing.drop('ocean_proximity',axis=1)


# housing_df["income"] = ps.cut(housing_df["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., ny.inf],labels=[1, 2, 3, 4, 5]) # this code simply means that if the median income is between      0(included) to 1.5(excludec) then it will be labeles(or valued as 1) and similar to other cases.


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
# print(housing_df.isnull().sum())
housing_df[:] = imputer.fit_transform(housing_df)  # updates the dataframe

# print(housing_df.isnull().sum())


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(housing[['ocean_proximity']])
# print(encoded_data.toarray())
# print(encoder.categories_)
# housing_df[['<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN']] = encoded_data.toarray()
# print(housing_df)



from sklearn.model_selection import train_test_split
housing_prices = housing_df['median_house_value']
train_values , test_values = train_test_split(housing_prices , test_size=0.3 , random_state=42)
train_values , eval_values = train_test_split( train_values, test_size=0.2 , random_state=42)


housing_df.drop('median_house_value',inplace=True,axis=1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaler.fit_transform(housing_df)
housing_df = ps.DataFrame(scaler.fit_transform(housing_df),columns=housing_df.columns)

# housing_df = housing_normal
# print(housing_df)
housing_df[['<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN']] = encoded_data.toarray()
# print(housing_df.isnull().sum())


#spliting the data

train_df , test_df = train_test_split(housing_df , test_size=0.3 , random_state=42)
train_df , eval_df = train_test_split(train_df , test_size=0.2 , random_state=42)


def rmse(prdicted,actual):
    return ny.sqrt(ny.mean(ny.square(actual - prdicted)))



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_df,train_values)

predicted = model.predict(eval_df)
# print(predicted)
# print(eval_values)
print(rmse(predicted,eval_values))

from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_df, train_values)
predicted = tree_reg.predict(eval_df)
# print(predicted)
# print(eval_values)
print(rmse(predicted,eval_values))

from sklearn.ensemble import RandomForestRegressor
random_model = RandomForestRegressor()
random_model.fit(train_df, train_values)
predicted = random_model.predict(eval_df)
print(rmse(predicted,eval_values))