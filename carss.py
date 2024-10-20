import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor  # Thay đổi ở đây
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu và xử lý chỉ thực hiện một lần
@st.cache_resource
def load_data():
    cars = pd.read_csv('CarPrice_Assignment.csv')

    # Kiểm tra các giá trị bị thiếu
    missing_values = cars.isnull().sum()
    if missing_values.sum() > 0:
        cars.fillna(cars.mean(), inplace=True)

    # Xóa các cột không cần thiết
    
        cars.drop(['CarName', 'car_ID'], axis=1, inplace=True)

    # Thêm cột 'fueleconomy'
    cars['fueleconomy'] = (cars['citympg'] + cars['highwaympg']) / 2

    # Lọc các đặc trưng cần thiết
    features = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower', 'boreratio', 'fueleconomy']
    cars = cars[features + ['price']]

    # Chuẩn hóa các đặc trưng số học
    
    scaler = MinMaxScaler()
    cars[features] = scaler.fit_transform(cars[features])
    return cars, features

# Tải dữ liệu và mô hình
@st.cache_resource
def load_data_and_models():
    cars, features = load_data()

    # Tách đặc trưng (X) và nhãn (y)
    
    X = cars[features]
    y = cars['price']

    # Chia dữ liệu
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # Chuẩn hóa dữ liệu
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện các mô hình
    lin_reg = LinearRegression().fit(X_train_scaled, y_train)
    ridge_reg = Ridge(alpha=5.0).fit(X_train_scaled, y_train)
    nn_reg = MLPRegressor(hidden_layer_sizes=(150, 75), max_iter=5000, learning_rate_init=0.01,
                          random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10).fit(X_train_scaled, y_train)
  
    stacking_reg = StackingRegressor(
        estimators=[('lr', lin_reg), ('ridge', ridge_reg), ('nn', nn_reg)],
        final_estimator=DecisionTreeRegressor()
    ).fit(X_train_scaled, y_train)
    
    return cars, scaler, lin_reg, ridge_reg, nn_reg, stacking_reg, features

# Chỉ tải và khởi tạo mô hình một lần
cars, scaler, lin_reg, ridge_reg, nn_reg, stacking_reg, features = load_data_and_models()

# Giao diện người dùng Streamlit
st.title("Car Price Prediction App")

# Nút chọn mô hình
model_options = {
    'Linear Regression': lin_reg,
    'Ridge Regression': ridge_reg,
    'Neural Network': nn_reg,
    'Stacking Model': stacking_reg
}
selected_model_name = st.selectbox("Chọn mô hình:", list(model_options.keys()))
selected_model = model_options[selected_model_name]

# Nhập thông tin xe để dự đoán
st.write("Nhập thông tin xe để dự đoán giá:")
wheelbase = st.number_input('Wheelbase', value=98.4)
carlength = st.number_input('Car Length', value=168.8)
carwidth = st.number_input('Car Width', value=64.1)
curbweight = st.number_input('Curb Weight', value=2548)
enginesize = st.number_input('Engine Size', value=130)
horsepower = st.number_input('Horsepower', value=111)
boreratio = st.number_input('Bore Ratio', value=3.31)
fueleconomy = st.number_input('Fuel Economy (Average of city and highway MPG)', value=24)

new_data = {
    'wheelbase': wheelbase,
    'carlength': carlength,
    'carwidth': carwidth,
    'curbweight': curbweight,
    'enginesize': enginesize,
    'horsepower': horsepower,
    'boreratio': boreratio,
    'fueleconomy': fueleconomy
}

if st.button("Dự đoán giá"):
    input_data = pd.DataFrame([new_data])
    input_data_scaled = scaler.transform(input_data)
    predicted_price = selected_model.predict(input_data_scaled)[0]
    st.write(f"Giá dự đoán: ${predicted_price:,.2f}")

