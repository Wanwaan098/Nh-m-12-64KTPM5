import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def train_models():
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

    # Huấn luyện các mô hình
    lin_reg = LinearRegression().fit(X_train_scaled, y_train)
    ridge_reg = Ridge(alpha=5.0).fit(X_train_scaled, y_train)
    nn_reg = MLPRegressor(hidden_layer_sizes=(150, 75), max_iter=5000, learning_rate_init=0.01,
                          random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10).fit(X_train_scaled, y_train)

    stacking_reg = StackingRegressor(
        estimators=[('lr', lin_reg), ('ridge', ridge_reg), ('nn', nn_reg)],
        final_estimator=DecisionTreeRegressor()
    ).fit(X_train_scaled, y_train)

    return lin_reg, ridge_reg, nn_reg, stacking_reg, X_train_scaled, y_train, X_val, y_val, X_test, y_test, scaler

# Gọi hàm để tải dữ liệu và huấn luyện mô hình
lin_reg, ridge_reg, nn_reg, stacking_reg, X_train_scaled, y_train, X_val, y_val, X_test, y_test, scaler = train_models()

# Phân tích các đặc trưng số học
cars, features = load_data()
print("Phân tích phân phối các đặc trưng số học:")
print(cars.describe())

# Biểu đồ phân phối (Histogram)
num_cols = 4
num_rows = (len(features) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for i, feature in enumerate(features):
    row, col = divmod(i, num_cols)
    sns.histplot(cars[feature], bins=20, kde=True, ax=axes[row, col])
    axes[row, col].set_title(f"Phân phối {feature}", fontsize=10)
    axes[row, col].tick_params(axis='x', labelsize=8)
    axes[row, col].tick_params(axis='y', labelsize=8)

for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes.flatten()[j])

plt.show()  # Hiển thị biểu đồ phân phối

# Biểu đồ hộp cho các đặc trưng
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for i, feature in enumerate(features):
    row, col = divmod(i, num_cols)
    sns.boxplot(x=cars[feature], ax=axes[row, col])
    axes[row, col].set_title(f"Biểu Đồ Hộp {feature}", fontsize=10)
    axes[row, col].tick_params(axis='x', labelsize=8)

for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes.flatten()[j])

plt.show()  # Hiển thị biểu đồ hộp

# Tính toán ma trận tương quan
correlation_matrix = cars.corr()

# Vẽ heatmap cho ma trận tương quan
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
ax.set_title("Ma Trận Tương Quan", fontsize=16)

plt.show()  # Hiển thị heatmap

# Tạo pairplot
pairplot_fig = sns.pairplot(cars[features])
pairplot_fig.fig.tight_layout()

plt.show()  # Hiển thị pairplot

def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    
    for model_name, model in models.items():
        # Dự đoán trên các tập dữ liệu
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Tính toán độ lỗi
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[model_name] = {
            'Train': {'RMSE': train_rmse, 'MAE': train_mae, 'R²': train_r2},
            'Validation': {'RMSE': val_rmse, 'MAE': val_mae, 'R²': val_r2},
            'Test': {'RMSE': test_rmse, 'MAE': test_mae, 'R²': test_r2}
        }
    
    return results

# Tạo dictionary chứa các mô hình
models = {
    'Linear Regression': lin_reg,
    'Ridge Regression': ridge_reg,
    'Neural Network': nn_reg,
    'Stacking Model': stacking_reg
}

# Chuẩn hóa dữ liệu cho validation và test bằng scaler đã học từ training set
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Đánh giá các mô hình
evaluation_results = evaluate_models(models, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)

# In kết quả
for model_name, metrics in evaluation_results.items():
    print(f"Đánh giá cho {model_name}:")
    for data_split, scores in metrics.items():
        print(f"  {data_split}: RMSE={scores['RMSE']:.2f}, MAE={scores['MAE']:.2f}, R²={scores['R²']:.2f}")
