import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


# Define polynomial regression function
def polynomial_regression(data, target, degree, alpha):
    polynomial_features = PolynomialFeatures(degree=degree)
    model = Ridge(alpha=alpha)
    pipeline = make_pipeline(polynomial_features, model)
    pipeline.fit(data, target)
    return pipeline


# Create data
def create_data(num_data_points):
    x = np.linspace(0, 2 * np.pi, num_data_points)
    np.random.seed(1)
    y = np.sin(x) + np.random.normal(0, 0.2, num_data_points)
    x_range = np.linspace(0, 2 * np.pi, 1000)
    return x_range, x, y


# Update and display the plot
def update_plot(data_points, degree, alpha):
    x_range, x_train, y_train = create_data(data_points)

    model = polynomial_regression(x_train.reshape(-1, 1), y_train, degree, alpha)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.plot(x_range, np.sin(x_range), color='blue', label='Sine Wave')
    ax.plot(x_range, y_range, label='Model Function', color='red', linestyle='dashed')
    ax.scatter(x_train, y_train, label='Data Points', color='purple', marker='o', s=100)
    ax.legend()

    st.pyplot(fig)


# Sidebar for sliders
st.sidebar.header("Controls")
data_points = st.sidebar.slider("Data Points", min_value=10, max_value=100, value=10, step=1)
degree = st.sidebar.slider("Polynomial Order", min_value=0, max_value=15, value=0, step=1)
alpha = st.sidebar.slider("Alpha", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

# Main content area for the plot
st.write("## Regression Plot")
update_plot(data_points, degree, alpha)
