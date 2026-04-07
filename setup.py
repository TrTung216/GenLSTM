from setuptools import setup, find_packages

setup(
    name="GenLSTM",
    version="1.0.0",
    description="Hệ thống dự báo giá cổ phiếu tự động tối ưu hóa bằng thuật toán GA-CNN-LSTM",
    author="TrTung216",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "yfinance>=0.2.40",
        "torch",
        "scikit-learn",
        "Flask",
        "joblib",
        "matplotlib",
        "pandas_market_calendars"
    ],
    python_requires=">=3.8",
)