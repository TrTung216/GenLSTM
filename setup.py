from setuptools import setup, find_packages

setup(
    name="GenLSTM",
    version="0.1.0",
    description="Dự báo giá cổ phiếu bằng GA-CNN-LSTM",
    author="Tên của bạn",
    packages=find_packages(), # Tự động tìm thư mục src/
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "torch",
        "scikit-learn",
        "Flask"
    ],
)