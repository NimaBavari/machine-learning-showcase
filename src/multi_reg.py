"""Machine learning fundamentals.

SUPERVISED LEARNING / Multiple linear regression
Given n data points (each a p-dimensional vectors) X_i and their respective n
output values y_i, we want to find such a^(j) real numbers such that
y(X) = a^(0) + a^(1) * x^(1) + ... + a^(p) * x^(p) is the closest approx. to
y_i over the values X_i; where X = (x^(1), x^(2), ..., x^(p)). Hence, we want
to minimize (y_1 - y(X_1))^2 + (y_2 - y(X_2))^2 + ... + (y_n - y(X_n))^2.
See: https://en.wikipedia.org/wiki/Linear_regression#Least-squares_estimation_and_related_techniques
for the solution.

https://www.github.com/NimaBavari
nima.bavari@gmail.com
"""
import csv
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import data_dir

HOUSE_PRICES_DATALOC = os.path.join(data_dir, "house-prices.csv")

# Mark the values for numerization
binary_num_dict = {"No": 0, "Yes": 1}
direction_num_dict = {"North": 0, "East": 1, "South": 2, "West": 3}

# Read the data with the numerized values
house_price_df = pd.read_csv(
    HOUSE_PRICES_DATALOC,
    converters={
        "Brick": lambda val: binary_num_dict[val],
        "Neighborhood": lambda val: direction_num_dict[val],
    },
    usecols=[
        "Price",
        "SqFt",
        "Bedrooms",
        "Bathrooms",
        "Offers",
        "Brick",
        "Neighborhood",
    ],
)

# Get the free and dependent values
X = house_price_df[["SqFt", "Bedrooms", "Bathrooms", "Offers", "Brick", "Neighborhood"]]
y = house_price_df["Price"]

# Define model, fit your data, and predict
lin_reg_mdl = LinearRegression()
lin_reg_mdl.fit(X, y)

sq_ft = float(input("Squate footage: "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
offers = int(input("Offers: "))
brick = 1 if input("Brick? (Y for Yes) ") == "Y" else 0
neighborhood = direction_num_dict[input("Neighborhood: ")]

input_vals = np.array([sq_ft, bedrooms, bathrooms, offers, brick, neighborhood], ndmin=2)
predicted_price = lin_reg_mdl.predict(input_vals).item()
print("Price is predicted to be %f." % predicted_price)

# Now user feedback and self-correction
actual_price = input("What is the actual price? (Return if actual price is not known.) ")
if actual_price:
    actual_price = float(actual_price)
    with open(HOUSE_PRICES_DATALOC, mode="a") as prices_datafile:
        writer = csv.writer(prices_datafile, delimiter=",")
        writer.writerow(["x", actual_price, sq_ft, bedrooms, bathrooms, offers, brick, neighborhood])
