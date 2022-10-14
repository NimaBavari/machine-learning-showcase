"""Machine learning fundamentals.

SUPERVISED LEARNING / Support Vector Classification
Given n data points (each a p-dimensional vectors) X_i and their respective
binary category as +1 or -1, we want to draw a boundary around each of the two
categories in such a way that the area between the two boundaries where they
face each other is maximized and with the condition that every point is
contained within the boundary of its category. Here we use the C-support Vector
Classifier with the radial basis function (RBF) as the kernel.
See: https://en.wikipedia.org/wiki/Support_vector_machine#Nonlinear_Kernels for
details.

https://www.github.com/NimaBavari
nima.bavari@gmail.com
"""
import csv
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.svm import SVC

from constants import data_dir

DIGITS_DATALOC = os.path.join(data_dir, "digits.csv")

# Build the dataframe
digits_df = None
if not os.path.exists(DIGITS_DATALOC):
    with open(DIGITS_DATALOC, "w") as digits_datafile:
        digits_container = load_digits(as_frame=True)
        digits_df = digits_container["frame"]
        digits_df.to_csv(digits_datafile, index=False)
else:
    digits_df = pd.read_csv(DIGITS_DATALOC)

# Get the free and dependent values
X = digits_df.loc[:, digits_df.columns != "target"]
y = digits_df["target"]

# Define classifier and fit your data
classifier = SVC(C=1.0, kernel="rbf")
classifier.fit(X, y)

# Get user input for an image
image_loc = input("8x8 image path: ").strip()

# Read the pixel information of the image; then predict it
image = Image.open(image_loc, "r")
if image.height == 8 and image.width == 8:
    image_data = np.array([pixel_rgb / 255 for pixel_rgb in image.getdata()], ndmin=2)
    predicted_digit = classifier.predict(image_data).item()
    print("The digit is predicted to be: %d." % predicted_digit)

    # User feedback and self-correction
    actual_digit = input("What is the actual digit? (Return if not known.) ")
    if actual_digit and actual_digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        actual_digit = int(actual_digit)
        with open(DIGITS_DATALOC, "a") as digits_datafile:
            writer = csv.writer(digits_datafile, delimiter=",")
            writer.writerow([*image_data.flatten(), actual_digit])
    else:
        print("You did not enter a valid digit.")
else:
    print("The image you entered is not 8x8.")
