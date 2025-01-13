import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import numpy as np


os.environ['PATH'] += r"C:\selenium drivers"
driver = webdriver.Chrome()
driver.get("https://www.cars24.com/buy-used-cars-delhi-ncr/?sort=bestmatch&serveWarrantyCount=true&storeCityId=1")
time.sleep(3)
height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if height == new_height:
        print("completed")
        break
    height = new_height

names = driver.find_elements(By.CLASS_NAME, "_2lmIw")

car_names = [name.text for name in names]


car_data = driver.find_elements(By.CLASS_NAME, "_13yb6")
car_info = [info.text.split('\n') for info in car_data]
car_dist = [sublist[0] for sublist in car_info]
car_owner = [sublist[1] for sublist in car_info]
car_fuel = [sublist[2] for sublist in car_info]
car_state = [sublist[3] if len(sublist) > 3 and sublist[3] != '' else np.nan for sublist in car_info]


ul_elements = driver.find_elements(By.CLASS_NAME, "_1hOnS")
car_geartype = []

for ul_element in ul_elements:
    elements = ul_element.find_elements(By.XPATH, ".//li[text()='Manual' or text()='Automatic']")
    car_geartype.extend([element.text for element in elements])



div_elements = driver.find_elements(By.CLASS_NAME, "_18ToE")

car_cost = []
for div_element in div_elements:
    span_element = div_element.find_element(By.TAG_NAME, "span")
    value = span_element.text
    car_cost.append(value)

if __name__ == "__main__":
    # Load dataset from kaggle path
    df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")

    #Data Cleaning - Remove year from Car Name
    df["Car Name"] = df["Car Name"].str.split(" ", n=1).str[1]

    #Rename columns to match scraped data format
    df = df.rename(columns={
        "car":"Car Name",
        "year": "Year",
        "km_driven":"Distance",
        "owner":"Owner",
        "fuel":"Fuel",
        "location":"Location",
        "transmission":"Gear Type",
        "price":"Price"
    })

    # Select columns and fill null values
    df = df[["Car Name", "Distance", "Owner", "Fuel", "Location","Gear Type","Price"]]
    df["Location"] = df["Location"].fillna("Unknown")

    df.to_csv("data/raw_data.csv", index=False)
    print("Raw data saved to data/raw_data.csv")