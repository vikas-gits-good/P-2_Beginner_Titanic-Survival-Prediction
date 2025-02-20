import re
import pandas as pd


df = pd.read_csv("artifacts/01_Data/train.csv", usecols=["Name"])

for name in df["Name"]:
    df.loc[df["Name"] == name, "LastName"] = (
        re.search(r"^([^,]+),", name).group(1).strip()
    )
    df.loc[df["Name"] == name, "Title"] = (
        re.search(r",\s*([^\.]+)\.", name).group(1).strip()
    )

last_name_list = df["LastName"].unique().tolist()
title_list = df["Title"].unique().tolist()
