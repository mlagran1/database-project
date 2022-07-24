'''
Author: Morris LaGrand
Date: July, 2022

A series of functions that will create and interact with the database
'''

# Imports
import pandas as pd
import numpy as np
import sqlite3
import os

# GLOBALS
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_FILE = os.environ.get("DATA_FILE", "train.csv")
DATABASE = os.environ.get("DATABASE", "titanic")

class DB:
    def __init__(self):
        self.conn = sqlite3.connect(f"{DATA_PATH}/{DATABASE}")
        self.c = self.conn.cursor()
        self.initialized = False

    def _load_clean_data(self, path):
        # Read data into dataframe
        df = pd.read_csv(path)

        # convert ages to ints
        df['Age'] = df['Age'].astype("int32", errors="ignore")
        # convert fare to 2 decimal place float
        df['Fare'] = df['Fare'].round(2)
        # encoding male and female to binary values for classification purposes
        df.loc[df['Sex'] == "male", "Sex"] = 0
        df.loc[df['Sex'] == "female", "Sex"] = 1
        # reorder the dataframe
        ordered_df = df[['PassengerId','Name','Sex','Age','Survived','Pclass','SibSp',
                         'Parch','Ticket','Fare','Embarked']]
        # Create class var for the dataframe
        self.df = ordered_df

    def _create_pclass_table(self):
        try:
            classes = [[1,"upper"], [2,"middle"], [3,"lower"]]
            # Create table
            self.c.execute('''
                CREATE TABLE Pclass(
                    pclass_id INT PRIMARY KEY,
                    class VARCHAR(10)
                )
            ''')
            # Load data into table
            self.c.executemany("INSERT INTO Pclass VALUES (?, ?)", classes)
        except:
            print("Table Pclass already exists.")

    def _create_port_table(self):
        try:
            ports = [["C","Cherbough"], ["Q","Queenstown"], ["S","Southhampton"]]
            # Create table
            self.c.execute('''
                CREATE TABLE Port(
                    port_id VARCHAR(1) PRIMARY KEY,
                    name VARCHAR(20)
                )
            ''')
            # Load data into table
            self.c.executemany("INSERT INTO Port VALUES (?, ?)", ports)
        except:
            print("Table Port already exists.")

    def _create_passenger_table(self):
        try:
            # Create table
            self.c.execute('''
                CREATE TABLE Passenger(
                    passenger_id INT PRIMARY KEY,
                    name VARCHAR(50),
                    sex BINARY,
                    age INT,
                    survived BINARY,
                    pclass_id INT,
                    sibsp INT,
                    parch INT,
                    ticket VARCHAR(50),
                    fare FLOAT(2),
                    port_id VARCHAR(1),
                    FOREIGN KEY (pclass_id) REFERENCES Pclass(pclass_id),
                    FOREIGN KEY (port_id) REFERENCES Port(port_id)
                )
            ''')
            # Load data into table
            self.c.executemany("INSERT INTO Passenger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", self.df.values)
        except:
            print("Table Passenger already exists.")

    def create_database(self):
        if not self.initialized:
            self._load_clean_data(f"{DATA_PATH}/{DATA_FILE}")
            self._create_pclass_table()
            self._create_port_table()
            self._create_passenger_table()
            # Set initialized to True
            self.initialized = True
        else:
            print("Database already initialized.")

    def get_all_data(self):
        df = pd.read_sql_query("select * from Passenger", self.conn)
        return df
