import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import ydata_profiling as pandas_profiling
from tkinterweb import HtmlFrame
from os.path import exists
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class App(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Sample Dataset Viewer")
        self.master.geometry("800x600")
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.create_csv_tab()
        self.create_desc_tab()
        self.create_model_tab()

    def create_csv_tab(self):
        self.csv_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.csv_tab, text="CSV Content")
        self.create_csv_widgets()

    def create_model_tab(self):
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model")
        self.create_model_widgets()

    def create_model_widgets(self):
        self.train_test_label = ttk.Label(
            self.model_tab, text="Train/Test Split (%):")
        self.train_test_label.grid(row=0, column=0, padx=10, pady=10)
        self.train_test_entry = ttk.Entry(self.model_tab)
        self.train_test_entry.insert(0, "80")
        self.train_test_entry.grid(row=0, column=1, padx=10, pady=10)

        self.reg_label = ttk.Label(
            self.model_tab, text="Regularization Hyperparameter (λ):")
        self.reg_label.grid(row=1, column=0, padx=10, pady=10)
        self.reg_entry = ttk.Entry(self.model_tab)
        self.reg_entry.insert(0, "0.0")
        self.reg_entry.grid(row=1, column=1, padx=10, pady=10)

        self.error_label = ttk.Label(self.model_tab, text="Error Metric:")
        self.error_label.grid(row=2, column=0, padx=10, pady=10)
        self.error_combobox = ttk.Combobox(
            self.model_tab, values=["MSE", "RMSE", "MAE"])
        self.error_combobox.current(0)
        self.error_combobox.grid(row=2, column=1, padx=10, pady=10)

        self.train_button = ttk.Button(
            self.model_tab, text="Calculate error", command=self.generate_predictions)
        self.train_button.grid(row=3, column=0, padx=10, pady=10)

        self.train_error_label = ttk.Label(
            self.model_tab, text="Train Error: N/A")
        self.train_error_label.grid(row=4, column=0, padx=10, pady=10)

        self.test_error_label = ttk.Label(
            self.model_tab, text="Test Error: N/A")
        self.test_error_label.grid(row=4, column=1, padx=10, pady=10)

    def create_csv_widgets(self):
        self.load_button = ttk.Button(
            self.csv_tab, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)
        self.csv_columns = []
        self.csv_rows = []
        self.csv_table = ttk.Treeview(
            self.csv_tab, columns=self.csv_columns, show="headings")
        self.csv_table.grid(row=1, column=0, padx=10, pady=10)

    def create_desc_tab(self):
        self.desc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.desc_tab, text="Data Description")
        self.create_desc_widgets()

    def create_desc_widgets(self):
        # Create a button to generate the data description HTML file
        self.desc_button = ttk.Button(
            self.desc_tab, text="Generate Data Description", command=self.generate_data_desc)
        self.desc_button.grid(row=0, column=0, padx=10, pady=10)

        # Create an HTMLLabel widget to display the data description HTML file
        self.desc_viewer = HtmlFrame(self.desc_tab)
        self.desc_viewer.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.desc_viewer.columnconfigure(0, weight=1)
        self.desc_viewer.rowconfigure(0, weight=1)

    def load_csv(self):
        try:
            df = pd.read_csv("diabetes.csv")
            self.csv_columns = list(df.columns)
            self.csv_table.config(columns=self.csv_columns)
            for col in self.csv_columns:
                self.csv_table.heading(col, text=col)
            for i, row in df.iterrows():
                self.csv_rows.append([str(row[col])
                                     for col in self.csv_columns])
            for row in self.csv_table.get_children():
                self.csv_table.delete(row)
            for row in self.csv_rows:
                self.csv_table.insert("", "end", values=row)
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find the CSV file.")

    def generate_data_desc(self):
        try:
            # Load the CSV file and generate the data description using pandas_profiling
            df = pd.read_csv("diabetes.csv")
            desc = pandas_profiling.ProfileReport(df)
            # Write the data description to an HTML file
            with open("profiling_file.html", "w") as f:
                f.write(desc.to_html())
            self.set_html();
        except FileNotFoundError:
                messagebox.showerror("Error", "Could not find the CSV file.")

    def set_html(self):
        try:
            # Display the data description in the HTML viewer
            with open("profiling_file.html", "r") as f:
                self.desc_viewer.load_html(f.read())
        except FileNotFoundError:
                messagebox.showerror("Error", "Could not find the CSV file.")
            

    def generate_predictions(self):
        # Get the train/test split ratio from the user input
        train_test_split_ratio = int(self.train_test_entry.get()) / 100

        # Get the regularization hyperparameter from the user input
        reg_param = float(self.reg_entry.get())

        # Get the error metric from the user input
        error_metric = self.error_combobox.get()

        # Load the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv("diabetes.csv")
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find the CSV file.")
            return

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["target"]), df["target"], test_size=1-train_test_split_ratio)

        # Fit a linear regression model with L2 regularization to the training data
        print(reg_param)
        print(len(X_train))
        self.model = Ridge(alpha=reg_param)
        self.model.fit(X_train, y_train)
        

        # Predict the target values for the training and testing data
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        # Calculate the error metric on the predicted values and the actual values for the training data
        if error_metric == "MSE":
            train_error = mean_squared_error(y_train, train_predictions)
            test_error = mean_squared_error(y_test, test_predictions)
        elif error_metric == "RMSE":
            train_error = mean_squared_error(y_train, train_predictions, squared=False)
            test_error = mean_squared_error(y_test, test_predictions, squared=False)
        elif error_metric == "MAE":
            train_error = mean_absolute_error(y_train, train_predictions)
            test_error = mean_absolute_error(y_test, test_predictions)

        # Update the GUI with the train error
        self.train_error_label.config(text="Train Error: {:.2f}".format(train_error))
        self.test_error_label.config(text="Test Error: {:.2f}".format(test_error))

root=tk.Tk()
app=App(root)
HtmlFile_exists=exists("profiling_file.html")
if (HtmlFile_exists):
    app.set_html()
app.mainloop()
