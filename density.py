import pandas as pd
import requests
from matplotlib import pyplot as plt

# Load data from a text file
url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv"
df_students = requests.get(url)

if df_students.status_code == 200:
    with open("grades.csv", "wb") as f:
        f.write(df_students.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {df_students.status_code}")

df_students = pd.read_csv('grades.csv', delimiter=',', header='infer')


def show_density(var_data):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10,4))

    # Plot density
    var_data.plot.density()

    # Add titles and labels
    plt.title('Data Density')

    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # Show the figure
    plt.show()

# Get the density of Grade
col = df_students['Grade']
show_density(col)