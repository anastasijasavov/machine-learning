import marimo

__generated_with = "0.9.32"
app = marimo.App()


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.impute import KNNImputer
    from sklearn import preprocessing
    return KNNImputer, np, pd, plt, preprocessing, sns


@app.cell
def __(pd):
    df = pd.read_csv(r'data/water_potability.csv')

    df.head()
    return (df,)


@app.cell
def __(df):
    df.info()
    return


@app.cell
def __(df):
    df.shape
    return


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df):
    df.isnull().sum()
    return


@app.cell
def __(df):
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())
    df_1 = df.drop(columns=['Sulfate'])
    return (df_1,)


@app.cell
def __(KNNImputer, df_1):
    imputer = KNNImputer(n_neighbors=2)
    df_filled = imputer.fit_transform(df_1)
    return df_filled, imputer


@app.cell
def __(df_1, df_filled, imputer):
    df_1[:] = imputer.fit_transform(df_filled)
    return


@app.cell
def __(df_1, plt, sns):
    sns.countplot(df_1, x=df_1['Potability'])
    plt.show()
    return


@app.cell
def __(df_1):
    # see charts compared to potability
    df_potable = df_1[df_1["Potability"] == 1]
    df_nonpotable = df_1[df_1["Potability"] == 0]
    return df_nonpotable, df_potable


@app.cell
def __(df_nonpotable, df_potable, plt):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
    cols1 = ["Hardness", "Conductivity"]
    df_potable[cols1].boxplot(ax=ax1)
    ax1.set_title('potable')
    df_nonpotable[cols1].boxplot(ax=ax2)
    ax2.set_title('non potable')

    plt.tight_layout()
    plt.show()
    return ax1, ax2, cols1, fig


@app.cell
def __(ax1, ax2, df_nonpotable, df_potable, plt):
    cols2= ["ph", "Chloramines", "Organic_carbon", "Turbidity"]
    fig2, (ax1_2, ax2_2) = plt.subplots(ncols=2, figsize=(15, 5))

    df_potable[cols2].boxplot(ax=ax1_2)
    ax1.set_title('potable')
    df_nonpotable[cols2].boxplot(ax=ax2_2)
    ax2.set_title('non potable')

    plt.tight_layout()
    plt.show()
    return ax1_2, ax2_2, cols2, fig2


@app.cell
def __(df_1, plt, sns):
    sns.heatmap(df_1.corr(), cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Correlation of Water Potability')
    plt.show()
    return


@app.cell
def __(df_1, pd, preprocessing):
    # scaling all columns using Standard Scaler 
    scaler = preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(df_1.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=df_1.columns)
    return df_scaled, scaler


if __name__ == "__main__":
    app.run()
