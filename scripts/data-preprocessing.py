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
    return KNNImputer, np, pd, plt, sns


@app.cell
def __():
    import os
    print(os.getcwd())
    return (os,)


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
def __(df_1, plt):
    box_plot_cols = df_1[['Hardness', 'Conductivity']]
    box_plot_cols.plot(kind='box')
    plt.show()
    return (box_plot_cols,)


@app.cell
def __(df_1, plt, sns):
    sns.countplot(df_1, x=df_1['Potability'])
    plt.show()
    return


@app.cell
def __(df_1, plt, sns):
    sns.heatmap(df_1.corr(), cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Correlation of Water Potability')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
