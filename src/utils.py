# librerías usadas en útils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import math
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from pickle import dump
# ----------------------------------------------------------------------------------------------
# ANALISIS VARIABLES NUMERICAS
def plot_numerical_data(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    n = len(numerical_columns)
    cols = 3
    rows = 3  # fijo a 3 filas: 1 o 2 filas para histogramas y 1 o 2 para boxplots según cantidad

    # Calculamos cuántas columnas y filas necesitamos para histogramas y boxplots
    # Distribuimos histogramas en la fila 0 y 1 si hay más de 3 variables
    # Boxplots en la fila 2 (y fila 3 si necesario, pero aquí fijo 3 filas, así que ajustamos)

    # Para simplificar, pondremos histogramas en la fila 0 y boxplots en la fila 1 y 2 si es necesario

    # Número de columnas para histogramas y boxplots
    hist_cols = min(cols, n)
    box_cols = min(cols, n)

    # Número de filas para histogramas y boxplots
    hist_rows = 1 if n <= cols else 2
    box_rows = 1 if n <= cols else 2

    fig_height = 4 * (hist_rows + box_rows)
    fig, axes = plt.subplots(hist_rows + box_rows, cols, figsize=(5 * cols, fig_height))

    # Aplanar axes para facilitar indexación
    axes = axes.reshape(hist_rows + box_rows, cols)

    # Graficar histogramas
    for i, column in enumerate(numerical_columns):
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        std_dev = np.std(dataframe[column])

        # Índices para histogramas
        hist_row = i // cols
        hist_col = i % cols

        ax_hist = axes[hist_row, hist_col]
        sns.histplot(data=dataframe, x=column, kde=True, ax=ax_hist).set(xlabel=None)
        ax_hist.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        ax_hist.axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        ax_hist.axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1, label='Std Dev')
        ax_hist.axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)
        ax_hist.legend()
        ax_hist.set_title(f'Distribución de {column}')

    # Graficar boxplots
    for i, column in enumerate(numerical_columns):
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        std_dev = np.std(dataframe[column])

        # Índices para boxplots
        box_row = hist_rows + (i // cols)
        box_col = i % cols

        # Si box_row excede filas disponibles, no graficar (ajustar según filas)
        if box_row >= hist_rows + box_rows:
            continue

        ax_box = axes[box_row, box_col]
        sns.boxplot(data=dataframe, x=column, ax=ax_box, width=0.6).set(xlabel=None)
        ax_box.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
        ax_box.axvline(median_val, color='orange', linestyle='dashed', linewidth=1)
        ax_box.axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1)
        ax_box.axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)
        ax_box.set_title(f'Boxplot de {column}')

    # Ocultar ejes vacíos
    total_plots = (hist_rows + box_rows) * cols
    used_plots = max(n, n) + max(0, n - cols)  # aproximado
    for idx in range(total_plots):
        row = idx // cols
        col = idx % cols
        # Si el subplot está fuera del rango de histogramas y boxplots usados
        if (row < hist_rows and idx >= n) or (row >= hist_rows and idx - hist_rows*cols >= n):
            fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()
#-------------------------------------------------------------------------------------
# MAPA DE CORRELACION ENTRE LA NUMERICAS

def heatmap_correlation(dataframe):
    # Seleccionar solo columnas numéricas
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64'])

    # Calcular matriz de correlación
    corr = numerical_columns.corr()

    # Crear máscara para la mitad inferior
     # mask = np.tril(np.ones_like
    mask = np.tril(np.ones_like(corr, dtype=bool))
   
    # Configurar tamaño del gráfico
    plt.figure(figsize=(10, 8))

    # Dibujar heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": .5})

    plt.title('Mapa de correlación')
    plt.show()
#---------------------------------------------------------------------------------------------------
#ANALISIS VARIABLES CATEGORICAS
def plot_categorical_data(dataframe):
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Conteo de cada categoría (barras)
        sns.countplot(x=column, data=dataframe, ax=axs[0],hue=column, palette='pastel')
        axs[0].set_title(f'Conteo de categorías en {column}')
        axs[0].set_xlabel('')
        axs[0].set_ylabel('Frecuencia')
        axs[0].tick_params(axis='x', rotation=45)
        
        # Gráfico de pastel (proporciones)
        dataframe[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axs[1], colors=sns.color_palette('pastel'))
        axs[1].set_ylabel('')
        axs[1].set_title(f'Proporción de categorías en {column}')
        
        plt.tight_layout()
        plt.show()
#------------------------------------------------------------
#ANALISIS DE CORRELACION NUMERICAS VS TARGET

def correlation_num_target(dataframe, target):
    # Verificar que la columna target exista
    if target not in dataframe.columns:
        print(f"La columna '{target}' no está en el DataFrame.")
        return
    
    # Seleccionar columnas numéricas excepto la target
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns.drop(target)
    
    # Calcular correlaciones
    correlations = dataframe[numerical_columns].corrwith(dataframe[target]).sort_values(key=abs, ascending=False)
    
    # Mostrar correlaciones
    print(f"Correlaciones con la variable objetivo '{target}':")
    print(correlations)
    
    # Graficode barras
    plt.figure(figsize=(8, 5))
    correlations.plot(kind='bar', color='skyblue')
    plt.title(f'Correlación de variables numéricas con {target}')
    plt.ylabel('Coeficiente de correlación')
    plt.xlabel('Variables')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    #-------------------------------------------------------------
# CREACION DE LOS DATAFRAME 
def process_dataframes_shapes(dataframe, target):
   
    # 1. total_data: dataset original
    total_data = dataframe.copy()

    # 2. total_data_no_outliers: ajustar outliers en variables numéricas (winsorizing)
    def adjust_outliers(data):
        df_adj = data.copy()
        numeric_cols = df_adj.select_dtypes(include=['float64', 'int64']).columns.drop(target)
        for col in numeric_cols:
            Q1 = df_adj[col].quantile(0.25)
            Q3 = df_adj[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_adj[col] = np.where(df_adj[col] < lower_bound, lower_bound, df_adj[col])
            df_adj[col] = np.where(df_adj[col] > upper_bound, upper_bound, df_adj[col])
        return df_adj

    total_data_no_outliers = adjust_outliers(total_data)

    # 3. total_data_factorized: factorizar variables categóricas en total_data
    total_data_factorized = total_data.copy()
    cat_cols = total_data_factorized.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        total_data_factorized[col] = total_data_factorized[col].astype('category').cat.codes

    # 4. total_data_no_outliers_factorized: factorizar variables categóricas en total_data_no_outliers
    total_data_no_outliers_factorized = total_data_no_outliers.copy()
    cat_cols_no_out = total_data_no_outliers_factorized.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols_no_out:
        total_data_no_outliers_factorized[col] = total_data_no_outliers_factorized[col].astype('category').cat.codes

    # Función para escalar (sin escalar la variable target)
    def scale_data(data, scaler):
        data_scaled = data.copy()
        numeric_cols = data_scaled.select_dtypes(include=['float64', 'int64']).columns.drop(target)
        data_scaled[numeric_cols] = scaler.fit_transform(data_scaled[numeric_cols])
        return data_scaled

    # 5. total_data_standard: escalado Standard en total_data_factorized
    total_data_standard = scale_data(total_data_factorized, StandardScaler())

    # 6. total_data_no_outliers_standard: escalado Standard en total_data_no_outliers_factorized
    total_data_no_outliers_standard = scale_data(total_data_no_outliers_factorized, StandardScaler())

    # 7. total_data_factorized_standard: igual que total_data_standard (ya factorizado)
    total_data_factorized_standard = total_data_standard.copy()

    # 8. total_data_no_outliers_factorized_standard: igual que total_data_no_outliers_standard
    total_data_no_outliers_factorized_standard = total_data_no_outliers_standard.copy()

    # 9. total_data_minmax: escalado MinMax en total_data_factorized
    total_data_minmax = scale_data(total_data_factorized, MinMaxScaler())

    # 10. total_data_no_outliers_minmax: escalado MinMax en total_data_no_outliers_factorized
    total_data_no_outliers_minmax = scale_data(total_data_no_outliers_factorized, MinMaxScaler())

    # 11. total_data_factorized_minmax: igual que total_data_minmax
    total_data_factorized_minmax = total_data_minmax.copy()

    # 12. total_data_no_outliers_factorized_minmax: igual que total_data_no_outliers_minmax
    total_data_no_outliers_factorized_minmax = total_data_no_outliers_minmax.copy()

    return {
        'total_data': total_data.shape,
        'total_data_no_outliers': total_data_no_outliers.shape,
        'total_data_factorized': total_data_factorized.shape,
        'total_data_no_outliers_factorized': total_data_no_outliers_factorized.shape,
        'total_data_standard': total_data_standard.shape,
        'total_data_no_outliers_standard': total_data_no_outliers_standard.shape,
        'total_data_factorized_standard': total_data_factorized_standard.shape,
        'total_data_no_outliers_factorized_standard': total_data_no_outliers_factorized_standard.shape,
        'total_data_minmax': total_data_minmax.shape,
        'total_data_no_outliers_minmax': total_data_no_outliers_minmax.shape,
        'total_data_factorized_minmax': total_data_factorized_minmax.shape,
        'total_data_no_outliers_factorized_minmax': total_data_no_outliers_factorized_minmax.shape
    }

#-------------------------------------------------------------------------
# DATASET TRAIN/TEST SPLITS

def generate_train_test_splits(datasets_dict, target, test_size=0.2, random_state=42):
   
    splits = {}

    for name, df in datasets_dict.items():
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        splits[f'{name}_X_train'] = X_train
        splits[f'{name}_X_test'] = X_test
        splits[f'{name}_y_train'] = y_train
        splits[f'{name}_y_test'] = y_test

    return splits
#-----------------------------------------------------------------------------




