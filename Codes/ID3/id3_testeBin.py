import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def calcular_entropia(y):
    valores, contagem = np.unique(y, return_counts=True)
    probabilidade = contagem / len(y)
    return -np.sum(probabilidade * np.log2(probabilidade))

def calcular_ganho_informacao(X, y, atributo):
    entropia_total = calcular_entropia(y)
    valores = np.unique(X[atributo])
    entropia_ponderada = 0
    for valor in valores:
        y_subconjunto = y[X[atributo] == valor]
        probabilidade_subconjunto = len(y_subconjunto) / len(y)
        entropia_ponderada += probabilidade_subconjunto * calcular_entropia(y_subconjunto)
    return entropia_total - entropia_ponderada

def id3(X, y, atributos):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if len(atributos) == 0:
        return np.argmax(np.bincount(y))
    ganhos = [calcular_ganho_informacao(X, y, atributo) for atributo in atributos]
    melhor_atributo = atributos[np.argmax(ganhos)]
    arvore = {melhor_atributo: {}}
    valores = np.unique(X[melhor_atributo])
    for valor in valores:
        X_sub = X[X[melhor_atributo] == valor]
        y_sub = y[X[melhor_atributo] == valor]
        atributos_restantes = [a for a in atributos if a != melhor_atributo]
        arvore[melhor_atributo][valor] = id3(X_sub, y_sub, atributos_restantes)
    return arvore

def prever(arvore, exemplo):
    if isinstance(arvore, dict):
        atributo = next(iter(arvore))
        valor = exemplo[atributo]
        if valor in arvore[atributo]:
            return prever(arvore[atributo][valor], exemplo)
        else:
            return None
    else:
        return arvore

def avaliar_modelo(arvore, X_test, y_test):
    previsoes = [prever(arvore, exemplo) for _, exemplo in X_test.iterrows()]
    acuracia = np.mean(np.array(previsoes) == y_test)
    return acuracia

def executar_id3_com_bins(df, bins):
    df_discretizado = df.copy()
    df_discretizado['qPA'] = pd.cut(df['qPA'], bins=10, labels=[f'bin{i}' for i in range(1, 10 + 1)])
    df_discretizado['pulso'] = pd.cut(df['pulso'], bins=bins, labels=[f'bin{i}' for i in range(1, bins + 1)])
    df_discretizado['resp'] = pd.cut(df['resp'], bins=10, labels=[f'bin{i}' for i in range(1, 10 + 1)])
    X = df_discretizado.drop('classe', axis=1)
    y = df_discretizado['classe']
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1, random_state=42)
    atributos = X_train.columns
    arvore = id3(X_train, y_train, atributos)
    acuracia = avaliar_modelo(arvore, X_test, y_test)
    return acuracia

df = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',')
df.columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
df = df.drop(['id', 'pSist', 'pDiast', 'gravidade'], axis=1)

bins_range = range(1, 201)
acuracias = []

for bins in bins_range:
    acuracia = executar_id3_com_bins(df, bins)
    acuracias.append(acuracia)
    print(f"Bins: {bins}, Acurácia: {acuracia * 100:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(bins_range, [a * 100 for a in acuracias], marker='o', linestyle='-')
plt.title("Impacto do Número de Bins na Acurácia do ID3")
plt.xlabel("Número de Bins")
plt.ylabel("Acurácia (%)")
plt.xticks(bins_range)
plt.grid(True)
plt.show()
plt.savefig('impacto_bins_acuracia_id3.png', format='png', dpi=300, bbox_inches='tight')
