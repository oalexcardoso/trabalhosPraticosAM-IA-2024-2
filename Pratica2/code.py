# pip install pandas plotly scikit-learn tqdm scipy

import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
from scipy.io.arff import loadarff
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore")

timeTotal = time.time()

# py.init_notebook_mode(connected=False)

# Função para chamada do gráfico interativo

# def configure_plotly_browser_state():
#    import IPython
#    display(IPython.core.display.HTML('''
#        <script src="/static/components/requirejs/require.js"></script>
#        <script>
#          requirejs.config({
#            paths: {
#              base: '/static/base',
#              plotly: 'https://cdn.plot.ly/plotly-1.43.1.min.js?noext',
#            },
#          });
#        </script>
#        '''))

# Caminho da pasta com os arquivos .arff
folder_path = "."

# Lista todos os arquivos na pasta
files = [f for f in os.listdir(folder_path) if f.endswith('.arff')]

# Itera sobre cada arquivo .arff
for file_name in files:
    # Caminho completo do arquivo
    file_path = os.path.join(folder_path, file_name)

    # Carrega o .arff
    raw_data = loadarff(file_path)

    # Transforma o .arff em um DataFrame
    df = pd.DataFrame(raw_data[0])

    # Cria o nome do arquivo de saída .txt
    output_txt = os.path.join(
        folder_path, file_name.replace('.arff', '.txt'))

    # Salva as informações no arquivo .txt
    with open(output_txt, 'w') as f:
        f.write(f"Arquivo: {file_name}\n")
        # Converte o DataFrame para string e salva

        # Com o iloc voce retira as linhas e colunas que quiser do Dataframe, no caso aqui sem as classes
        X = df.iloc[:, 0:-1].values

        # Aqui salvamos apenas as classes agora
        y = df['class']
        # Substituimos os valores binários por inteiro
        bow = []
        int_value = 0
        y_aux = []
        for i in y:
            if i in bow:
                y_aux.append(int_value)
            else:
                bow.append(i)
                int_value += 1
                y_aux.append(int_value)
        # Novo y
        y = y_aux

        # Dividindo o conjunto em 80% Treino e 20% Teste.
        # O parâmetro random_state = 327 define que sempre será dividido da mesma forma o conjunto.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=327)

        f.write('\n Tamanho do conjuntos: Treino:{} e Teste:{} '.format(
            X_train.shape, X_test.shape))

        # f.write('\n Tamanho do conjunto de Treino: {}'.format(X_train.shape))
        # f.write('\n Tamanho do conjunto de Teste: {}'.format(X_test.shape))

        # Escolha umas das 4 técnicas de normalização existentes
        # 1 = MinMaxScaler, 2 = StandardScaler, 3 = MaxAbsScaler, 4 = RobustScaler
        for selectedNormalization in range(1, 5):

            f.write('\n --------------------------------------------')

            if selectedNormalization == 1:
                f.write('\n MinMaxScaler:')
            if selectedNormalization == 2:
                f.write('\n StandardScaler:')
            if selectedNormalization == 3:
                f.write('\n MaxAbsScaler:')
            if selectedNormalization == 4:
                f.write('\n RobustScaler:')

            if selectedNormalization == 1:
                scaler = preprocessing.MinMaxScaler()
            if selectedNormalization == 2:
                scaler = preprocessing.StandardScaler()
            if selectedNormalization == 3:
                scaler = preprocessing.MaxAbsScaler()
            if selectedNormalization == 4:
                scaler = preprocessing.RobustScaler()

            # Escalando os dados de treinamento
            X_train = scaler.fit_transform(X_train)
            # Escalando os dados de teste com os dados de treinamento, visto que os dados de teste podem ser apenas 1 amostra
            X_test = scaler.transform(X_test)

            # f.write('\n Média do Conjunto de Treinamento por Feature:')
            # f.write(X_train.mean(axis=0))
            # f.write(np.array2string(X_train.mean(axis=0)))
            # f.write('\n Desvio Padrão do Conjunto de Treinamento por Feature:')
            # f.write(X_train.std(axis=0))
            # f.write(np.array2string(X_train.std(axis=0)))

            # Inicializar os classificadores

            # Gaussian Naive Bayes
            t = time.time()
            gnb = GaussianNB()
            model1 = gnb.fit(X_train, y_train)
            # f.write('\n Treino do Gaussian Naive Bayes Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # Logistic Regression
            t = time.time()
            logreg = LogisticRegression()
            model2 = logreg.fit(X_train, y_train)
            # f.write('\n Treino do Logistic Regression Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # Decision Tree
            t = time.time()
            dectree = DecisionTreeClassifier()
            model3 = dectree.fit(X_train, y_train)
            # f.write('\n Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # K-Nearest Neighbors
            t = time.time()
            knn = KNeighborsClassifier(n_neighbors=3)
            model4 = knn.fit(X_train, y_train)
            # f.write('\n Treino do K-Nearest Neighbors Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # Linear Discriminant Analysis
            t = time.time()
            lda = LinearDiscriminantAnalysis()
            model5 = lda.fit(X_train, y_train)
            # f.write('\n Treino do Linear Discriminant Analysis Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # Support Vector Machine
            t = time.time()
            svm = SVC()
            model6 = svm.fit(X_train, y_train)
            # f.write('\n Treino do Support Vector Machine Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # RandomForest
            t = time.time()
            rf = RandomForestClassifier()
            model7 = rf.fit(X_train, y_train)
            # f.write('\n Treino do RandomForest Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # Neural Net
            t = time.time()
            nnet = MLPClassifier(alpha=1)
            model8 = nnet.fit(X_train, y_train)
            # f.write('\n Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(time.time() - t))

            # Cria 2 vetores de predicoes para armazenar todas acuracias e outros para as métricas
            acc_train = []
            acc_test = []
            f1score = []
            precision = []
            recall = []

            # Gaussian Naive Bayes

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = gnb.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(gnb.score(X_train, y_train))
            acc_test.append(gnb.score(X_test, y_test))
            f.write('\n Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[0]))
            f.write('\n Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Teste: {:.2f}'.format(
                acc_test[0]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[0]))
            f.write('\n Recall: {:.5f}'.format(recall[0]))
            f.write('\n F1-score: {:.5f}'.format(f1score[0]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # Logistic Regression

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = logreg.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(logreg.score(X_train, y_train))
            acc_test.append(logreg.score(X_test, y_test))
            f.write('\n Acuracia obtida com o Logistic Regression no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[1]))
            f.write('\n Acuracia obtida com o Logistic Regression no Conjunto de Teste: {:.2f}'.format(
                acc_test[1]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[1]))
            f.write('\n Recall: {:.5f}'.format(recall[1]))
            f.write('\n F1-score: {:.5f}'.format(f1score[1]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # Decision Tree

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = dectree.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(dectree.score(X_train, y_train))
            acc_test.append(dectree.score(X_test, y_test))
            f.write('\n Acuracia obtida com o Decision Tree no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[2]))
            f.write('\n Acuracia obtida com o Decision Tree no Conjunto de Teste: {:.2f}'.format(
                acc_test[2]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[2]))
            f.write('\n Recall: {:.5f}'.format(recall[2]))
            f.write('\n F1-score: {:.5f}'.format(f1score[2]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # K-Nearest Neighbors

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = knn.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(knn.score(X_train, y_train))
            acc_test.append(knn.score(X_test, y_test))
            f.write(
                '\n Acuracia obtida com o K-Nearest Neighbors no Conjunto de Treinamento: {:.2f}'.format(acc_train[3]))
            f.write(
                '\n Acuracia obtida com o K-Nearest Neighbors no Conjunto de Teste: {:.2f}'.format(acc_test[3]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[3]))
            f.write('\n Recall: {:.5f}'.format(recall[3]))
            f.write('\n F1-score: {:.5f}'.format(f1score[3]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # Linear Discriminant Analysis

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = lda.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(lda.score(X_train, y_train))
            acc_test.append(lda.score(X_test, y_test))
            f.write('\n Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[4]))
            f.write('\n Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Teste: {:.2f}'.format(
                acc_test[4]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[4]))
            f.write('\n Recall: {:.5f}'.format(recall[4]))
            f.write('\n F1-score: {:.5f}'.format(f1score[4]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # Support Vector Machine

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = svm.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(svm.score(X_train, y_train))
            acc_test.append(svm.score(X_test, y_test))
            f.write('\n Acuracia obtida com o Support Vector Machine no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[5]))
            f.write('\n Acuracia obtida com o Support Vector Machine no Conjunto de Teste: {:.2f}'.format(
                acc_test[5]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[5]))
            f.write('\n Recall: {:.5f}'.format(recall[5]))
            f.write('\n F1-score: {:.5f}'.format(f1score[5]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # RandomForest

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = rf.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(rf.score(X_train, y_train))
            acc_test.append(rf.score(X_test, y_test))
            f.write('\n Acuracia obtida com o RandomForest no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[6]))
            f.write('\n Acuracia obtida com o RandomForest no Conjunto de Teste: {:.2f}'.format(
                acc_test[6]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[6]))
            f.write('\n Recall: {:.5f}'.format(recall[6]))
            f.write('\n F1-score: {:.5f}'.format(f1score[6]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # Neural Net

            # Variavel para armazenar o tempo
            t = time.time()
            # Usando o modelo para predição das amostras de teste
            aux = nnet.predict(X_test)
            # Método para criar a matriz de confusão
            cm = confusion_matrix(y_test, aux)
            # Método para calcular o valor F1-Score
            f1score.append(f1_score(y_test, aux, average='macro'))
            # Método para calcular a Precision
            precision.append(precision_score(y_test, aux, average='macro'))
            # Método para calcular o Recall
            recall.append(recall_score(y_test, aux, average='macro'))
            # Salvando as acurácias nas listas
            acc_train.append(nnet.score(X_train, y_train))
            acc_test.append(nnet.score(X_test, y_test))
            f.write('\n Acuracia obtida com o Neural Net no Conjunto de Treinamento: {:.2f}'.format(
                acc_train[7]))
            f.write('\n Acuracia obtida com o Neural Net no Conjunto de Teste: {:.2f}'.format(
                acc_test[7]))
            f.write('\n Matriz de Confusão:')
            f.write(np.array2string(cm))
            f.write('\n Precision: {:.5f}'.format(precision[7]))
            f.write('\n Recall: {:.5f}'.format(recall[7]))
            f.write('\n F1-score: {:.5f}'.format(f1score[7]))
            f.write('\n (Tempo de execucao: {:.5f})'.format(
                time.time() - t))
            f.write('\n ')

            # Criando valores do eixo X
            eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree',
                      'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']

            # Gráfico 1: Acurácia dos Classificadores
            dados_train = go.Bar(x=eixo_x, y=acc_train,
                                 name='Conjunto de Treino')
            dados_test = go.Bar(x=eixo_x, y=acc_test, name='Conjunto de Teste')

            layout_1 = go.Layout(
                title='Acurácia dos Classificadores',
                xaxis={'title': 'Classificadores'},
                yaxis={'title': 'Acurácia'},
                paper_bgcolor='rgba(245, 246, 249, 1)',
                plot_bgcolor='rgba(245, 246, 249, 1)'
            )

            fig_1 = go.Figure(data=[dados_train, dados_test], layout=layout_1)

            # Salvar o gráfico 1
            nomeArquivo = file_name.replace('.arff', '')
            nomeNormalizacao = scaler.__class__.__name__
            output_file_1 = nomeArquivo + "_" + nomeNormalizacao + "_" + \
                "acuraciaClassificadores_" + ".png"
            fig_1.write_image(output_file_1)

            # Gráfico 2: Métricas de Avaliação
            dados_precision = go.Scatter(
                x=eixo_x, y=precision, name='Precision', mode='lines+markers')
            dados_recall = go.Scatter(
                x=eixo_x, y=recall, name='Recall', mode='lines+markers')
            dados_f1score = go.Scatter(
                x=eixo_x, y=f1score, name='F1-Score', mode='lines+markers')

            layout_2 = go.Layout(
                title='Métricas de Avaliação',
                xaxis={'title': 'Classificadores'},
                paper_bgcolor='rgba(245, 246, 249, 1)',
                plot_bgcolor='rgba(245, 246, 249, 1)'
            )

            fig_2 = go.Figure(
                data=[dados_precision, dados_recall, dados_f1score], layout=layout_2)

            # Salvar o gráfico 2
            nomeArquivo = file_name.replace('.arff', '')
            nomeNormalizacao = scaler.__class__.__name__
            output_file_2 = nomeArquivo + "_" + nomeNormalizacao + "_" + \
                "metricasAvaliacao" + ".png"
            fig_2.write_image(output_file_2)

print('\n (Tempo de execucao: {})'.format(time.time() - timeTotal))
