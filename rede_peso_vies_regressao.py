# Segunda rede um pouco mais complexa 
# Autor: Hemerson Pistori e VOCÊ AQUI !!!
# Diferenças em relação ao modelo mínimo:
# 1. A rede tem 2 pesos (um é o viés)
# 2. A rede é treinada para resolver um problema de regressão de uma função linear

import numpy as np # Importa a biblioteca que contém funções matemáticas
import random # Importa a biblioteca que contém funções aleatórias
import matplotlib.pyplot as plt # Importa a biblioteca que contém funções para plotar gráficos

EPOCAS=300  # Define total de vezes que IA vai passar por todos os exemplo de treinamento
# LIMIAR_DECISAO=0.5   # Será um problema de regressão desta vez (não classificação)
PESOS_INICIAIS=[random.uniform(-2, 2),random.uniform(-2,2)]   # Agora são 2 pesos (o segundo é, na verdade, um tipo especial de peso chamado viés)
TAXA_APRENDIZAGEM=0.01  # Define a taxa de aprendizagem da rede

erros=[] ## Armazena os erros durante o treinamento para plotar o gráfico
pesos=[] ## Armazena os pesos durante o treinamento para plotar o gráfico
vieses=[] ## Armazena os vieses durante o treinamento para plotar o gráfico
gradientes=[] ## Armazena os gradientes durante o treinamento para plotar o gráfico





# Função que treina a IA (aqui é onde ocorre o aprendizado) 
#
# Rede: vetor com os pesos da rede
# Entradas: vetor com os dados de entrada
# Saidas_corretas: vetor com as saídas corretas para cada entrada
def treina(rede, entradas, saidas_corretas):

    # Para cada época, calcula a saída do neurônio e atualiza os pesos
    for i in range(EPOCAS):

        # Calcula a saída do neurônio para cada entrada
        saidas_IA = entradas*rede[0]+rede[1] 

        # Calcula o erro quadrático médio (que é a função de perda - loss - escolhida)
        erro=0.5*np.sum((saidas_IA - saidas_corretas)**2)

        # Atualize o peso e o viés do neurônio (aqui é a chave do aprendizado)
        # Usando a descida de gradiente
        gradiente_peso=np.sum((saidas_IA-saidas_corretas)*entradas)
        gradiente_vies=np.sum((saidas_IA-saidas_corretas))
        rede[0]=rede[0]-TAXA_APRENDIZAGEM*gradiente_peso
        rede[1]=rede[1]-TAXA_APRENDIZAGEM*gradiente_vies

        # Armazena os valores para plotar o gráfico
        erros.append(erro)
        pesos.append(rede[0])
        vieses.append(rede[1])
        gradientes.append(gradiente_peso)

        # Imprime os valores para acompanhar o treinamento
        print('Época:',i,' Peso:',rede,'Erro:',erro, 'Gradiente',gradiente_peso, 'Gradiente do viés',gradiente_vies) 


    print('------------------------------------')
    print('Treinamento finalizado')
    print('------------------------------------')

    # Retorna a saída final da rede após o treinamento
    return saidas_IA





# Função que usa a IA já treinada
def testa(rede, entrada):

    # Calcule a saída da rede para uma única entrada
    saida=entrada*rede[0]+rede[1]

    # Retorna a saída e um valor True ou False a partir do limiar de decisão
    return saida




# Cria a rede com um neurônio artificial com uma única entrada
# Por isso, tem apenas um peso
rede = np.array(PESOS_INICIAIS)   



# Define os dados de treinamento para ensinar a rede a identificar números maiores que 0.8) 
entradas = np.array([[0.3],[0.88],[0.2],[0.18],[0.66],[0.70],[0.33],[0.4],[0.20],[0.49],[0.6]])
saidas = 2*entradas+1 # Define a saída como uma função linear dos dados de entrada



# Treina a rede 
saida_IA = treina(rede, entradas, saidas)



# Cria um conjunto de teste com valores que não estavam no conjunto de treinamento
testes=[[0.15],[0.001],[0.1015],[0.95],[0.314],[0.43],[0.81],[0.232],[0.24],[0.1],[0.90]]



# Imprima a saída do neurônio para os dados de teste
for teste in testes:
    entrada=teste[0]
    saida=testa(rede, entrada)
    saida_correta=2*entrada+1
    erro=saida_correta-saida
    print('Entrada:',entrada,'Saída:',saida,'Saída correta:',saida_correta,'Erro:',erro)
    




# Cria 3 subplots lado a lado para mostrar os gráficos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))


# O primeiro gráfico mostra os valores anotados do conjunto de treinamento (ground truth) 
# mostrados com quadrados azuis
# e os valores preditos mostrados com x vermelhos
ax1.scatter(entradas,saidas)
ax1.scatter(entradas,saida_IA,marker='x',color='red')
ax1.legend(['Saída correta','Saída predita'])
ax1.set_xlabel('Entradas')
ax1.set_ylabel('Saídas')
ax1.set_title('Dados de treinamento')

# O segundo mostra a função de perde em relação ao peso e ao viés
ax2.plot(pesos,erros)
ax2.scatter(pesos,erros,marker='x',color='red')
ax2.quiver(pesos,erros,pesos,gradientes,color='green')
ax2.set_xlabel('Pesos')
ax2.set_ylabel('Erros')
ax2.set_title('Função de Perda (ou Erro)')

# O terceiro mostra o erro em função das épocas
ax3.plot(erros)
ax3.set_xlabel('Épocas')
ax3.set_ylabel('Perda')
ax3.set_title('Histórico da aprendizagem')
plt.show()
