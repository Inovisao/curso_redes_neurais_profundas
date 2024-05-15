# Código que reproduz o exemplo utilizado no
# vídeo "Auto Atenção com Múltiplas Cabeças - Explicando as Contas"

import torch

# Hiperparâmetros
tamanho_contexto = 4
tamanho_embedding = 6
n_cabecas = 2
tamanho_cabeca = tamanho_embedding // n_cabecas




##################################################  
# Criação dos tensores de peso e entrada (x)
##################################################

# Cria o tensor de entrada x, com
# dimensões 4x6, que representa
# a sequência com as 4 palavras 
# "Uma Pilha de Pilhas"
# usando embeddings de tamanho 6
x = torch.tensor(
    [[-3,  2,  4,  4, -4, -4],
     [ 4,  3, -1, -5,  3,  2],
     [ 1, -1, -1, -1, -3, -4],
     [ 5,  5,  0, -4,  -1,  2]],dtype=torch.float32)


# Cria o tensor com os parâmetros
# da camada de atenção para Q.
# Inicializa com 1
wQ = torch.ones(6, 6, dtype=torch.float32)




# Cria o tensor com os parâmetros
# da camada de atenção para K
# Inicializa com uma sequência de 1 a 36 
wK = torch.arange(1, 37, dtype=torch.float32).reshape(6, 6)


# Cria o tensor com os parâmetros
# da camada de atenção para V.
# Usando valores aleatórios (gerados
# previamente para reprodução dos resultados
# para reprodução dos resultados do video
# no youtube)
wV = torch.tensor(
    [[ 3, -4,  4,  4, -2, -2],
     [-2, -3,  3, -2,  0,  3],
     [-1, -3,  0,  3, -3, -1],
     [-2,  4,  0, -2,  3, -4],
     [ 1, -3, -1,  4, -3, -1],
     [ 4, -1, -5,  2,  0,  0]],dtype=torch.float32)


print('\n\n------------------------------------------')
print('Entrada e Pesos (ou Parâmetros)')
print('------------------------------------------')

print("x:")
print(x)
print("Shape de x:", x.shape)

print("\nwQ:")
print(wQ)
print("Shape de wQ:", wQ.shape)

print("\nwK:")
print(wK)
print("Shape de wK:", wK.shape)

print("\nwV:")
print(wV)
print("Shape de wV:", wV.shape)






##################################################
# Cálculo de Q, K e V
##################################################

# Calcula Q, usando o produto escalar entre x e wQ
Q = torch.matmul(x, wQ)

# Calcula K
K = torch.matmul(x, wK)

# Calcula V
V = torch.matmul(x, wV)

print('\n\n------------------------------------------')
print('Q, K e V')
print('------------------------------------------')

print("\nQ:")
print(Q)
print("Shape de Q:", Q.shape)

print("\nK:")
print(K)
print("Shape de K:", K.shape)

print("\nV:")
print(V)
print("Shape de V:", V.shape)







##################################################
# Divisão de Q, K e V em cabeças (duas)
################################################## 

# Tamanho do embedding
tamanho_embedding = 6

# Número de cabeças
n_cabecas = 2

# Tamanho da cabeça
tamanho_cabeca = tamanho_embedding // n_cabecas

# Separa a primeira cabeça
Qh1 = Q[:, :tamanho_cabeca]
Kh1 = K[:, :tamanho_cabeca]
Vh1 = V[:, :tamanho_cabeca]

# Separa a segunda cabeça
Qh2 = Q[:, tamanho_cabeca:]
Kh2 = K[:, tamanho_cabeca:]
Vh2 = V[:, tamanho_cabeca:]

print('\n\n------------------------------------------')
print('Q, K e V divididos em DUAS cabeças')
print('------------------------------------------')

print("\nQh1:")
print(Qh1)
print("Shape de Qh1:", Qh1.shape)

print("\nKh1:")
print(Kh1)
print("Shape de Kh1:", Kh1.shape)

print("\nVh1:")
print(Vh1)
print("Shape de Vh1:", Vh1.shape)

print("\nQh2:")
print(Qh2)
print("Shape de Qh2:", Qh2.shape)

print("\nKh2:")
print(Kh2)
print("Shape de Kh2:", Kh2.shape)

print("\nVh2:")
print(Vh2)
print("Shape de Vh2:", Vh2.shape)






##################################################
# Produto Escalar entre Q e K para cada cabeça 
##################################################

# Calcula a atenção para a primeira cabeça
QKh1 = torch.matmul(Qh1, Kh1.T)

# Calcula a atenção para a segunda cabeça
QKh2 = torch.matmul(Qh2, Kh2.T)

# Normaliza a atenção usando a raiz quadrada do tamanho da cabeça
QKh1N = QKh1 / (tamanho_cabeca ** 0.5)
QKh2N = QKh2 / (tamanho_cabeca ** 0.5)

print('\n\n------------------------------------------')
print('Produto Escalar entre Q e K para cada cabeça')
print('------------------------------------------')

print("\nQKh1:")
print(QKh1)
print("Shape de QKh1:", QKh1.shape)

print("\nQKh2:")
print(QKh2)
print("Shape de QKh2:", QKh2.shape)


print('\n\n------------------------------------------')
print('Produto Escalar Normalizado entre Q e K para cada cabeça')
print('------------------------------------------')

print("\nQKh1N:")
print(QKh1N)
print("Shape de QKh1N:", QKh1N.shape)

print("\nQKh2N:")
print(QKh2N)
print("Shape de QKh2N:", QKh2N.shape)









##################################################
# Softmax e Pesos de Atenção
##################################################

# Calcula o softmax para a primeira cabeça
softmaxH1 = torch.nn.functional.softmax(QKh1N, dim=1)

# Calcula o softmax para a segunda cabeça
softmaxH2 = torch.nn.functional.softmax(QKh2N, dim=1)

print('\n\n------------------------------------------')
print('Softmax para cada cabeça')
print('------------------------------------------')

print("\nSoftmaxH1:")
print(softmaxH1)
print("Shape de softmaxH1:", softmaxH1.shape)

print("\nSoftmaxH2:")
print(softmaxH2)
print("Shape de softmaxH2:", softmaxH2.shape)






##################################################
# Cálculo da Saída Final
##################################################

# Multiplicação do resultado do software pelo V 
yH1 = torch.matmul(softmaxH1, Vh1)
yH2 = torch.matmul(softmaxH2, Vh2)

print('\n\n------------------------------------------')
print('Saída Final para cada cabeça')
print('------------------------------------------')

print("\nyH1:")
print(yH1)
print("Shape de yH1:", yH1.shape)

print("\nyH2:")
print(yH2)
print("Shape de yH2:", yH2.shape)






##################################################
# Concatenação das Saídas
##################################################

# Concatena as saídas das duas cabeças
y = torch.cat((yH1, yH2), dim=1)

print('\n\n------------------------------------------')
print('Saída Final Concatenada')
print('------------------------------------------')

print("\ny:")
print(y)
print("Shape de y:", y.shape)



