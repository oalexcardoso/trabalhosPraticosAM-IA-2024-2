import os

# Caminho para o diretório CIFAR100
cifar100_path = ""

# Lista todas as pastas no diretório
pastas = [pasta for pasta in os.listdir(
    cifar100_path) if os.path.isdir(os.path.join(cifar100_path, pasta))]

# Ordena as pastas para garantir que a sequência seja consistente
pastas.sort()

# Renomeia as pastas e os arquivos dentro delas
for i, pasta in enumerate(pastas, start=1):
    # Substitui espaços por underscores no nome da pasta
    pasta_sem_espacos = pasta.replace(' ', '_')
    novo_nome_pasta = f"X{i:02d}_{pasta_sem_espacos}"
    caminho_antigo_pasta = os.path.join(cifar100_path, pasta)
    caminho_novo_pasta = os.path.join(cifar100_path, novo_nome_pasta)

    # Renomeia a pasta
    os.rename(caminho_antigo_pasta, caminho_novo_pasta)
    print(f"Renomeado: {pasta} -> {novo_nome_pasta}")

    # Renomeia os arquivos dentro da pasta renomeada
    arquivos = os.listdir(caminho_novo_pasta)
    for arquivo in arquivos:
        # Substitui espaços por underscores no nome do arquivo
        arquivo_sem_espacos = arquivo.replace(' ', '_')
        novo_nome_arquivo = f"{novo_nome_pasta}_{arquivo_sem_espacos}"
        caminho_antigo_arquivo = os.path.join(caminho_novo_pasta, arquivo)
        caminho_novo_arquivo = os.path.join(
            caminho_novo_pasta, novo_nome_arquivo)

        os.rename(caminho_antigo_arquivo, caminho_novo_arquivo)
        print(f"    Renomeado: {arquivo} -> {novo_nome_arquivo}")

print("Renomeação concluída!")
