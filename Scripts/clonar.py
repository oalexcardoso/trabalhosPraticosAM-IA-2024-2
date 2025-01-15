import os
import shutil

# Caminho para o diretório CIFAR100
cifar100_path = "SPORTS"
# Caminho da nova pasta chamada "ALL"
nova_pasta = os.path.join(cifar100_path, "ALL")

# Cria a nova pasta "ALL" se não existir
os.makedirs(nova_pasta, exist_ok=True)

# Percorre todas as subpastas dentro do diretório CIFAR100
for root, dirs, files in os.walk(cifar100_path):
    # Ignora a própria pasta "ALL" no processo
    if os.path.basename(root) == "ALL":
        continue

    for arquivo in files:
        # Caminho completo do arquivo atual
        caminho_antigo = os.path.join(root, arquivo)
        # Caminho para onde o arquivo será copiado
        caminho_novo = os.path.join(nova_pasta, arquivo)

        # Verifica se já existe um arquivo com o mesmo nome na pasta "ALL"
        if os.path.exists(caminho_novo):
            # Garante que o arquivo terá um nome único
            base, ext = os.path.splitext(arquivo)
            i = 1
            while os.path.exists(caminho_novo):
                caminho_novo = os.path.join(nova_pasta, f"{base}_{i}{ext}")
                i += 1

        # Copia o arquivo para a pasta "ALL"
        shutil.copy2(caminho_antigo, caminho_novo)
        print(f"Copiado: {caminho_antigo} -> {caminho_novo}")

print("Todos os arquivos foram copiados para a pasta 'ALL'!")
