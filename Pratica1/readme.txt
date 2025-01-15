Programa desenvolvido em JAVA com 18 extratores (ANEXO referência "descritores.pdf") 

-- Link para download --> https://drive.google.com/open?id=19gv1W7SPvG7AfqaG0LrVM1ikVUAD3zbJ 
-- Execução 
-- Coloque as imagens em algum diretório, altere o nome para o padrão:
---- Os três primeiros caracteres do nome de cada imagem deve ser referente a classe. 
----------- Example: X01_xxxxxx.jpg (X01 é a classe)
-- Clique em (File - Open Image Directory) e escolha o diretório contendo o conjunto de imagens no formato indicado.
-- Clique em (File - Directory to save) e escolha o diretório onde será salvo os arquivos arff e txt.
-- Escolha o botão do extrator desejado (BIC, CGH, ... Tamura) e clique OK. Aguarde os botões serem ativados para nova extração.

Para execução digitar no terminal "java -jar FeaturesExtractions_Bressan_V1_0_3.jar"

-------------------------------------------------------------------------------------------
Anotações do grupo

Bases de Dados:

	CIFAR100 - 10000 imagens: https://www.kaggle.com/datasets/melikechan/cifar100
	100 Sports Image Classification - 500 imagens: https://www.kaggle.com/datasets/gpiosenka/sports-classification/discussion?sort=hotness
	Cards Image Dataset-Classification - 265 imagens: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

Extratores selecionados:

	F1 - AutoColorCorrelogram [Huang et al. 1997] 	Cor 		768
	F4 - FCTH [Chatzichristofis and Boutalis 2008b] Cor e Textura 	192
	F14 - PHOG [Bosch et al. 2007] 			Textura 	40	