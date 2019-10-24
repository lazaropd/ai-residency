# Desafio-1-2-005

## Equipe:

* Alessandra Buso
* Gabriela Kimura
* Lazaro Domiciano
* Wana Batista

## Descrição do Projeto

* Estudos na área de ecologia e conservação da biodiversidade são baseados em observações da natureza. Para que tais estudos normalmente é necessário utilizar uma grande quantidade de dados em grandes escala geografica e temporal.

O que é Informática para Biodiversidade: https://figshare.com/articles/Introduction_to_Biodiversity_Informatics/1295382

* Diversos portais reunem dados de observações que são chamados dados de ocorrência. Um dado de ocorrência é definido como uma observação individual de um animal ou planta, para essa observação muitas informações podem ser registradas, tais como, taxonomia da espécie, data, hora, local (lat/log), nome comum, coleção a que pertence, entre outros. Alguns metadados definem centenas de campos que podem ser preenchidos para cada observação.

* Diversos portais, tais como, Gbif, ALA, bson, canadensys, entre outros, reunem informações de dados de ocorrência para uso público.

* Uma área de pesquisa referente a dados de biodiversidade é quanto a qualidade desses dados. (https://www.researchgate.net/publication/264387406_Data_Quality_Control_in_Biodiversity_Informatics_The_Case_of_Species_Occurrence_Data)

* Nessa atividade dados de ocorrência de diferentes portais em formato CSV são fornecidos para que seja feita uma análise descritiva dessa informação, com base na qualidade dos dados. Nesse sentido, crie uma classe python que realize as seguintes funções:

1) Para cada coluna identique a quantidade de linhas com dados faltantes (em alguns casos, o dado faltante é uma string vazia, em outros casos é uma string contendo algum valor do tipo: "sem informação"). Faça um método que retorna a média de dados faltantes por coluna

2) Para cada item identifique até qual nível taxônomico a ocorrência foi identificada.

Seis campos definem a espécie: Filo;Classe;Ordem;Familia;Genero;Especie do mais genérico (Filo) para o mais específico (Especie), é comum que em alguns casos, o preenchimento pare em família ou genêro por exemplo.

3) Monte filtros de ocorrências por estados, nome de espécie (nome exato ou parte do nome) e categoria de ameaça, e outros filtros que julgar relevante.

4) Crie uma funcionalidade para avaliar se a informação de longitude e latitude corresponde a informação presente na localização, para isso você pode utilizar uma biblioteca de consulta reversa de lat/log como por exemplo o https://opencagedata.com/tutorials/geocode-in-python
