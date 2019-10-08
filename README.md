# ai-residency
#### TESTE FORK AND COMMIT

## Como sincronizar o GitHub e a pasta local

Para criar um arquivo .py diretamente pelo terminal, podemos acessar o comando "nano script.py", na interface "ctrl+o" para salvar e depois "ctrl+x" para sair. 

Para remover um repositório, usar "git remove rm (nome)".

Para sincronizar os arquivos entre o disco local e o repositório na nuvem, acessar o site do GitHub e copiar o endereço do repositório. Exemplo https://github.com/lazaropd/ai-residency

De volta ao terminal, navegar até o diretório onde se deseja salvar os arquivos e executar o comando "git clone (git_repo_link). Isto criará uma pasta com o nome do repositório, contendo todos os arquivos do repositório online.

Em seguida, basta pegar os arquivos locais que se deseja sincronizar neste repositório e mover para esta nova pasta. Voltar ao terminal. Navegar até a pasta e executar na sequência "git add * ", depois "git commit -m 'algum texto detalhado' " e finalmente "git push".

Vai pedir a senha do GitHub e depois sincronizar todos os arquivos para a nuvem. Se desejar, pode executar "git status" em algum momento para verificar os arquivos e pastas que serão sincronizados.

Importante: pastas vazias não são sincronizadas, então caso necessário, colocar um arquivo qualquer em cada pasta antes de executar a rotina acima.

Lucas HUB
Greg wants to build a string, S of length N. Starting with an empty string, he can perform 2 operations:
1. Add a character to the end of S for A dollars.
2. Copy any substring of S, and then add it to the end of S for B dollars.
Calculate minimum amount of money Greg needs to build S.
