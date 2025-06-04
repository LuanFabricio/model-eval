# Qual é a função desse projeto?
- Implementar um benchmark para modelos de reconhecimento facial utilizado modelos extraídos para o TensorFlowLite, de forma que seja fácil e prático para avaliar o seu tempo de inferência e acurácia.

# Dependências
- `python` (>=3.9)
- `pip` (>=23.0)
- `make`

# Criando ambiente
1. Crie um ambiente virtual do python (`venv`)
```sh
$ python -m venv venv
```
2. Defina ambiente virtual como ambiente padrão do terminal
```sh
$ source venv/bin/activate
```
3. Instale as dependências pelo `pip`
```bash
$ pip install -r requirements.txt
```
- Obs1: Caso tenha algum problema com a instalação das dependências pelo pip, você pode executar o programa e instalando as dependências de acordo com os erros gerado pela ausência das bibliotecas.
    - As principais depenências são:
        - [tensorflow](https://pypi.org/project/tensorflow/)
        - [scikit-learn](https://pypi.org/project/scikit-learn/)
        - [matplotlib](https://pypi.org/project/matplotlib/)
        - [opencv](https://pypi.org/project/opencv-python/)
- Obs2: No Raspberry, você pode ter problemas na hora de instalar essas dependências. Em alguns casos, será melhor instalar as bibliotecas pelo gerenciador de pacotes.

4. Execute o benchmark
    - Usando o `make`
        ```sh
        $ make
        ```
        - Obs: Caso você esteja no Raspberry, utilize o `Makefile.raspberry` no lugar do `Makefile`. Pois não foi possível carregar o modelo `MobileNetV3Small`.
    - Chamado um modelo em específico
    ```sh
    $ python main models/<nome do modelo>.tflite
    # ou
    $ python main models/<nome do modelo>.tflite > <caminho>/<para>/<log>

    ```

- Observação: Talvez seja necessário ajustar o caminho das imagens de entrada, no projeto.

# Referências
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector?hl=pt-br
