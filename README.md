# Reconhecimento Facial com OpenCV e Deepface

Este projeto implementa um sistema de reconhecimento facial utilizando OpenCV e o reconhecedor. O sistema é capaz de detectar e reconhecer faces em tempo real a partir de um feed de vídeo.

## Funcionalidades

- **Detecção de Faces:** Utiliza o classificador Haar para detectar faces em frames de vídeo.
- **Reconhecimento Facial:** Utiliza a biblioteca DeepFace para reconhecer faces previamente treinadas.
- **Treinamento do Modelo:** Permite treinar o modelo com um dataset de imagens faciais.

## Como Usar

1. **Treinamento:**
   - Use o script `treinamento.py` para treinar o modelo com suas imagens faciais.
   - As imagens devem estar organizadas em diretórios por pessoa.

2. **Reconhecimento:**
   - Execute o script `reconhece.py` para iniciar o reconhecimento facial em tempo real.
   - O sistema exibirá retângulos ao redor das faces reconhecidas, juntamente com os nomes e a confiança.

## Como Contribuir

1. Melhore o código 
2. Crie uma branch para sua feature (`git checkout -b minha-feature`).
3. Faça commit das suas alterações (`git commit -am 'Adicionei minha feature'`).
4. Faça push para a branch (`git push origin minha-feature`).
5. Abra um Pull Request.

## Agradecimentos

Agradecimentos a todos os contribuidores deste projeto. Suas sugestões e melhorias são sempre bem-vindas!
