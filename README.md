# variational_decoder

This architecture was heavily inspired by the Variational Neural Recurrent Auto-Encoder (VNRAE) architecture presented by Tong, Li, and Yen in https://www.cs.cmu.edu/~epxing/Class/10708-17/project-reports/project12.pdf.

We substitute the pre-trained sentences embeddings for the inner-LSTM segment of their architecture, then follow the same variational approach to decode the embedded sentence back into text. The system currently supports only a bag-of-words (based on FastText embeddings) sentence embedding. Planned expansions include skipthoughts and Universal Sentence Encoder lite and large.
