## InfoTheo
Python implementations of Information Bottleneck with Side Information (Chechik & Tishby), + basic information theory functionalities.

Information Bottleneck with Side Information is an extension of Information Bottleneck method (Tishby, Pereira, & Bialek). <br>
Given a random variable $X$ and target variables $Y_1$ & $Y_2$, with known joint distributions $P(X,Y_1)$ & $P(X,Y_2)$,
Information Bottleneck with Side Information aims to map $X$ to a compressed varibale $T$, while retaining 
information about $Y_1$, and minimizing information about $Y_2$. Formally, minimizing the cost:
```math
L = I(X;T) - \beta_1 I(T;Y_1) + \beta_2 I(T;Y_2)
```
Or, generally, for N target variables:
```math
L = I(X;T) - \sum_i^N \beta_i I(T;Y_i)
```
Where $\beta_i$ are scalars determining the compression/information tradeoff. <br>
Implemented here are the Iterative (soft clustering), and the Agglomerative (hard, hierarchical clustering) variants.

## Usage:
Main functions are aib (agglomerative IB) and iib (iterative IB) in ib.py. <br>
See /tests/test_ib.py for test examples.

## Papers 
- Tishby, Naftali, Fernando C. Pereira, and William Bialek. "The information bottleneck method." arXiv preprint physics/0004057 (2000).
- Slonim, Noam, and Naftali Tishby. "Agglomerative information bottleneck." Advances in neural information processing systems 12 (1999).
- Chechik, Gal, and Naftali Tishby. "Extracting relevant structures with side information." Advances in Neural Information Processing Systems 15 (2002).

