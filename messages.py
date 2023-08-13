welcome = \
    """
    # Inteli Exec Módulo 3 - Análise Preditiva
    
    Bem-vindo(a)! Neste site, você poderá simular o treino e avaliação de modelos em diferentes conjuntos de dados.
    """


data_upload_explanation = \
    """
                Para treinarmos um modelo, precisamos de uma base com covariáveis (indicada por `X`) e um alvo de previsão (`y`). 
    
                A tabela de dados deve conter o nome das variáveis (covariáveis e alvo) na primeira linha e os valores nas células abaixo:
    
                | |*A*|*B*|...|*N*|*N+1*|
                |---|---|---|---|---|---|
                |*1*|Variável_1| Variável_2| ... |Variável_N| Alvo|
                |*2*|1.22|3.1|...|4.5| 1000|
                |*3*|3.6|2.8|...|5.0| 2000|
                |...|...|...|...|...|
            """

feature_importance = \
    """Para modelos lineares (regressão linear/logística), a importância é definida como $-\log_{10}(p)$ em 
            que $p$ é o p-valor. Assim, p-valores baixos terão importâncias grandes, e vice versa. \n Já para modelos 
            de árvore (RandomForest), a importância é definida com base na melhora em impureza nos splits. Não há uma
             noção clara de "coeficientes" ou seus p-valores
    """