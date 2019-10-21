##  Experimento Artificial LP 20 realizaçoes,##
### AVERAGE RESULTS ####
                     	 acc  	  Se  	  Es  	  F1  
        0,0         : 	0,9983	1,0000	0,9975	0,9976
        1,0         : 	0,9967	0,9950	0,9975	0,9950
        2,0         : 	0,9983	0,9950	1,0000	0,9974
     All Class      :	0,9978	0,9967	0,9983	0,9967

### STD RESULTS ####
                     	 acc  	  Se  	  Es  	  F1  
        0,0         : 	0,0073	0,0000	0,0109	0,0104
        1,0         : 	0,0100	0,0218	0,0109	0,0151
        2,0         : 	0,0073	0,0218	0,0000	0,0115
     All Class      :	0,0067	0,0100	0,0050	0,0100



##  Experimento Iris LP 20 realizaçoes,##
### AVERAGE RESULTS ####
                             acc  	  Se  	  Es  	  F1  
        Iris-setosa     : 	0,9950	0,9850	1,0000	0,9921
      Iris-versicolor   : 	0,9583	0,9450	0,9650	0,9382
       Iris-virginica   : 	0,9633	0,9450	0,9725	0,9438
         All Class      :	0,9722	0,9583	0,9792	0,9580

### STD RESULTS ####
                             acc  	  Se  	  Es  	  F1  
        Iris-setosa     : 	0,0119	0,0357	0,0000	0,0188
      Iris-versicolor   : 	0,0314	0,0805	0,0502	0,0465
       Iris-virginica   : 	0,0314	0,0921	0,0402	0,0510
         All Class      :	0,0209	0,0314	0,0157	0,0320



##  Experimento COLUNA 3C LP 20 realizaçoes,##
### AVERAGE RESULTS ####
                     	 acc  	  Se  	  Es  	  F1  
         DH         : 	0,8185	0,5583	0,8810	0,4757
         SL         : 	0,9218	0,9217	0,9219	0,9190
         NO         : 	0,7806	0,6400	0,8476	0,6297
     All Class      :	0,8403	0,7067	0,8835	0,6748

### STD RESULTS ####
                     	 acc  	  Se  	  Es  	  F1  
         DH         : 	0,0338	0,3396	0,0928	0,2355
         SL         : 	0,0559	0,0985	0,1103	0,0570
         NO         : 	0,0569	0,2606	0,1547	0,1278
     All Class      :	0,0395	0,0812	0,0343	0,0937



##  Experimento Dermatologia LP 20 realizaçoes,##
### AVERAGE RESULTS ####
                     	 acc  	  Se  	  Es  	  F1  
        1,0         : 	0,9938	0,9787	0,9973	0,9824
        2,0         : 	0,9719	0,9263	0,9830	0,9287
        3,0         : 	0,9966	0,9918	0,9978	0,9921
        4,0         : 	0,9764	0,9177	0,9868	0,9191
        5,0         : 	0,9910	0,9843	0,9926	0,9736
        6,0         : 	0,9961	0,9625	0,9976	0,9607
     All Class      :	0,9876	0,9602	0,9925	0,9594

### STD RESULTS ####
                     	 acc  	  Se  	  Es  	  F1  
        1,0         : 	0,0097	0,0486	0,0070	0,0282
        2,0         : 	0,0157	0,0521	0,0159	0,0403
        3,0         : 	0,0051	0,0205	0,0052	0,0126
        4,0         : 	0,0154	0,0609	0,0110	0,0556
        5,0         : 	0,0084	0,0355	0,0101	0,0253
        6,0         : 	0,0102	0,0931	0,0079	0,1119
     All Class      :	0,0057	0,0213	0,0035	0,0238

##  Experimento Cancer LP 20 realizaçoes,
### AVERAGE RESULTS 
                     	 acc  	  Se  	  Es  	  F1  
        2,0         : 	0,9606	0,9726	0,9377	0,9701
        4,0         : 	0,9606	0,9377	0,9726	0,9414
     All Class      :	0,9606	0,9551	0,9551	0,9557
 
### STD RESULTS 
                     	 acc  	  Se  	  Es  	  F1  
        2,0         : 	0,0232	0,0170	0,0696	0,0168
        4,0         : 	0,0232	0,0696	0,0170	0,0388
     All Class      :	0,0232	0,0337	0,0337	0,0276


# Matrizes confusao melhor treinamento.
Matriz confusao: Artificial

    10  0  0 
    0 10  0 
    0  0 10  

Matriz confusao: Iris 

    10  0  0 
    0 10  0 
    0  0 10  

Matriz confusao: Coluna 3c

    10  0  2 
    2 27  1 
    3  1 16  

Matriz confusao: Dermatologia 

    16  0  0  0  0  0 
    0 20  0  0  0  0 
    0  0 21  0  0  0 
    0  0  0 14  0  0 
    0  0  0  0 12  0 
    0  0  0  0  0  6  
    
Matriz confusao: Cancer 

    96 01
    00 40   