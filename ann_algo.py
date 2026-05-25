
import os 

import numpy as np
import pandas as pd
import math

from logger import logger


def relu_derivative(z_value):

        if z_value>0:

            return 1
        
        return 0 

def sigmoid_derivative(output):
     

        return round(output*(1-output),5)
     


def relu(z_value):


    if (z_value<=0):

        return 0
    
    return z_value


def sigmoid(z_value):

    
    return round(1/(1+np.exp(-z_value)),5)

epsilon = 1e-15 # A very small number


def ann_from_scratch(path):

    
    data = pd.read_csv(path)

    data.drop("sample",axis=1,inplace = True)

    data["Class"]=data["Class"].replace({2:0,4:1})

    data = data.iloc[:20]

    df = data.to_numpy()

    # i already know columns shape
    # i want to take 2 nuerons and one output neuron 

    w1,w2,w3,w4,w5,w6,w7,w8,w9 = 0.1,0.5,0.2,0.7,0.2,0.5,0.4,0.2,0.9

    b1,b2,b3 = 0.4,0,1

    w10,w11,w12,w13,w14,w15,w16,w17,w18 = 0.3,0.1,0.2,0.9,0.3,0.4,0.2,0.1,0

    w19 = 0.6
    w20 = 0.3

    lr = 0.001


    
    target_1_loss = 0
    target_0_loss = 0

    for epoch in range(1):
         
        der_w1 = 0
        der_w2 = 0
        der_w3 = 0
        der_w4=0
        der_w5=0
        der_w6 =0
        der_w7 = 0
        der_w8 =0
        der_w9 =0

        der_b1=0
        der_b2=0
        der_b3 =0

        der_w10 = 0
        der_w11 = 0
        der_w12 = 0
        der_w13=0
        der_w14=0
        der_w15 =0
        der_w16 = 0
        der_w17 =0
        der_w18 =0
        
        der_w19 = 0
        der_w20 =0

        for i in range(len(df)):

            loss_derivective = 0

            x1,x2,x3,x4,x5,x6,x7,x8,x9 = df[i][:9].tolist()

            # 9 input layers 

            # 1 neuron

            z1 = x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + x9*w9 + b1


            # now i taken relu function 

            a1 = relu(z_value=z1)



            z2 = x1*w10 + x2*w11 + x3*w12 + x4*w13 + x5*w14 + x6*w15 + x7*w16 + x8*w17 + x9*w18 + b2



            a2 = relu(z_value=z2)



            z3 = a1*w19 + a2*w20 + b3

            y_hat = sigmoid(z_value=z3)


            if df[i][9]==1:
                
            
                loss_derivective  = - 1 / (y_hat+epsilon)



                target_1_loss += (- math.log(y_hat+epsilon))

            
            else:

                loss_derivective = 1/(1-y_hat+epsilon)

                            
                target_0_loss += (- math.log(1 - y_hat+epsilon))
            

            ## backword propagation for all weights and biases

            logger.info(f"for evry row loss is { i } row {loss_derivective}")

            der_of_loss_and_sigmoid = loss_derivective * sigmoid_derivative(y_hat)

            logger.info(f"for evry row deriavtive of loss and sigmoid {i} row {der_of_loss_and_sigmoid}")


            der_w20 =der_of_loss_and_sigmoid  * a2

            der_w19 = der_of_loss_and_sigmoid * a1

            der_b3 = der_of_loss_and_sigmoid * 1

            der_w18 = der_of_loss_and_sigmoid * w20 * relu_derivative(z2) * x9

            der_w17 = der_of_loss_and_sigmoid * w20 * relu_derivative(z2) * x8 


            # for all weights who involved in z2 affect these is the same  der_of_loss_and_sigmoid * w20 * relu_derivative(z2) 

            same_for_z2_weights =  der_of_loss_and_sigmoid * w20 * relu_derivative(z2)

            der_w16 =  same_for_z2_weights * x7

            der_w15 =  same_for_z2_weights * x6

            der_w14 = same_for_z2_weights * x5

            der_w13 = same_for_z2_weights * x4

            der_w12 = same_for_z2_weights * x3

            der_w11 = same_for_z2_weights * x2

            der_w10 = same_for_z2_weights * x1

            der_b2  = same_for_z2_weights * 1

            # now for 1 neuron 

            der_w9 = der_of_loss_and_sigmoid * w19 * relu_derivative(z1) * x9


            der_w8 = der_of_loss_and_sigmoid * w19 * relu_derivative(z1) * x8

            # for all weights who involved in z1 affect these is the same  der_of_loss_and_sigmoid * w19 * relu_derivative(z1)

            same_for_z1_weights = der_of_loss_and_sigmoid * w19 * relu_derivative(z1)

            der_w7 = same_for_z1_weights * x7

            der_w6 = same_for_z1_weights * x6

            der_w5 = same_for_z1_weights * x5 

            der_w4 = same_for_z1_weights * x4

            der_w3 = same_for_z1_weights * x3 

            der_w2 = same_for_z1_weights * x2

            der_w1 = same_for_z1_weights * x1 

            der_b1 = same_for_z1_weights * 1
        
            der_w1+=der_w1

            der_w2+=der_w2

            der_w3+=der_w3

            der_w4+=der_w4

            der_w5+=der_w5

            der_w6+=der_w6

            der_w7+=der_w7

            der_w8+=der_w8

            der_w9+=der_w9

            der_w10+=der_w10

            der_w11+=der_w11

            der_w12+=der_w12

            der_w13+=der_w13

            der_w14+=der_w14

            der_w15+=der_w15

            der_w16+=der_w16

            der_w17+=der_w17

            der_w18+=der_w18

            der_w19+=der_w19

            der_w20+=der_w20

            der_b1+=der_b1

            der_b2+=der_b2

            der_b3+=der_b3
    
    logger.info(f" for der_w20 {der_w20}")
    logger.info(f"for der_w15 {der_w15}")
    logger.info(f"for der_w10 {der_w10}")

    length = len(df)

    print(length)
    
    w1 = w1 - lr*(der_w1/length)
    w2 = w2 - lr*(der_w2/length)
    w3 = w3 - lr*(der_w3/length)
    w4 = w4 - lr*(der_w4/length)
    w5 = w5 - lr*(der_w5/length)
    w6 = w6 - lr*(der_w6/length)
    w7 = w7 - lr*(der_w7/length)
    w8 = w8 - lr*(der_w8/length)
    w9 = w9 - lr*(der_w9/length)
    w10 = w10 - lr*(der_w10/length)
    w11 = w11 - lr*(der_w11/length)
    w12 = w12 - lr*(der_w12/length)
    w13 = w13 - lr*(der_w13/length)
    w14 = w14 - lr*(der_w14/length)
    w15 = w15 - lr*(der_w15/length)
    w16 = w16 - lr*(der_w16/length)
    w17 = w17 - lr*(der_w17/length)
    w18 = w18 - lr*(der_w18/length)
    w19 = w19 - lr*(der_w19/length)
    w20 = w20 - lr*(der_w20/length)
    b1 = b1 - lr*(der_b1/length)
    b2 = b2 - lr*(der_b2/length)
    b3 = b3 - lr*(der_b3/length)


        
    total_error = (target_1_loss + target_0_loss)/len(df)

    logger.info(f"for epoch {epoch} error {total_error}")

    logger.info(f"weights for {epoch} {w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,b1,b2,b3}")



    return total_error


path = os.path.join("notebooks","Data_cls.csv")

print(ann_from_scratch(path))

print("something")













