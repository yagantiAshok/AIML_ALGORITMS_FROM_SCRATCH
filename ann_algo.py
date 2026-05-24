


import numpy as np
import pandas as pd
import math

def relu(z_value, derivative=None,default = False):


    if default:

        if derivative>0:

            return 1
        
        return 0

    if (z_value<=0):

        return 0
    
    return z_value


def sigmoid(z_value, derivative=None,default=False):

    if default:

        return round(derivative*(1-derivative),5)
    
    return round(1/(1+np.exp(-z_value)),5)




def ann_from_scratch(path):

    
    data = pd.read_csv(path)

    data.drop("Sample code number",axis=1,inplace = True)

    data["Class"].replace({2:0,4:1})

    df = data.to_numpy()

    # i already know columns shape
    # i want to take 2 nuerons and one output neuron 

    w1,w2,w3,w4,w5,w6,w7,w8,w9 = 0,0,0,0,0,0,0,0,0

    b1,b2,b3 = 1,1,1

    w10,w11,w12,w13,w14,w15,w16,w17,w18 = 0,0,0,0,0,0,0,0,0

    w19 = 0
    w20 = 0
    

    
    target_1_loss = 0
    target_0_loss = 0

    for i in range(len(df)):

        x1,x2,x3,x4,x5,x6,x7,x8,x9 = df[i][:9].tolist()

        # 9 input layers 

        # 1 neuron

        z1 = x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + x9*w9 + b1

        # now i taken relu function 

        a1 = relu(z_value=z1,derivative=None)

        z2 = x1*w10 + x2*w11 + x3*w12 + x4*w13 + x5*w14 + x6*w15 + x7*w16 + x8*w17 + x9*w18 + b2

        a2 = relu(z_value=z2,derivative = None)

        z3 = a1*w19 + a2*w20 + b3

        y_hat = sigmoid(z_value=z3,derivative=None)

        if df[i][10]==1:

            traget_1_loss = target_1_loss + (- math.log(y_hat))
        
        else:
                        
            traget_0_loss = target_0_loss + (- math.log(1 - y_hat))

        
    total_error = (target_1_loss + target_0_loss)/len(df)

    return total_error

print("something")













