1. Nornalization.ipynb  notebook file to run all the 3 models for 20 epochs each

2. model.py file has Net class takes an argument to decide which normalization to include(BN/LN/GN)

3. Nornalization.ipynb use below command to initialize model

        a.import model
        b. from model import Net
        c. modelBN = Net('bn').to(device)  -- to create the Model having Batch Normalization
        d. model_LN =  Net('ln').to(device) --to create the Model having Layer Normalization
        e. model_GN =  Net('gn').to(device) -- to create the Model having Group Normalization
  
  
  
![image](https://user-images.githubusercontent.com/11747515/215314396-b86b7d05-0a2d-4536-8a9a-bec0a12096d6.png)

<details>
  <summary>Batch Normalization Epochs Details</summary>
        
        Epoch 0 : 
        Train set: Average loss: 0.1282, Accuracy: 92.89

        Test set: Average loss: 0.084, Accuracy: 97.76

        Epoch 1 : 
        Train set: Average loss: 0.0912, Accuracy: 97.62

        Test set: Average loss: 0.087, Accuracy: 97.59

        Epoch 2 : 
        Train set: Average loss: 0.0094, Accuracy: 97.94

        Test set: Average loss: 0.041, Accuracy: 98.75

        Epoch 3 : 
        Train set: Average loss: 0.0495, Accuracy: 98.23

        Test set: Average loss: 0.052, Accuracy: 98.57

        Epoch 4 : 
        Train set: Average loss: 0.1233, Accuracy: 98.39

        Test set: Average loss: 0.036, Accuracy: 99.00

        Epoch 5 : 
        Train set: Average loss: 0.0413, Accuracy: 98.48

        Test set: Average loss: 0.034, Accuracy: 99.00

        Epoch 6 : 
        Train set: Average loss: 0.0971, Accuracy: 98.61

        Test set: Average loss: 0.028, Accuracy: 99.17

        Epoch 7 : 
        Train set: Average loss: 0.0113, Accuracy: 98.67

        Test set: Average loss: 0.029, Accuracy: 99.14

        Epoch 8 : 
        Train set: Average loss: 0.0577, Accuracy: 98.75

        Test set: Average loss: 0.027, Accuracy: 99.29

        Epoch 9 : 
        Train set: Average loss: 0.0115, Accuracy: 98.77

        Test set: Average loss: 0.028, Accuracy: 99.22

        Epoch 10 : 
        Train set: Average loss: 0.0504, Accuracy: 98.78

        Test set: Average loss: 0.027, Accuracy: 99.26

        Epoch 11 : 
        Train set: Average loss: 0.0645, Accuracy: 98.93

        Test set: Average loss: 0.026, Accuracy: 99.22

        Epoch 12 : 
        Train set: Average loss: 0.1004, Accuracy: 98.94

        Test set: Average loss: 0.022, Accuracy: 99.36

        Epoch 13 : 
        Train set: Average loss: 0.0253, Accuracy: 98.95

        Test set: Average loss: 0.023, Accuracy: 99.29

        Epoch 14 : 
        Train set: Average loss: 0.0384, Accuracy: 98.94

        Test set: Average loss: 0.023, Accuracy: 99.35

        Epoch 15 : 
        Train set: Average loss: 0.0423, Accuracy: 99.07

        Test set: Average loss: 0.020, Accuracy: 99.37

        Epoch 16 : 
        Train set: Average loss: 0.0028, Accuracy: 99.05

        Test set: Average loss: 0.024, Accuracy: 99.36

        Epoch 17 : 
        Train set: Average loss: 0.0480, Accuracy: 99.03

        Test set: Average loss: 0.023, Accuracy: 99.36

        Epoch 18 : 
        Train set: Average loss: 0.0542, Accuracy: 99.09

        Test set: Average loss: 0.020, Accuracy: 99.46

        Epoch 19 : 
        Train set: Average loss: 0.0453, Accuracy: 99.02

        Test set: Average loss: 0.022, Accuracy: 99.32  

</details>

<details>
  <summary>Layer Normalization Epochs Details</summary>
  
        Epoch 0 : 
        Train set: Average loss: 2.2793, Accuracy: 10.98

        Test set: Average loss: 2.296, Accuracy: 10.28

        Epoch 1 : 
        Train set: Average loss: 0.1399, Accuracy: 61.21

        Test set: Average loss: 0.264, Accuracy: 92.14

        Epoch 2 : 
        Train set: Average loss: 0.0423, Accuracy: 94.71

        Test set: Average loss: 0.094, Accuracy: 97.48

        Epoch 3 : 
        Train set: Average loss: 0.0194, Accuracy: 96.73

        Test set: Average loss: 0.066, Accuracy: 98.16

        Epoch 4 : 
        Train set: Average loss: 0.0259, Accuracy: 97.34

        Test set: Average loss: 0.057, Accuracy: 98.29

        Epoch 5 : 
        Train set: Average loss: 0.2529, Accuracy: 97.76

        Test set: Average loss: 0.054, Accuracy: 98.26

        Epoch 6 : 
        Train set: Average loss: 0.0525, Accuracy: 97.91

        Test set: Average loss: 0.060, Accuracy: 98.18

        Epoch 7 : 
        Train set: Average loss: 0.0265, Accuracy: 98.03

        Test set: Average loss: 0.042, Accuracy: 98.62

        Epoch 8 : 
        Train set: Average loss: 0.0071, Accuracy: 98.28

        Test set: Average loss: 0.040, Accuracy: 98.83

        Epoch 9 : 
        Train set: Average loss: 0.0181, Accuracy: 98.36

        Test set: Average loss: 0.037, Accuracy: 98.82

        Epoch 10 : 
        Train set: Average loss: 0.1982, Accuracy: 98.44

        Test set: Average loss: 0.040, Accuracy: 98.81

        Epoch 11 : 
        Train set: Average loss: 0.0356, Accuracy: 98.44

        Test set: Average loss: 0.036, Accuracy: 98.96

        Epoch 12 : 
        Train set: Average loss: 0.0108, Accuracy: 98.62

        Test set: Average loss: 0.037, Accuracy: 98.87

        Epoch 13 : 
        Train set: Average loss: 0.0038, Accuracy: 98.67

        Test set: Average loss: 0.031, Accuracy: 99.05

        Epoch 14 : 
        Train set: Average loss: 0.1602, Accuracy: 98.73

        Test set: Average loss: 0.032, Accuracy: 99.01

        Epoch 15 : 
        Train set: Average loss: 0.0063, Accuracy: 98.70

        Test set: Average loss: 0.036, Accuracy: 98.93

        Epoch 16 : 
        Train set: Average loss: 0.1476, Accuracy: 98.77

        Test set: Average loss: 0.031, Accuracy: 98.99

        Epoch 17 : 
        Train set: Average loss: 0.0157, Accuracy: 98.82

        Test set: Average loss: 0.034, Accuracy: 98.99

        Epoch 18 : 
        Train set: Average loss: 0.0500, Accuracy: 98.74

        Test set: Average loss: 0.028, Accuracy: 99.17

        Epoch 19 : 
        Train set: Average loss: 0.0031, Accuracy: 98.89
        
        Test set: Average loss: 0.031, Accuracy: 99.05


</details>

<details>
  <summary>Group Normalization Epochs Details</summary>
        
        Epoch 0 : 
        Train set: Average loss: 0.1335, Accuracy: 81.96

        Test set: Average loss: 0.113, Accuracy: 97.20

        Epoch 1 : 
        Train set: Average loss: 0.0774, Accuracy: 96.08

        Test set: Average loss: 0.081, Accuracy: 97.77

        Epoch 2 : 
        Train set: Average loss: 0.1647, Accuracy: 97.31

        Test set: Average loss: 0.073, Accuracy: 97.88

        Epoch 3 : 
        Train set: Average loss: 0.1985, Accuracy: 97.60

        Test set: Average loss: 0.057, Accuracy: 98.47

        Epoch 4 : 
        Train set: Average loss: 0.1771, Accuracy: 97.92

        Test set: Average loss: 0.047, Accuracy: 98.61

        Epoch 5 : 
        Train set: Average loss: 0.0722, Accuracy: 98.11

        Test set: Average loss: 0.044, Accuracy: 98.81

        Epoch 6 : 
        Train set: Average loss: 0.0534, Accuracy: 98.26

        Test set: Average loss: 0.046, Accuracy: 98.75

        Epoch 7 : 
        Train set: Average loss: 0.0094, Accuracy: 98.46

        Test set: Average loss: 0.046, Accuracy: 98.75

        Epoch 8 : 
        Train set: Average loss: 0.0074, Accuracy: 98.45

        Test set: Average loss: 0.044, Accuracy: 98.68

        Epoch 9 : 
        Train set: Average loss: 0.1583, Accuracy: 98.57

        Test set: Average loss: 0.053, Accuracy: 98.33

        Epoch 10 : 
        Train set: Average loss: 0.1386, Accuracy: 98.61

        Test set: Average loss: 0.040, Accuracy: 98.85

        Epoch 11 : 
        Train set: Average loss: 0.0684, Accuracy: 98.71

        Test set: Average loss: 0.043, Accuracy: 98.71

        Epoch 12 : 
        Train set: Average loss: 0.0147, Accuracy: 98.72

        Test set: Average loss: 0.044, Accuracy: 98.64

        Epoch 13 : 
        Train set: Average loss: 0.0063, Accuracy: 98.71

        Test set: Average loss: 0.031, Accuracy: 99.09

        Epoch 14 : 
        Train set: Average loss: 0.0433, Accuracy: 98.91

        Test set: Average loss: 0.034, Accuracy: 98.99

        Epoch 15 : 
        Train set: Average loss: 0.1065, Accuracy: 98.94

        Test set: Average loss: 0.036, Accuracy: 98.94

        Epoch 16 : 
        Train set: Average loss: 0.0166, Accuracy: 98.85

        Test set: Average loss: 0.032, Accuracy: 99.07

        Epoch 17 : 
        Train set: Average loss: 0.0071, Accuracy: 98.98

        Test set: Average loss: 0.028, Accuracy: 99.10

        Epoch 18 : 
        Train set: Average loss: 0.0333, Accuracy: 98.95

        Test set: Average loss: 0.029, Accuracy: 99.18

        Epoch 19 : 
        Train set: Average loss: 0.0242, Accuracy: 98.90

        Test set: Average loss: 0.032, Accuracy: 99.01

 
</details>

 ***Findings***

 ***Batch Normaliazation Model Test Accuracy > Group Normalization Model Test Accuracy > Layer Normalization Model Test Accuracy***
 
 ***Batch Normaliazation Model Training Accuracy > Group Normalization Model Training Accuracy > Layer Normalization Model Training Accuracy***


![image](https://user-images.githubusercontent.com/11747515/215314563-aa4fdeeb-545e-4bf2-aac4-886c83673c09.png)

![image](https://user-images.githubusercontent.com/11747515/215314634-3279fe6d-df6a-4ba0-ab7d-18052c7428c8.png)

![image](https://user-images.githubusercontent.com/11747515/215314649-4c062022-b615-429d-a2c4-04abc8b24e30.png)

![image](https://user-images.githubusercontent.com/11747515/215314662-5bcea0ec-3950-46a8-916d-f1ecad030e9d.png)


