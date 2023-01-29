1. Nornalization.ipynb  notebook file to run all the 3 models for 20 epochs each

2. model.py file has Net class takes an argument to decide which normalization to include(BN/LN/GN)

3. Nornalization.ipynb use below command to initialize model

        a.import model
        b. from model import Net
        c. modelBN = Net('bn').to(device) to create the Model having Batch Normalization
  
  
  
![image](https://user-images.githubusercontent.com/11747515/215314396-b86b7d05-0a2d-4536-8a9a-bec0a12096d6.png)



![image](https://user-images.githubusercontent.com/11747515/215314563-aa4fdeeb-545e-4bf2-aac4-886c83673c09.png)

![image](https://user-images.githubusercontent.com/11747515/215314634-3279fe6d-df6a-4ba0-ab7d-18052c7428c8.png)

![image](https://user-images.githubusercontent.com/11747515/215314649-4c062022-b615-429d-a2c4-04abc8b24e30.png)

![image](https://user-images.githubusercontent.com/11747515/215314662-5bcea0ec-3950-46a8-916d-f1ecad030e9d.png)


