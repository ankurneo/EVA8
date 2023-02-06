
**Goals** 

1. The code should use GPU if available
2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
3. Total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. use albumentation library and apply:
    a.  horizontal flip
    b. shiftScaleRotate
    c. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
8. Achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.


**Done**
1.  Augmentation applied using albumentation library 

      train_transforms = A.Compose(
          [
              A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,rotate_limit=45, interpolation=1,border_mode=4, p=0.2),
                  A.CoarseDropout(max_holes=2, max_height=8,max_width=8, p=0.1),A.Normalize(mean = (0.491, 0.482, 0.447),std = (0.247, 0.243, 0.262)),ToTensorV2()
          ]
      )

      test_transforms = A.Compose(
          [
              A.Normalize(mean = (0.491, 0.482, 0.447),std = (0.247, 0.243, 0.262)),ToTensorV2()
          ]
      )
 
 2. Parameter used -: **Total params: 183,946 **
 3. Depthwise Separable Convolution is used
 4. Dilated Convolution is Used
 5. GAP is Used
 6. (Epoch 74)
    a. Train set: Average loss: 0.6409, Accuracy: 87.12 
    b. Test set: Average loss: 0.486, Accuracy: 84.66




