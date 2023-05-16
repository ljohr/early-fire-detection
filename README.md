# Early Forest Fire Detection with CNN Models

## Commands
To train the YOLOv4 model
```
!./darknet detector train cfg/coco.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map
```

To train the YOLOv5 model
```

!python train.py --img 416 --batch 16 --epochs 100  --data /content/drive/MyDrive/Yolov5_FireDetection/Yolo_smoke/data.yaml --weights yolov5m.pt --cache --name result_5m_weight
```

To train the YOLOv8 model
```
model.train(data=os.path.join(ROOT_DIR, "data.yaml"), epochs=50, degrees=45, scale=0.5, hsv_h=0.015, hsv_v=0.4, hsv_s=0.7, shear=0.5, mosaic=1.0)
```  

To train the Faster-RCNN model
```
!python train.py --data data_configs/custom_data.yaml -e 50 -m fasterrcnn_resnet50_fpn -n custom_training3 -b 16 
```  

# YOLOv4
## Method
Trained for a total of 50 epochs
Pretrained on COCO Dataset

<details>
  <summary>Major Changes</summary>
  
  ## Adjusting Hyperparameters
  - Scale: ±50%
  - Hue: ±1.5%
  - Saturation: ±50%
  - Exposure: ±50%
  - Angle: ±45º
  - Learning Rate: 0.001 then 0.0001
    
</details>

<details>
  <summary>Other Changes</summary>
  
  ## Other Changes
  **cfg/yolov4-custom.cfg**
  - Changed batch size to 64
  - Changed subdivisions to 8
  - Changed filter sizes to fit custom class size of 2
  
  **data/coco.names**
  - Changed to customize fire and smoke classes
  
  **cfg/coco.data**
  - Changed to set train, test, and validation folder paths

</details>

# YOLOv5

## Method
Trained

<details>
  <summary>Major Changes</summary>
  
  ## Major Changes
  **changes**
  - Changed 
    
</details>

<details>
  <summary>Other Changes</summary>
  
  ## Other Changes
  **changes**
  - Changed 

</details>

# YOLOv8
## Method
Trained for a total of 50 epochs
Pretrained on Image Net

<details>
  <summary>Major Changes</summary>
  
  ## Change Learning Rate Scheduler
  **ultralytics/yolo/engine/trainer.py**
  - Changed to from LambdaLR to CosineAnnealingLR 
    - Negligible impact overall, but reduced the number of background images predicted as smoke by 10% and is likely to have a bigger impact with a larger dataset. 
  
  ## Freeze Backbone
  - Included for loop in the get\_model function found in the yolo/v8/segment/train.py file
  
</details>

<details>
  <summary>Other Changes</summary>
  
  ## Other Changes
  **cfg/yolov4-custom.cfg**
  - Changed batch size to 64
  - Changed subdivisions to 8
  - Changed filter sizes to fit custom class size of 2
  
  **data/coco.names**
  - Changed to customize fire and smoke classes
  
  **cfg/coco.data**
  - Changed to set train, test, and validation folder paths

</details>


# Faster-RCNN

## Method
Trained

<details>
  <summary>Major Changes</summary>
  
  ## Major Changes
  **changes**
  - Changed 
    
</details>

<details>
  <summary>Other Changes</summary>
  
  ## Other Changes
  **changes**
  - Changed 

</details>


# Contribution
```
Sara wrote
```

```
Danni wrote
```

# Dataset
Our custom dataset was too large to push to GitHub. [Link to Download]()

# Citation
D-Fire Dataset: Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa: An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. In: Neural Computing and Applications, 2022. [Link to GitHub Repository](https://github.com/gaiasd/DFireDataset)

Open Wildfire Smoke Datasets: AI for Mankind [Link to GitHub Repository](https://github.com/aiformankind/wildfire-smoke-dataset)

Cloud Dataset: AI for Mankind [Link to GitHub Repository](https://github.com/aiformankind/wildfire-smoke-dataset)
