# Early Forest Fire Detection with CNN Models

## Commands
Run in colab found in .ipynb files
  

# YOLOv4
## Method
Trained for a total of 50 epochs
Pretrained on Image Net

<details>
  <summary>Major Changes</summary>
  
  ## Adjusting Hyperparameters
  - Scale: ±50%
  - Hue: ±1.5%
  - Saturation: ±50%
  - Exposure: ±50%
  - Angle: ±45º
    
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

<details>
  <summary>Results</summary>
  
  ## Results: mAP @0.5, AP@0.5
  
  |                          |    mAP    |  Fire AP  |
  | ---                      |     ---   |   ---     |
  | Baseline (No Changes)    |     |     |
  | Adjusted Hyperparameters |   N/A     |           | 
  | Adjusted Hyperparameters |   N/A     |           | 

</details>

# YOLOv5

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

<details>
  <summary>Results</summary>
  
  ## Results: mAP @0.5, AP@0.5
  
  |                          |    mAP    |  Fire AP  |
  | ---                      |     ---   |   ---     |
  | Baseline (No Changes)    |     |     |
  | Adjusted Hyperparameters |   N/A     |           | 
  | Adjusted Hyperparameters |   N/A     |           | 

</details>


# Faster-RCNN

# Citation
