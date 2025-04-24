export WANDB_PROJECT="dental-genie-quadrant-enumeration-disease-yolov8m-1000-2"
export WANDB_ENTITY="destrogamer-aston-university"

start_model=yolov8m
training_data=/home/ubuntu/projects/dental-genie/training_data
suffix=train_${start_model}_1000


#yolo task=detect mode=train model=$start_model.pt data=$training_data/quadrant/quadrant.yaml epochs=1000 imgsz=640 name=quadrant_$suffix project="$WANDB_PROJECT"  
#yolo task=detect mode=train model=dental-genie-quadrant-yolov8m-1000/quadrant_train_yolov8m_1000/weights/best.pt data=$training_data/quadrant_enumeration/quadrant_enumeration.yaml epochs=1000 imgsz=640 name=quadrant_enumeration_$suffix project="$WANDB_PROJECT" 
yolo task=detect mode=train model=dental-genie-quadrant-enumeration-yolov8m-1000/quadrant_enumeration_train_yolov8m_1000/weights/best.pt data=$training_data/quadrant-enumeration-disease/quadrant_enumeration_disease.yaml epochs=1000 patience=0 imgsz=640 name=quadrant_enumeration_disease_$suffix project="$WANDB_PROJECT" 
