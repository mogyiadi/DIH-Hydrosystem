from ultralytics import YOLO

train_model = False
plant_image = 'plant_example.jpeg'

if train_model:
    model_a = YOLO('yolo26n.pt')

    print('Starting training...')

    results = model_a.train(
        data='Potted-plants.5.v2i.yolo26/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',
    )

    print('Training finished.')
else:
    # Fine tuned model
    model_a = YOLO("dih_model_a_results/runs/train2/weights/best.pt")
    results = model_a.predict(plant_image)
    results[0].show()

    # Original Model
    model_a = YOLO('yolo26n.pt')
    results = model_a.predict(plant_image, classes=[58])
    results[0].show()



