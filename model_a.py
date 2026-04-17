from ultralytics import YOLO

train_model = False

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
    results = model_a.predict('plant_example.jpeg')
    results[0].show()

    # Original Model
    # model_a = YOLO('yolo26n.pt')
    # results = model_a.predict('Potted-plants.5.v2i.yolo26/test/images/3a32b4711e3078b308f0c3123c2a0229_jpeg.rf.c92be15c433fab7316f82ffb22a56ec9.jpg', classes=[58])
    # results[0].show()



