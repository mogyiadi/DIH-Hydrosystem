import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

with open('class_names.txt') as f:
    class_names = [l.strip() for l in f.readlines()]

# model_a = YOLO('yolo26n.pt')
model_a_finetuned = YOLO("dih_model_a_results/runs/train2/weights/best.pt")

interpreter = Interpreter('model_b.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def get_plant_type(cropped_image):
    img = cropped_image.convert('RGB').resize((224, 224))
    input = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    output_data = np.exp(output_data - output_data.max())
    output_data /= output_data.sum()
    idx = int(np.argmax(output_data))

    return class_names[idx], float(output_data[idx])

image_path = 'plant_example_2.png'
# results_a = model_a.predict(image_path, classes=[58])
results_a = model_a_finetuned.predict(image_path)
full_image = Image.open(image_path)

for result in results_a:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print('No plants detected.')
        continue

    print(f"Found {len(boxes)} plant(s).")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence_a = float(box.conf[0])

        cropped_image = full_image.crop((x1, y1, x2, y2))
        plant_class, confidence_b = get_plant_type(cropped_image)
        print(f"  Plant {i + 1}:")
        print(f"    Bounding box : ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"    Plant class  : {plant_class} ({confidence_b * 100:.1f}%)")
