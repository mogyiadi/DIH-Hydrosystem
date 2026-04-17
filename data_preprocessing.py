import os
import random
import shutil


base_dir = 'Potted-plants.5.v2i.yolo26'

train_images = os.path.join(base_dir, 'train', 'images')
train_labels = os.path.join(base_dir, 'train', 'labels')

val_images = os.path.join(base_dir, 'valid', 'images')
val_labels = os.path.join(base_dir, 'valid', 'labels')

test_images = os.path.join(base_dir, 'test', 'images')
test_labels = os.path.join(base_dir, 'test', 'labels')

os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

os.makedirs(test_images, exist_ok=True)
os.makedirs(test_labels, exist_ok=True)

all_images = [f for f in os.listdir(train_images) if f.endswith('.jpg')]

val_split_count = int(len(all_images) * 0.2)
test_split_count = int(len(all_images) * 0.1)

print(f'Total images: {len(all_images)}')
print(f'Moving {val_split_count} images to validation set...')
print(f'Moving {test_split_count} images to test set...')

images_to_move = random.sample(all_images, val_split_count)

for image in images_to_move:
    scr_img = os.path.join(train_images, image)
    dst_img = os.path.join(val_images, image)
    shutil.move(scr_img, dst_img)

    label = image.rsplit('.', 1)[0] +'.txt'

    scr_label = os.path.join(train_labels, label)
    dst_label = os.path.join(val_labels, label)
    shutil.move(scr_label, dst_label)

all_images = [f for f in os.listdir(train_images) if f.endswith('.jpg')]
images_to_move = random.sample(all_images, test_split_count)

for image in images_to_move:
    scr_img = os.path.join(train_images, image)
    dst_img = os.path.join(test_images, image)
    shutil.move(scr_img, dst_img)

    label = image.rsplit('.', 1)[0] +'.txt'

    scr_label = os.path.join(train_labels, label)
    dst_label = os.path.join(test_labels, label)
    shutil.move(scr_label, dst_label)