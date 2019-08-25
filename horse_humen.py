from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.optimizers import RMSprop

import os
import zipfile

local_zip = 'tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/horse-or-human')
zip_ref.close()

exit()

# Directory with out training horse pictures
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horse images', len(os.listdir(train_horse_dir)))
print('total traing human images', len(os.listdir(train_human_dir)))


nrows = 4
ncols = 4

# index for iterating over images
pic_index = 0

# set up matplotlib fig and size it to fit 4x4 pics

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index - 8:pic_index]]

next_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    # set up subplots; subplot indice start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # layer one
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # layer two
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Thired layer
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # fourth layer
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # fifth layer
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flattern layer
    tf.keras.layers.Flatten(),
    # 512 nuron hiddn layer
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # layer with sigmoid activation function
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),

])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001),
             metrics=['acc'])

# rescalling all images  to 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                    '/tmp/horse-or-human/',
                    target_size=(300, 300),
                    batch_size=128,
                    class_mode='binary'
                    )

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1
)