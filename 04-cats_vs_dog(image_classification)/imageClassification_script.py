{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.7.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"markdown","source":"**COPYING IMAGES TO TRAIN, VALIDATION AND TEST DIRECTORIES**","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19"}},{"cell_type":"code","source":"!unzip -q '/kaggle/input/dogs-vs-cats/train.zip' -d '/kaggle/working'","metadata":{"execution":{"iopub.status.busy":"2023-01-24T09:26:30.784399Z","iopub.execute_input":"2023-01-24T09:26:30.784829Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import os\nimport shutil\nimport zipfile\n\nbase_dir = '/kaggle/input/dogs-vs-cats/cats_and_dogs_small'\n\nwith zipfile.ZipFile(\"/kaggle/input/dogs-vs-cats/train.zip\", 'r') as zip_ref:\n    zip_ref.extractall(\"/kaggle/input/dogs-vs-cats/train_data\")\n\nos.mkdir(base_dir)","metadata":{"execution":{"iopub.status.busy":"2023-01-24T09:24:55.511514Z","iopub.execute_input":"2023-01-24T09:24:55.512735Z","iopub.status.idle":"2023-01-24T09:24:55.900841Z","shell.execute_reply.started":"2023-01-24T09:24:55.512665Z","shell.execute_reply":"2023-01-24T09:24:55.899487Z"},"trusted":true},"execution_count":3,"outputs":[{"traceback":["\u001b[0;31m---------------------------------------------------------------------------\u001b[0m","\u001b[0;31mOSError\u001b[0m                                   Traceback (most recent call last)","\u001b[0;32m/tmp/ipykernel_23/1220688824.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;32mwith\u001b[0m \u001b[0mzipfile\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mZipFile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"/kaggle/input/dogs-vs-cats/train.zip\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'r'\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mzip_ref\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m     \u001b[0mzip_ref\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mextractall\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"/kaggle/input/dogs-vs-cats/train_data\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      9\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmkdir\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbase_dir\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n","\u001b[0;32m/opt/conda/lib/python3.7/zipfile.py\u001b[0m in \u001b[0;36mextractall\u001b[0;34m(self, path, members, pwd)\u001b[0m\n\u001b[1;32m   1634\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1635\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mzipinfo\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mmembers\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1636\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_extract_member\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mzipinfo\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpwd\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1637\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1638\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mclassmethod\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n","\u001b[0;32m/opt/conda/lib/python3.7/zipfile.py\u001b[0m in \u001b[0;36m_extract_member\u001b[0;34m(self, member, targetpath, pwd)\u001b[0m\n\u001b[1;32m   1680\u001b[0m         \u001b[0mupperdirs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdirname\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtargetpath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1681\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mupperdirs\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexists\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mupperdirs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1682\u001b[0;31m             \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmakedirs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mupperdirs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1683\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1684\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmember\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mis_dir\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n","\u001b[0;32m/opt/conda/lib/python3.7/os.py\u001b[0m in \u001b[0;36mmakedirs\u001b[0;34m(name, mode, exist_ok)\u001b[0m\n\u001b[1;32m    221\u001b[0m             \u001b[0;32mreturn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    222\u001b[0m     \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 223\u001b[0;31m         \u001b[0mmkdir\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    224\u001b[0m     \u001b[0;32mexcept\u001b[0m \u001b[0mOSError\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    225\u001b[0m         \u001b[0;31m# Cannot rely on checking for EEXIST, since the operating system\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n","\u001b[0;31mOSError\u001b[0m: [Errno 30] Read-only file system: '/kaggle/input/dogs-vs-cats/train_data'"],"ename":"OSError","evalue":"[Errno 30] Read-only file system: '/kaggle/input/dogs-vs-cats/train_data'","output_type":"error"}]},{"cell_type":"code","source":"# 1 COPYING IMAGES TO TRAIN, VALIDATION AND TEST DIRECTORIES\n\nimport os\nimport shutil\n\noriginal_dataset_dir = '/Users/dip/Downloads/kaggle_original_data'\nbase_dir = '/Users/dip/Downloads/cats_and_dogs_small'\nos.mkdir(base_dir)\n\n#->directory for train, validation and test split\ntrain_dir = os.path.join(base_dir, 'train')\nos.mkdir(train_dir)\nvalidation_dir = os.path.join(base_dir, 'validation')\nos.mkdir(validation_dir)\ntest_dir = os.path.join(base_dir, 'test')\nos.mkdir(test_dir)\n\n#->directory with training cat pitchures\ntrain_cats_dir = os.path.join(train_dir, 'cats')\nos.mkdir(train_cats_dir)\n\n#->directory with training dog pitchures\ntrain_dogs_dir = os.path.join(train_dir, 'dogs')\nos.mkdir(train_dogs_dir)\n\n#->directory with validation cat pitchures\nvalidation_cats_dir = os.path.join(validation_dir, 'cats')\nos.mkdir(validation_cats_dir)\n\n#->directory with validation dog pitchures\nvalidation_dogs_dir = os.path.join(validation_dir, 'dogs')\nos.mkdir(validation_dogs_dir)\n\n#->directory with test cat pitchures\ntest_cats_dir = os.path.join(test_dir, 'cats')\nos.mkdir(test_cats_dir)\n\n#->directory with test dog pitchures\ntest_dogs_dir = os.path.join(test_dir, 'dogs')\nos.mkdir(test_dogs_dir)\n\n#->Copies the first 1,000 cat images to train_cats_dir\nfnames = ['cat.{}.jpg'.format(i) for i in range(1000)]\nfor fname in fnames:\n    src = os.path.join(original_dataset_dir, fname)\n    dst = os.path.join(train_cats_dir, fname)\n    shutil.copyfile(src, dst)\n\n#->Copies the next 500 cat images to validation_cats_dir\nfnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]\nfor fname in fnames:\n    src = os.path.join(original_dataset_dir, fname)\n    dst = os.path.join(validation_cats_dir, fname)\n    shutil.copyfile(src, dst)\n\n#->Copies the next 500 cat images to test_cats_dir    \nfnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]\nfor fname in fnames:\n    src = os.path.join(original_dataset_dir, fname)\n    dst = os.path.join(test_cats_dir, fname)\n    shutil.copyfile(src, dst)\n    \n#->similarly for dogs\nfnames = ['dog.{}.jpg'.format(i) for i in range(1000)]\nfor fname in fnames:\n    src = os.path.join(original_dataset_dir, fname)\n    dst = os.path.join(train_dogs_dir, fname)\n    shutil.copyfile(src, dst)\n\nfnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]\nfor fname in fnames:\n    src = os.path.join(original_dataset_dir, fname)\n    dst = os.path.join(validation_dogs_dir, fname)\n    shutil.copyfile(src, dst)\n\nfnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]\nfor fname in fnames:\n    src = os.path.join(original_dataset_dir, fname)\n    dst = os.path.join(test_dogs_dir, fname)\n    shutil.copyfile(src, dst)\n    \n# 2 BUILDING NETWORK\n\nfrom keras import layers\nfrom keras import models\n\nmodel = models.Sequential()\nmodel.add(layers.Conv2D(32, (3, 3), activation='relu',\ninput_shape=(150, 150, 3)))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Conv2D(128, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Conv2D(128, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Flatten())\nmodel.add(layers.Dense(512, activation='relu'))\nmodel.add(layers.Dense(1, activation='sigmoid'))\n\n# 3 CONFIGURING THE MODEL FOR TRAINING\n\nfrom keras import optimizers\n\nmodel.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])\n\n# 4 DATA PROCESSING\n#Using ImageDataGenerator to read images from directories\n\nfrom keras.preprocessing.image import ImageDataGenerator\n#->resclaes all images by 1/255\ntrain_datagen = ImageDataGenerator(rescale=1./255)\ntest_datagen = ImageDataGenerator(rescale=1./255)\ntrain_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150)  batch_size=20, class_mode='binary')\nvalidation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')\n\n# 5 FITTING THE MODEL USING BATCH GENEROTER\n\nhistory = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator,validation_steps=50)\n\n# saving the model\nmodel.save('cats_and_dogs_small_1.h5')\n\n# 6 Displaying curves of loss and accuracy during training\n\nimport matplotlib.pyplot as plt\nacc = history.history['acc']\nval_acc = history.history['val_acc']\nloss = history.history['loss']\nval_loss = history.history['val_loss']\nepochs = range(1, len(acc) + 1)\nplt.plot(epochs, acc, 'bo', label='Training acc')\nplt.plot(epochs, val_acc, 'b', label='Validation acc')\nplt.title('Training and validation accuracy')\nplt.legend()\nplt.figure()\nplt.plot(epochs, loss, 'bo', label='Training loss')\nplt.plot(epochs, val_loss, 'b', label='Validation loss')\nplt.title('Training and validation loss')\nplt.legend()\nplt.show()\n\n\"\"\"The above plots are characteristic of overfitting. The training accuracy increases linearly over time, until it reaches nearly 100%, whereas the validation accuracy stalls at 70–72%. \nThe validation loss reaches its minimum after only five epochs and then stalls, whereas the training loss keeps decreasing linearly until it reaches nearly 0 \"\"\"\n\n\n# 7 USING DATA AUGMENTATION TO REDUCE OVERFITTING\n\n# Setting up a data augmentation configuration via ImageDataGenerator\n\ndatagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, \n                             horizontal_flip=True, fill_mode='nearest')\n\n#Displaying some randomly augmented training images\n\nfrom keras.preprocessing import image\n\nfnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]\nimg_path = fnames[3]\nimg = image.load_img(img_path, target_size=(150, 150))\nx = image.img_to_array(img)\nx = x.reshape((1,) + x.shape)\n#Generates batches of randomly transformed images. Loops indefinitely,so we need to break the loop at some point\ni = 0\nfor batch in datagen.flow(x, batch_size=1):\n    plt.figure(i)\n    imgplot = plt.imshow(image.array_to_img(batch[0]))\n    i += 1\n    if i % 4 == 0:\n        break\n        \nplt.show()\n\n#Defining a new convnet that includes dropout\n\nmodel = models.Sequential()\nmodel.add(layers.Conv2D(32, (3, 3), activation='relu',\ninput_shape=(150, 150, 3)))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Conv2D(128, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Conv2D(128, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\nmodel.add(layers.Flatten())\nmodel.add(layers.Dropout(0.5))\nmodel.add(layers.Dense(512, activation='relu'))\nmodel.add(layers.Dense(1, activation='sigmoid'))\nmodel.compile(loss='binary_crossentropy',\noptimizer=optimizers.RMSprop(lr=1e-4),\nmetrics=['acc'])\n\n# Training the convnet using data-augmentation generators\n\ntrain_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,\n                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)\n\ntest_datagen = ImageDataGenerator(rescale=1./255)\n\ntrain_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')\n\nvalidation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')\n\nhistory = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)","metadata":{},"execution_count":null,"outputs":[]}]}