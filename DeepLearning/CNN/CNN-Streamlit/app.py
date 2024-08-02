# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import streamlit as st
from pathlib import Path



def performCNN():
    # Part 1 - Data Preprocessing

    # Preprocessing the Training set
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
    training_set = train_datagen.flow_from_directory('dataset',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('face_test',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    # Part 2 - Building the CNN

    # Initialising the CNN
    cnn = tf.keras.models.Sequential()

    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Part 3 - Training the CNN

    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training the CNN on the Training set and evaluating it on the Test set
    cnn.fit(x = training_set, validation_data = test_set, epochs = 20)

    cnn.save("ganesh-cnn.h5")

    return cnn

# Part 4 - Making a single prediction
my_file = Path("ganesh-cnn.h5")
if my_file.exists():
    st.session_state["CNN"] = load_model(my_file)
else:
    if 'CNN' not in st.session_state:
        st.session_state["CNN"] = performCNN()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    import numpy as np
    import keras.utils as image
        
    cnn = st.session_state["CNN"]
    copy_image = image.load_img(uploaded_file, target_size = (480, 320))
    test_image = image.load_img(uploaded_file, target_size = (64, 64))
   
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
   
    if result[0][0] == 1:
        prediction = 'Male'
    else:
        prediction = 'Female'
    print(prediction)
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Uploaded image")
        st.image(copy_image)

    with col2:
        
        st.header("Prediction")
        for i in range(12):
            st.write(" ")
        
        html_str = f"""
                        <style>
                        p.a {{
                        font: bold 35px Courier;color:Red;
                        }}
                        </style>
                        <p class="a">{prediction}</p>
                        """
        st.markdown(html_str, unsafe_allow_html=True)