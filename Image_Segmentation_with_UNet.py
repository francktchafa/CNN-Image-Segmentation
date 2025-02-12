"""
Image segmentation using UNet

Step-by-step Outline:
1. Load, split, and process the dataset
2. Build UNet model and specify the loss function
3. Train the model
4. Show predictions and save the trained model
"""

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import os
import imageio.v2 as imageio


# LOAD AND PROCESS THE DATASET
# Load images and masks
image_path = os.path.join(os.getcwd(), "images", "CameraRGB")
mask_path = os.path.join(os.getcwd(), "images", "CameraMask")

# List of files in the directory
image_list_orig = os.listdir(image_path)
mask_list_orig = os.listdir(mask_path)

# Filter files to include only those with expected extensions
# valid_extensions = (".png", ".jpg", ".jpeg")
valid_extensions = (".png")

image_paths_list = [os.path.join(image_path, img) for img in image_list_orig if img.endswith(valid_extensions)]
mask_paths_list = [os.path.join(mask_path, msk) for msk in mask_list_orig if msk.endswith(valid_extensions)]
print(f"Total images: {len(image_paths_list)}, Total masks: {len(mask_paths_list)}")

# Check some images and corresponding masks
N = 1
img = imageio.imread(image_paths_list[N])
mask = imageio.imread(mask_paths_list[N])

fig, ax = plt.subplots(1, 2, figsize=(9, 5))
ax[0].imshow(img)
ax[0].set_title('Image')
ax[1].imshow(mask[:, :, 0])
ax[1].set_title('Segmentation')
plt.show()

# Create a TensorFlow dataset object from a list of file paths
dataset = tf.data.Dataset.from_tensor_slices((image_paths_list, mask_paths_list))  # No shuffling: order is maintained

for start_ind, (image, mask) in enumerate(dataset.take(3)):
    print(f"Image{start_ind}: {image}")
    print(f"Mask{start_ind}: {mask}")


def process_path(img_path, msk_path):
    """
    Reads and processes the image and mask files.

    Arguments:
        img_path -- Path to the image file.
        msk_path -- Path to the mask file.

    Returns:
        img -- Processed image tensor.
        msk -- Processed mask tensor.
    """
    # Read and decode the image file
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Scales to [0, 1]

    # Read and decode the mask file
    msk = tf.io.read_file(msk_path)
    msk = tf.image.decode_png(msk, channels=3)
    # Reduce the mask along the channel axis
    msk = tf.math.reduce_max(msk, axis=-1, keepdims=True)

    return img, msk


def preprocess(img, msk, new_size=(96, 128)):
    """
    Resizes the image and mask to the target size.

    Arguments:
        img -- Input image tensor.
        msk -- Input mask tensor.
        new_size -- New size of the target image (default = (96, 128))

    Returns:
        input_image -- Resized image tensor.
        input_mask -- Resized mask tensor.
    """
    # Resize the image to new_size using nearest neighbor interpolation
    resized_image = tf.image.resize(img, size=new_size, method='nearest')
    # Resize the mask to new_size using nearest neighbor interpolation
    resized_mask = tf.image.resize(msk, size=new_size, method='nearest')

    return resized_image, resized_mask


image_ds = dataset.map(process_path)
resized_image_ds = image_ds.map(preprocess)

# BUILD THE UNET MODEL
model_path = os.path.join(os.getcwd(), "images", "ModelArchitecture", "unet.png")
encoder_path = os.path.join(os.getcwd(), "images", "ModelArchitecture", "encoder.png")
decoder_path = os.path.join(os.getcwd(), "images", "ModelArchitecture", "decoder.png")

# Show the model architecture
model_image = imageio.imread(model_path)
plt.imshow(model_image)
plt.axis('off')
plt.tight_layout()
plt.show()

# First let's focus on building the encoder block
plt.imshow(imageio.imread(encoder_path))
plt.axis('off')
plt.tight_layout()
plt.show()


# Build the encoder block
def encoder_block(inputs=None, n_filters=32, dropout_prob=0., max_pooling=True):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers (default = 32)
        dropout_prob -- Dropout probability (default = 0)
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume (default = True)

    Returns:
        next_layer -- Next layer will go into the next block.
        skip_connection -- Skip_connection will go into the corresponding decoding block
    """

    # First convolutional layer
    conv = Conv2D(filters=n_filters,  # Number of filters
                  kernel_size=3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)

    # Second convolutional layer with same settings
    conv = Conv2D(filters=n_filters,  # Number of filters
                  kernel_size=3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    # Add a dropout layer if dropout_prob > 0
    if dropout_prob > 0:
        conv = Dropout(rate=dropout_prob)(conv)

    # Add a MaxPooling layer if max_pooling is True
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # Set the skip connection
    skip_connection = conv

    return next_layer, skip_connection


# Build a decoder block
plt.imshow(plt.imread(decoder_path))
plt.axis('off')
plt.tight_layout()
plt.show()


# Build the decoder block
def decoder_block(previous_layer_input, skip_connection_input, n_filters=32):
    """
    Convolutional upsampling block

    Arguments:
        previous_layer_input -- Input tensor from the previous layer
        skip_connection_input -- Input tensor from the previous skip layer
        n_filters -- Number of filters for the convolutional layers

    Returns:
        conv -- Tensor output
    """

    # Transposed convolution (upsampling) layer
    up = Conv2DTranspose(filters=n_filters,  # Number of filters
                         kernel_size=3,  # Kernel size
                         strides=(2, 2),
                         padding='same')(previous_layer_input)

    # Merge the previous output and the skip connection
    merge = concatenate([up, skip_connection_input], axis=3)  # Concatenate along the channels axis

    # First convolutional layer after concatenation
    conv = Conv2D(filters=n_filters,  # Number of filters
                  kernel_size=3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)

    # Second convolutional layer
    conv = Conv2D(filters=n_filters,  # Number of filters
                  kernel_size=3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv


# Build the unet model
def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    U-Net model for semantic segmentation

    Arguments:
        input_size -- Input shape
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes

    Returns:
        model -- tf.keras.Model
    """
    inputs = Input(input_size)

    # Contracting Path (i.e., encoding)
    eblock1 = encoder_block(inputs, n_filters)
    eblock2 = encoder_block(eblock1[0], n_filters * 2)
    eblock3 = encoder_block(eblock2[0], n_filters * 4)
    eblock4 = encoder_block(eblock3[0], n_filters * 8, dropout_prob=0.3)
    eblock5 = encoder_block(eblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    # Expanding Path (i.e., decoding)
    dblock6 = decoder_block(eblock5[0], eblock4[1], n_filters * 8)
    dblock7 = decoder_block(dblock6, eblock3[1], n_filters * 4)
    dblock8 = decoder_block(dblock7, eblock2[1], n_filters * 2)
    dblock9 = decoder_block(dblock8, eblock1[1], n_filters)

    # Final convolution
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dblock9)

    # Output layer
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


img_height, img_width, num_channels = 96, 128, 3
unet = unet_model((img_height, img_width, num_channels), n_filters=32, n_classes=23)
unet.summary()


# TRAIN THE MODEL
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
# Define the loss
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # For pixel-wise multiclass pred.
             metrics=['accuracy'])

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = resized_image_ds.cache().shuffle(buffer_size=500).batch(batch_size=32).prefetch(buffer_size=AUTOTUNE)

# Check the element spec
print("Total images: ", resized_image_ds.cardinality().numpy())
print("Element spec: ", resized_image_ds.element_spec)

num_epochs = 40
unet_history = unet.fit(train_dataset, epochs=num_epochs)


# Plots
def history_plots(model_history):
    """
    Plots the training loss and accuracy from the model history.

    Arguments:
        model_history -- A History object returned from model.fit()
    """
    # Extract loss and accuracy
    loss = model_history.history['loss']
    accuracy = model_history.history['accuracy']

    # Extract validation loss and accuracy if they exist
    val_loss = model_history.history.get('val_loss')
    val_accuracy = model_history.history.get('val_accuracy')

    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    if val_accuracy:
        plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot history
history_plots(unet_history)


# SHOW PREDICTIONS AND SAVE TRAINED MODEL
def display(display_list):
    """
    Display a list of images side by side.

    Arguments:
        display_list -- List of image tensors to be displayed
    """
    # Titles for the various subplots
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    # Create subplots
    fig, axes = plt.subplots(1, len(display_list), figsize=(15, 5))

    # Loop through the display list and plot each image
    for i, ax in enumerate(axes):
        ax.set_title(title[i])
        # Convert the tensor to an image and display it
        ax.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        ax.axis('off')  # Turn off the axis for a cleaner look
    plt.tight_layout()
    plt.show()


def create_mask(pred_mask):
    # Use argmax to get the index of the class with the highest probability
    pred_mask = tf.argmax(pred_mask, axis=-1, )

    # Add a new axis to maintain the shape for further processing
    pred_mask = pred_mask[..., tf.newaxis]

    # Return the first image in the batch
    return pred_mask[0]


def show_predictions(data=None, num=1, model=unet, sample_image=None, sample_mask=None):
    """
    Displays predictions for the first image of each of the 'num' batches.

    Arguments:
        data -- tf.data.Dataset containing image and mask pairs.
        num -- Number of batches to display.
        model -- Trained model for making predictions.
        sample_image -- A sample image to use for predictions if no dataset is provided.
        sample_mask -- The corresponding mask for the sample image.
    """
    if data:
        for img, msk in data.take(num):
            pred_mask = model.predict(img)
            display([img[0], msk[0], create_mask(pred_mask)])
    else:
        if sample_image is not None and sample_mask is not None:
            pred_mask = model.predict(sample_image[tf.newaxis, ...])
            display([sample_image, sample_mask, create_mask(pred_mask)])
        else:
            print("No dataset or sample image provided.")


# Show the predictions for the first image of the first 5 batches
show_predictions(train_dataset, 5)

# Save model and weight
model_json = unet.to_json()  # Convert model to JSON

# Specify the file path
model_path = os.path.join(os.getcwd(), "my-unet", "unet.json")
with open(model_path, 'w') as json_file:
    json_file.write(model_json)

weight_path = os.path.join(os.getcwd(), "my-unet", "unet.h5")
unet.save_weights(weight_path)
