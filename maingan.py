

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from gan import GAN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Parameters
image_dir = r'C:/Users/youss/anime_faces/images'
noise_dim = 100
batch_size = 64
epochs = 50

# 2. Preprocessing Images

def preprocess(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, (64, 64))
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image




images = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]

# Afficher les 6 premières images
images[:6]

training_dataset = tf.data.Dataset.from_tensor_slices((images)) #Si images contient les chemins d'accès : alors le Dataset résultant contiendra seulement les chemins des fichiers

print(len(training_dataset))
training_dataset = training_dataset.map(preprocess)
training_dataset = training_dataset.shuffle(1000).batch(batch_size)



# visualize some of them
fig, axes = plt.subplots(5,5, figsize = (14,14))
sample = training_dataset.unbatch().take(25)
sample = [image for image in sample]

idx = 0
for row in range(5):
    for column in range(5):
        axes[row, column].imshow(sample[idx])#imshow peut détecter que les valeurs des pixels sont entre 0 et 1 (au lieu de 0 et 255) et ajuste l'affichage pour montrer les images correctement
        idx+=1



sample_image = preprocess(images[0])
print(sample_image.shape)





gan = GAN(noise_dim)


#@tf.function

 
# Ensure the output directory exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Your existing train_step and train functions remain the same

def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = gan.generator(noise, training=True)
        real_output = gan.discriminator(images, training=True)
        fake_output = gan.discriminator(generated_images, training=True)
        disc_loss = gan.discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, gan.discriminator.trainable_variables)
    gan.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, gan.discriminator.trainable_variables))

    # Train generator
    with tf.GradientTape() as gen_tape:
        generated_images = gan.generator(noise, training=True)
        fake_output = gan.discriminator(generated_images, training=True)
        gen_loss = gan.generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gan.generator.trainable_variables)
    gan.generator_optimizer.apply_gradients(zip(gradients_of_generator, gan.generator.trainable_variables))

    return disc_loss, gen_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            disc_loss, gen_loss = train_step(image_batch)

        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}')

        # Generate and save sample images
        generate_and_save_images(gan.generator, epoch + 1, tf.random.normal([16, noise_dim]))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')

    # Save the image to the output directory
    image_path = os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(image_path)
    plt.show()


# Call the training function
train(training_dataset, epochs)

train(training_dataset, epochs)

noise = tf.random.normal([1, noise_dim])
generated_image = gan.generator(noise, training=False)
print(generated_image.shape)  # Should be (1, 64, 64, 3)


noise = tf.random.normal([1, noise_dim])
generated_image = gan.generator(noise, training=False)
gan.generator.summary()




















