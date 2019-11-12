import tensorflow as tf
import numpy as np
from sys import maxsize
from PIL import Image
import matplotlib.pyplot as plt
from math import floor


def plot_image(params):
    # BEGIN MAIN PROGRAM
    im = Image.open('cat_01_vignetted.jpg')
    img = np.array(im)
    r_matrix = get_r_matrix(img)
    img_applied = vignetting_rgb(img, r_matrix, params)

    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    # Show the tinted image
    plt.subplot(1, 2, 2)

    # A slight gotcha with imshow is that it might give strange results
    # if presented with data that is not uint8. To work around this, we
    # explicitly cast the image to uint8 before displaying it.
    plt.imshow(np.uint8(img_applied))
    plt.show()


def vignetting(i, r, params):
    deg = 0
    s = 0
    for p in params:
        s += (p * (r ** deg))
        deg += 1
    return i * s


def vignetting_rgb(i, r, params):
    deg = 0
    s = 0
    for p in params:
        s += (p * (r ** deg))
        deg += 1

    img_out = np.zeros(i.shape, np.float32)
    img_out[:, :, 0] = s * i[:, :, 0]
    img_out[:, :, 1] = s * i[:, :, 1]
    img_out[:, :, 2] = s * i[:, :, 2]
    return img_out


def data_loader(fold=0, shift=0):
    kitten_vig = []
    kitten = []
    kitten_vig.append(np.array(Image.open("cat_01_vignetted.jpg")))
    kitten_vig.append(np.array(Image.open("cat_02_vignetted.jpg")))
    kitten_vig.append(np.array(Image.open("cat_03_vignetted.jpg")))
    kitten.append(np.array(Image.open("cat_01.jpg")))
    kitten.append(np.array(Image.open("cat_02.jpg")))
    kitten.append(np.array(Image.open("cat_03.jpg")))

    r_flat = list(map(lambda x: np.repeat(get_r_matrix(x)[:, :, np.newaxis], 3, axis=2).flatten(), kitten_vig))
    kitten_vig_flat = list(map(lambda x: x.flatten(), kitten_vig))
    kitten_flat = list(map(lambda x: x.flatten(), kitten))

    r_per_pixel = np.array(r_flat).flatten()
    kitten_vig_per_pixel = np.array(kitten_vig_flat).flatten()
    kitten_per_pixel = np.array(kitten_flat).flatten()

    if fold == 0:
        return {X: kitten_vig_per_pixel, Y: r_per_pixel, Z: kitten_per_pixel}, \
               {X: [], Y: [], Z: []}, \
               kitten_vig_per_pixel.shape[0]

    total_elements = r_per_pixel.shape
    training_elements = floor(total_elements / fold)
    shift_base = shift * training_elements
    shift_target = (shift+1) * training_elements

    if shift == (fold - 1):
        shift_target = training_elements - 1

    r_train = r_per_pixel[0:shift_base] + r_per_pixel[min(shift_target + 1, training_elements - 1):training_elements - 1]
    vig_train = kitten_vig_per_pixel[0:shift_base] + r_per_pixel[min(shift_target + 1, training_elements - 1):training_elements - 1]
    kitten_train = kitten_per_pixel[0:shift_base] + r_per_pixel[min(shift_target + 1, training_elements - 1):training_elements - 1]

    r_vali = r_per_pixel[shift_base:shift_target]
    vig_vali = kitten_vig_per_pixel[shift_base:shift_target]
    kitten_vali = kitten_per_pixel[shift_base:shift_target]

    return {X: vig_train, Y: r_train, Z: kitten_train},\
           {X: vig_vali, Y: r_vali, Z: kitten_vali}, vig_train.shape[0]
    # return {X: kitten_vig_flat, Y: r_flat, Z: kitten_flat}


def get_r_matrix(img):
    img_width = img.shape[1]
    img_height = img.shape[0]
    width_center = img_width / 2
    height_center = img_height / 2

    xv, yv = np.meshgrid(np.arange(img_width) - width_center, np.arange(img_height) - height_center)
    return np.sqrt(xv ** 2 + yv ** 2) / np.sqrt(width_center ** 2 + height_center ** 2)


def generate_parameters(num=7):
    params = []
    for i in range(0, num):
        if i == 0:
            params.append(tf.Variable([0.0], shape=[1], dtype=tf.float32, name='a{0}'.format(str(i))))
        else:
            params.append(tf.Variable([-0.1], shape=[1], dtype=tf.float32, name='a{0}'.format(str(i))))
        #params.append(tf.get_variable(shape=[1], dtype=tf.float32, name='a{0}'.format(str(i))))
    return params


X = tf.placeholder(tf.float32, None)
Y = tf.placeholder(tf.float32, None)
Z = tf.placeholder(tf.float32, None)

# 4 works well with 0.03 convergence
parameters = generate_parameters(3)

sess = tf.InteractiveSession()
gvi = tf.global_variables_initializer()
gvi.run(session=sess)

intensity = vignetting(X, Y, parameters)

feed_dict_train, feed_dict_vali, data_amount = data_loader()
result = sess.run(parameters, feed_dict=feed_dict_train)
print(result)

optimizer = tf.train.GradientDescentOptimizer(1e-6)
loss = tf.losses.mean_squared_error(X, intensity)
minimizer_step = optimizer.minimize(loss)

epoch = 0
iter = 0
previous_loss = sess.run(loss, feed_dict=feed_dict_train)
convergence = 0.04

while True:
    print('Epoch: {0}'.format(epoch))
    print('Step: {0} / {1}'.format(iter, data_amount))
    iter += 1
    if (iter % data_amount) == 0:
        epoch += 1

    pars, step, current_loss = sess.run([parameters, minimizer_step, loss], feed_dict=feed_dict_train)
    print('Parameters: {0}'.format(str(pars)))
    print('Current loss: {0}'.format(current_loss))
    if previous_loss > current_loss:
        difference = previous_loss - current_loss
        print('Current difference: {0} / {1}'.format(difference, convergence))
        if difference < convergence:
            break
    previous_loss = current_loss

eval = sess.run(parameters)
result = []
for e in eval:
    result.append(eval[0].item())

plot_image(result)

print('done')
