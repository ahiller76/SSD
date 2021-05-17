### Aaron Hiller, Kitt Sloan
import cv2
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

# Resize Image
img_height = 300
img_width = 300

# Clear previous models from memory.
K.clear_session()

# Creates new model.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# Load the trained weights into the model.
weights_path = 'VGG_VOC0712_SSD_300x300_iter_120000.h5'

model.load_weights(weights_path, by_name=True)

# Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

capture = cv2.VideoCapture(0)
frameRate = capture.get(cv2.CAP_PROP_FPS)
has_frame = 1
while True:
    has_frame, imageFeed = capture.read()
    imageResize = cv2.resize(imageFeed, (img_height, img_width))

    y_pred = model.predict(np.expand_dims(imageResize, 0))

    confidence_threshold = 0.5

    y_pred_dec = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    plt.figure(figsize=(20,12))

    current_axis = plt.gca()
    image = imageFeed

    for box in y_pred_dec[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * image.shape[1] / img_width
        ymin = box[3] * image.shape[0] / img_height
        xmax = box[4] * image.shape[1] / img_width
        ymax = box[5] * image.shape[0] / img_height
        color = np.dot(colors[int(box[0])],255)
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)
        image = cv2.rectangle(image,(int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        boxed_text = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmin)+text_size[0][0], int(ymin) - 25), color, -1)
        text = cv2.putText(image,label,(int(xmin),int(ymin) - 5),cv2.FONT_HERSHEY_COMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
    cv2.imshow("image", image)
    k = cv2.waitKey(int((1/frameRate)*1000))
    if k == 27:
        print("Pressed esc")
        break
cv2.destroyAllWindows()