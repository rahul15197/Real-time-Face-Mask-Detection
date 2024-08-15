"""
Program to create cross-platform GUI using Streamlit

Perform face detection using caffe pre-trained ResNet10
Perform mask detection using saved MobileNetV2 model

Image and Webcam video option is available in GUI sidebar
"""
# all necessary imports
import base64
import os
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# convert base64 version of bin file, used for local background image
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# background image of mask in GUI
def set_jpg_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/jpg;base64,%s");
    background-size: 250px 250px;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: bottom right;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_jpg_as_page_bg('mask_image.jpg')


# method to detect mask in the uploaded image
def mask_image():
    # global variable for output_image
    global detect_output
    # load prototype and weights for the face detection caffe model
    prototype_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
    caffe_weights = os.path.sep.join(["face_detector",
                                      "res10_300x300_ssd_iter_140000.caffemodel"])
    # create ResNet10 model
    caffe_face_detect = cv2.dnn.readNet(prototype_path, caffe_weights)
    # load the saved MobileNetV2 model
    model = load_model("model_MNV2_full")

    # read the uploaded image which is saved as out.jpg
    image = cv2.imread("out.jpg")
    # create image copy
    orig = image.copy()
    # extract height and weight of image
    (height, weight) = image.shape[:2]

    # construct a blob from the image for face detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    caffe_face_detect.setInput(blob)
    # detect faces using ResNet10
    faces = caffe_face_detect.forward()

    # for all detected faces
    for i in range(0, faces.shape[2]):
        # the probability of model with which the current face is detected
        face_prob = faces[0, 0, i, 2]
        # if prob is above threshold(70%) then  only perform detection of mask
        # to reduce the chance of getting wrong detections that are actually not faces
        if face_prob > 0.5:
            # box to make a frame around detected face
            face_box = faces[0, 0, i, 3:7] * np.array([weight, height, weight, height])
            (startX, startY, endX, endY) = face_box.astype("int")
            # boundary of the box
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(weight - 1, endX), min(height - 1, endY))

            # from the face, extract the face ROI and perform resizing and
            # pre-processing for performing the detection
            face_ROI = image[startY:endY, startX:endX]
            face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_BGR2RGB)
            face_ROI = cv2.resize(face_ROI, (200, 200))
            face_ROI = img_to_array(face_ROI)
            face_ROI = preprocess_input(face_ROI)
            face_ROI = np.expand_dims(face_ROI, axis=0)

            # make predictions for mask detection
            # model returns 2 probabilities (softmax output layer)
            # mask and no mask
            prob_mask, prob_no_mask = model.predict(face_ROI)[0]

            # based on the max out of above prob values
            # label and color of the box is decided
            if prob_mask > prob_no_mask:
                label = "Mask"
                label = f"{label} - {round(prob_mask * 100, 2)} %"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                label = f"{label} - {round(prob_no_mask * 100, 2)} %"
                color = (0, 0, 255)

            # display label and probability text
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
            # display rectangle frame around face
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            # convert BGR image to RGB and save as detect_output
            # OpenCV returns image in BGR sequence
            detect_output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_video():
    prototype_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
    caffe_weights = os.path.sep.join(["face_detector",
                                      "res10_300x300_ssd_iter_140000.caffemodel"])
    # load face detection model
    caffe_face_detect = cv2.dnn.readNet(prototype_path, caffe_weights)
    # load saved MNV2 model
    model = load_model("model_MNV2_full")
    # capture webcam video stream
    vid = cv2.VideoCapture(0)
    # counter for saving frames on key press
    number = 0
    # usage information
    st.subheader("Press 'Space' to display frame")
    st.subheader("Press 's' to save frame")
    st.subheader("Press 'Esc' to exit")
    # while video stream is running
    while True:
        # extract image from video stream
        (rval, image) = vid.read()
        # rest of the process is similar to above mentioned process
        (height, weight) = image.shape[:2]

        # construct a blob
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        caffe_face_detect.setInput(blob)
        # detect faces
        faces = caffe_face_detect.forward()
        # iterate over all faces
        for i in range(0, faces.shape[2]):
            face_prob = faces[0, 0, i, 2]
            # if probability is above threhold, then perform mask detection
            # otherwise capture only the webcam stream
            if face_prob > 0.5:
                face_box = faces[0, 0, i, 3:7] * np.array([weight, height, weight, height])
                (startX, startY, endX, endY) = face_box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(weight - 1, endX), min(height - 1, endY))

                face_ROI = image[startY:endY, startX:endX]
                face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_BGR2RGB)
                face_ROI = cv2.resize(face_ROI, (200, 200))
                face_ROI = img_to_array(face_ROI)
                face_ROI = preprocess_input(face_ROI)
                face_ROI = np.expand_dims(face_ROI, axis=0)

                prob_mask, prob_no_mask = model.predict(face_ROI)[0]

                if prob_mask > prob_no_mask:
                    label = "Mask"
                    label = f"{label} - {round(prob_mask * 100, 2)} %"
                    color = (0, 255, 0)
                else:
                    label = "No Mask"
                    label = f"{label} - {round(prob_no_mask * 100, 2)} %"
                    color = (0, 0, 255)

                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                cv2.imshow("Realtime Webcam (Esc to exit)", image)
            else:
                cv2.imshow("Realtime Webcam (Esc to exit)", image)
        # for key press event capture
        key = cv2.waitKey(1)
        # display captured frame in GUI on Spacebar press
        if key % 256 == 32:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, use_column_width=True)
        # save captured frame on s keypress
        if key % 256 == 115:
            detected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detected_image = Image.fromarray(detected_image)
            detected_image.save(f'detected_image{number}.jpg')
            number += 1
        # exit the webcam stream windows on Esc keypress
        if key == 27:
            break
    # close webcam window
    vid.release()
    cv2.destroyAllWindows()


def detection_display():
    # GUI styling
    st.title("Real time Face and mask on-off detection")
    # sidebar options
    activities = ["Image", "Video", "Performance", "Information"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # selected choice
    selected_choice = st.sidebar.selectbox("Select Option", activities)

    if selected_choice == 'Image':
        st.subheader("Mask Detection on Image")
        # upload image using file uploader
        image_file = st.file_uploader("Upload Image", type=['jpg'])
        if image_file is not None:
            orig_image = Image.open(image_file)
            # save uploaded image as out.jpg which is used during mask_image() call
            orig_image.save('out.jpg')
            # display selected image in browser
            st.image(orig_image, caption='Image uploaded', use_column_width=True)
            # button event
            if st.button('Submit'):
                progress_bar = st.progress(0)
                for percent in range(100):
                    time.sleep(0.000001)
                    progress_bar.progress(percent + 1)
                # perform detection of faces and mask on image
                mask_image()
                # display output_image
                st.image(detect_output, use_column_width=True)
                # convert array to image
                detected_image = Image.fromarray(detect_output, 'RGB')
                # save image locally
                detected_image.save('detected_image.jpg')
                st.balloons()

    if selected_choice == 'Video':
        st.subheader("Real time Mask detection on Video")
        # perform detection of faces and mask on webcam video stream
        mask_video()

    if selected_choice == "Performance":
        # show classification report and plots
        st.subheader("Model classification Report")
        cf_report = cv2.imread("full_class_report.jpg")
        acc_plot = cv2.imread("Figure_3.png")
        loss_plot = cv2.imread("Figure_4.png")
        cf_report = cv2.cvtColor(cf_report, cv2.COLOR_BGR2RGB)
        st.image(cf_report, use_column_width=True)
        st.subheader("Model Accuracy Plot")
        acc_plot = cv2.cvtColor(acc_plot, cv2.COLOR_BGR2RGB)
        st.image(acc_plot, use_column_width=True)
        st.subheader("Model Loss Plot")
        loss_plot = cv2.cvtColor(loss_plot, cv2.COLOR_BGR2RGB)
        st.image(loss_plot, use_column_width=True)

    if selected_choice == "Information":
        # display project name and information
        st.subheader("Detection of faces and masks using ResNet10 and MobileNetV2 on images and real-time video stream")
        st.subheader("Created by Rahul Maheshwari (MT19027)")
        st.text("IIIT Delhi, New Delhi")
        st.markdown("[Github](https://github.com/rahul15197) [Email](mailto:rahul19027@iiitd.ac.in)")


detection_display()
