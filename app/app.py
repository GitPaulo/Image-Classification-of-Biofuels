import cv2 as cv
import PySimpleGUI as sg
import tensorflow as tf
import os

# Make directory for pictures
if not os.path.exists('pictures'):
    os.makedirs('pictures')

# Globals
PICTURES_ROOT = './pictures/'
pic_counter = 1
kr = tf.keras
vgg16 = kr.applications.vgg16
model = kr.models.load_model("./trained_model")

# Add a touch of color
sg.change_look_and_feel('Reds')	

# Define gui layout
gui_layout = [  
    [sg.Text('Place object in front of camera and select classification model.')],
    [sg.InputCombo(('VGG16', 'Resnet'), size=(20, 3))],
    [sg.Button('Classify'), sg.Button('Exit')] 
]

# Launch layout
window = sg.Window('ICB - Quick Picture Classifier', gui_layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Exit'):	# if user closes window or clicks cancel
        break
    if event in (None, 'Classify'):
        # initialize the camera
        cam = cv.VideoCapture(0)   # 0 -> index of camera
        s, img = cam.read()

        if s: # frame captured without any errors
            # initial name of file
            pic_name = PICTURES_ROOT + str(pic_counter) + ".jpg"
            
            # OpenCV s
            cv.imshow("Picture Taken", img)
            cv.waitKey(0)
            cv.destroyWindow("Picture Taken")
            cv.imwrite(pic_name, img)

            # get combo_value
            combo_value = values[0]

            if combo_value == "VGG16":
                # load an image from file
                image = kr.preprocessing.image.load_img(pic_name, target_size=(224, 224))
                # convert the image pixels to a numpy array
                image = kr.preprocessing.image.img_to_array(image)
                # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = vgg16.preprocess_input(image)
                # predict the probability across all output classes
                yhat = model.predict(image).flatten()
                # display prediction data
                sg.Popup("[Prediction Data]\n biofuel: " + str(yhat[0]) + "\n non_biofuel: " + str(yhat[1]))
                # rename based on result
                old_pic_name = pic_name
                pic_name = PICTURES_ROOT + (("biofuel_" + str(pic_counter)) if yhat[0] >= 0.5 else ("non-biofuel_" + str(pic_counter))) + ".jpg" 
            elif combo_value == "Resnet":
                sg.Popup("Resnet not implemented yet!")
            else:
                sg.Popup("No model selected!")

            # finally save image
            os.rename(old_pic_name, pic_name)
            pic_counter = pic_counter + 1
        else:
            sg.Popup("Problem with camera frame :(")

# Exit!
window.close()
