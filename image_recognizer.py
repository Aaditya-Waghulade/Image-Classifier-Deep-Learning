import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

def main():
    st.title('Image Classifier using Convolutional Neural Network') #Title Of Our Website
    st.write('Upload any image among the classes you want to know that my predciton are true and Accuracy of my model')#Normal text
    st.write('You can put following pictures to recognize: Airplane , Automobile , Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck')
#1. Here by using 'fileuplader' You can upload a file by deciding there types such as jpg and other you can also upload pdfs if in any case
    file =  st.file_uploader("Upload Image here.....",type=['jpg','jpeg','png','webp']) 
    st.write("developed by aaditya.....")
    if file: #if we have the file the do operations
         image =Image.open(file)# declaring the path of the file which you have uploaded
         st.image(image) #Setting that path we declared in 'st.image'

#2. Now building and Putting our uploaded image in the model for predicting.
# 2.1 resizing uploaded image into 32,32
         resized_image = image.resize((32,32))
         img_array = np.array(resized_image)/255 #because we have the value scaled by dividing it with 255. *we convert the image into array as because our model only accepts the array.*
         img_array = img_array.reshape((1,32,32,3)) # 1 image with size 32x32x3
#2.2 Implimenting Predictions
         model = tf.keras.models.load_model('cifar10.h5') #Loading the model from file
         predictions = model.predict(img_array)#providing an array which is converted version of Image
         cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# 2.3 Ploting Graphs to showing how accurately my model is predicting the Image from the given class
         fig, ax = plt.subplots()
         y_pos = np.arange(len(cifar10_classes))
         ax.barh(y_pos,predictions[0],align='center')
         ax.set_yticks(y_pos)
         ax.set_yticklabels(cifar10_classes)
         ax.invert_yaxis()
         ax.set_xlabel("Probability")
         ax.set_title("Prediction")

         st.pyplot(fig)
         st.write('Accuracy of my model is : ',np.max(predictions[0])*100,'%')
        
    else:
        #if image not uploaded then show this text
        st.text("Not Uploaded Image yet... ")

if __name__=='__main__':
    main()


