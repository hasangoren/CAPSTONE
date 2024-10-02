import streamlit as st 
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('CAPSTONE1.h5')

def process_image(img):
    img = img.resize((32, 32))  # Görüntüyü 224x224 boyutuna küçült
    img = np.array(img)  # PIL Image'ı NumPy dizisine çevir
    img = img.astype(np.uint8)  # uint8 formatına dönüştür
    img = img / 255.0  # Normalize et
    img = np.expand_dims(img, axis=0)  # Boyutunu genişlet
    return img

st.title('CAPSTONE TRAFIK ISARETLERI')
st.write('Bir trafik resmi yükle ve hangi çeşit olduğunu öğren')

file = st.file_uploader('Bir Resim Seç', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Yüklenen resim')
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    class_names = ['7','17','19','22','2','35','23','10','5','36','20','27','41','39','32','25','42','8','38','12','0','31','34','18','28','16','13','26','15','3','1','30','14','4','9','21','40','6','11','37','33','29','24']
    st.write(f"Predicted class: {class_names[predicted_class]}")