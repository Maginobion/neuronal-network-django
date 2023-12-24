from django.shortcuts import render
import pandas as pd 
import pickle
from django.http import HttpResponseRedirect, HttpResponse
from .forms.form import ImageForm
import cv2
import numpy as np
from django.urls import reverse
import tempfile
import os
import base64
from urllib.parse import quote, unquote

# Create your views here.

def final_page(request, temp_file_path):
    res = request.GET.get('res')
    unquoted_path = unquote(temp_file_path)
    with open(unquoted_path, 'rb') as temp_file:
        image_bytes = temp_file.read()

    # Encode the image bytes to base64
    encoded_image = base64.b64encode(image_bytes).decode()

    # Pass the base64-encoded image and other data to the template context
    context = {'encoded_image': encoded_image, 'res': res}

    # Remove the temporary file after reading
    os.remove(unquoted_path)

    return render(request, 'network/final.html', context)

def process_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Obtener imagen
            uploaded_image = form.cleaned_data['image']

            # Leer imagen
            file_content = uploaded_image.read()

            image_array = np.frombuffer(file_content, np.uint8)

            # Decodifica la imagen
            image_raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Procesar imagen
            result = process_image_function(image_raw)

            res = ""

            if result[0][0] == 0:
                res = "Es una hoja sana"

            if result[0][0] == 1:
                res = "Es una hoja seca"

            if result[0][0] == 2:
                res = "Es una hoja enferma"

            _, buffer = cv2.imencode('.jpg', image_raw)
            image_bytes = buffer.tobytes()

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_file_path = temp_file.name

            quoted_temp_file_path = quote(temp_file_path)

            # Pass the processed image as a URL parameter
            url = reverse('final_page', args=[quoted_temp_file_path]) + f'?res={res}'
            return HttpResponseRedirect(url)

    return HttpResponse('No se envió imagen')

def resize_and_crop(image, target_size):
    # Obtener las dimensiones originales
    height, width = image.shape[:2]

    # Encuentra las coordenadas del recorte
    center_x, center_y = width // 2, height // 2
    crop_size = min(width, height)

    # Calcula las coordenadas del cuadro de recorte
    crop_x1 = max(0, center_x - crop_size // 2)
    crop_x2 = min(width, center_x + crop_size // 2)
    crop_y1 = max(0, center_y - crop_size // 2)
    crop_y2 = min(height, center_y + crop_size // 2)

    # Recorta la imagen
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Redimensiona la imagen al tamaño deseado
    resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Normaliza los valores de píxeles al rango [0, 1]
    resized_image = resized_image / 255.0

    return resized_image

def process_image_function(image_raw):

    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)

    # Normaliza la imagen
    image_normalized = cv2.equalizeHist(image_gray)

    # Definir el kernel para la operación morfológica
    kernel = np.ones((5, 5), np.uint8)

    # Aplicar la operación de cierre
    image_closed = cv2.morphologyEx(image_normalized, cv2.MORPH_CLOSE, kernel)

    image_rgb = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2RGB)

    # # # Convertir la imagen a espacio de color HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Se definen rangos para el color verde en el espacio de color HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([160, 255, 255])

    # Crear una máscara para el color verde
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

    # Aplicar la máscara para obtener solo las regiones verdes
    green_filtered = cv2.bitwise_and(image_raw, image_raw, mask=mask_green)

    # Se definen rangos para el color blanco en el espacio de color HSV
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([255, 30, 255])

    # Crear una máscara para el color blanco
    mask_white = cv2.inRange(image_hsv, lower_white, upper_white)

    # Aplicar la máscara para obtener solo las regiones blancas
    white_filtered = cv2.bitwise_and(image_raw, image_raw, mask=mask_white)

    # Se definen rangos para el color negro en el espacio de color HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 255, 40])

    # Crear una máscara para el color negro
    mask_black = cv2.inRange(image_hsv, lower_black, upper_black)

    # Aplicar la máscara para obtener solo las regiones negras
    black_filtered = cv2.bitwise_and(image_raw, image_raw, mask=mask_black)

    # Combinar las regiones verdes, blancas y negras
    combined_image = cv2.bitwise_or(green_filtered, white_filtered)
    combined_image = cv2.bitwise_or(combined_image, black_filtered)

    # Redimensionar la imagen combinada
    resized_image = resize_and_crop(combined_image, 48)

    # Aplanar la imagen
    flattened_array = resized_image.flatten()
    flattened_array = flattened_array.reshape(1, -1)

    gaussian = pickle.load(open('my_network.sav','rb'))
    y_pred = gaussian.predict(flattened_array)
    output = pd.DataFrame(y_pred)
    return output

def index(request):
    return render(request, "network/hey.html", {})