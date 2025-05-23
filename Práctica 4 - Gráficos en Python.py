# TEAM markoboquiñechainajesu

# ESTE CÓDIGO REALIZA UNA FILTRACIÓN Y GRAFICACIÓN DE DATOS ORIGINARIOS DE ARCHIVOS CSV.

# 12 / 05 / 2025 - V. 2. 0. 1 - GRÁFICOS EN PYTHON

import pandas as pd # Importa la biblioteca Pandas para manipulación de datos en forma de tablas (DataFrames)
import numpy as np # Importa NumPy para operaciones matemáticas y manejo de arreglos numéricos
import matplotlib.pyplot as plt # Importa matplotlib.pyplot para crear gráficos y visualizaciones
from scipy.signal import butter, filtfilt # Importa funciones para diseñar y aplicar filtros digitales
from scipy.fft import fft, fftfreq # Importa funciones para calcular la Transformada Rápida de Fourier y sus frecuencias asociadas

frecuencia_muestreo = 0.2  #Frecuencia de muestreo (Hz)
intervalo_tiempo = 1 / frecuencia_muestreo  #Tiempo entre muestras (5 segundos)
muestras_promedio = 3  #Tamaño de la ventana para el promedio móvil
frecuencia_corte = 0.1  #Frecuencia de corte para filtro pasa bajas
banda_pasa_bandas = [0.02, 0.08]  #Frecuencia baja y alta para filtro pasa bandas

#Función principal de filtración y graficación
def grafico_y_filtrado(tiempo, datos_senal, etiqueta_eje_y, titulo_grafica): #Graficar una señal en función del tiempo.
    cantidad_muestras = len(datos_senal) #sirve para contar cuántos datos tiene la señal, es decir, mide la longitud de la lista o arreglo datos_senal.
    frecuencias_fft = fftfreq(cantidad_muestras, d=intervalo_tiempo)[:cantidad_muestras // 2] #se utiliza en el contexto del análisis de frecuencia con Transformada Rápida de Fourier (FFT)

    #Promedio móvil
    senal_promediada = datos_senal.rolling(window=muestras_promedio, center=True).mean() #Aplica promedio móvil centrado

    #Filtro pasa bajas
    b_pasabajas, a_pasabajas = butter(N=4, Wn=frecuencia_corte, btype='low', analog=False) #Diseño del filtro pasa bajas de cuarto orden con frecuencia de corte definida
    senal_filtrada_pasabajas = filtfilt(b_pasabajas, a_pasabajas, datos_senal) # Filtrado sin desfase

    #Filtro pasa bandas
    b_pasabandas, a_pasabandas = butter(N=4, Wn=banda_pasa_bandas, btype='bandpass', analog=False) #Diseño del filtro pasa bandas de cuarto orden con el rango de frecuencias definido
    senal_filtrada_pasabandas = filtfilt(b_pasabandas, a_pasabandas, datos_senal) #Aplicación del filtro pasa bandas con filtfilt para mantener la fase

    #FFT
    transformada_fft = np.abs(fft(datos_senal))[:cantidad_muestras//2] #Magnitud de la FFT

    # Crear figura con 5 gráficas
    fig, ejes = plt.subplots(5, 1, figsize=(12, 15)) # Crea una figura con 5 subgráficas verticales (5 filas, 1 columna) de tamaño 12x15 pulgadas 
    fig.suptitle(f'{titulo_grafica}', fontsize=16) # Establece el título general de la figura usando el parámetro recibido

    ejes[0].plot(tiempo, datos_senal) # Grafica la señal original contra el tiempo en el primer eje
    ejes[0].set_title('1. Señal Original') # Título del primer subgráfico

    ejes[1].plot(tiempo, datos_senal, label='Original') # Grafica la señal original en el segundo eje
    ejes[1].plot(tiempo, senal_promediada, label='Promediado Móvil', linestyle='--') # Superpone la señal con promedio móvil
    ejes[1].set_title('2. Promediado Móvil') # Título del segundo subgráfico
    ejes[1].legend() # Muestra la leyenda para identificar las curvas

    ejes[2].plot(tiempo, datos_senal, label='Original') # Grafica la señal original en el tercer eje
    ejes[2].plot(tiempo, senal_filtrada_pasabajas, label='Filtro Pasa Bajas', linestyle='--') # Superpone la señal filtrada con filtro pasa bajas
    ejes[2].set_title('3. Filtro Pasa Bajas') # Título del tercer subgráfico
    ejes[2].legend() # Muestra la leyenda para identificar las curvas

    ejes[3].plot(tiempo, datos_senal, label='Original') #forma parte de un conjunto de subgráficas (subplots), y específicamente dibuja una curva en el cuarto eje
    ejes[3].plot(tiempo, senal_filtrada_pasabandas, label='Filtro Pasa Bandas', linestyle='--') #dibuja una segunda curva en el mismo gráfico del eje ejes[3] (cuarto subplot), y representa una versión filtrada de la señal original usando un filtro pasa bandas.
    ejes[3].set_title('4. Filtro Pasa Bandas')#sirve para colocar un título en la cuarta subgráfica (subplot) de una figura creada con matplotlib.pyplot.subplots().


    ejes[3].legend()#sirve para mostrar la leyenda en la subgráfica número 4 (índice 3) de una figura con subplots, creada usando matplotlib.

    ejes[4].plot(frecuencias_fft, transformada_fft)#dibuja una gráfica de espectro de frecuencias en el quinto subplot (ejes[4]) de una figura creada con matplotlib.pyplot.subplots().
    ejes[4].set_title('5. Transformada Rápida de Fourier (FFT)') #Aestablece el título del quinto subplot (ejes[4]) en una figura de múltiples gráficas creada con matplotlib

    for eje in ejes: #Recorre cada subgráfico (subplot) de la figura generada
        eje.set_xlabel('Tiempo (s)' if 'FFT' not in eje.get_title() else 'Frecuencia (Hz)') #Etiqueta del eje X: 'Tiempo (s)' si no es FFT, o 'Frecuencia (Hz)' si es el gráfico de la FFT
        eje.set_ylabel(etiqueta_eje_y) #Etiqueta del eje Y con el nombre de la magnitud (ej. Humedad, Temperatura, etc.)
        eje.grid()#activa la cuadrícula (grid) en el gráfico correspondiente a eje en una figura de matplotlib.

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #Ajuste de espacio
    plt.show() #Mostrar la figura

#Cargar los datos desde los archivos CSV
datos_humedad = pd.read_csv('humedad.csv')  # Carga los datos del archivo humedad.csv en un DataFrame llamado datos_humedad
datos_temperatura = pd.read_csv('temperatura.csv') # Carga los datos del archivo temperatura.csv en un DataFrame llamado datos_temperatura
datos_viento = pd.read_csv('viento.csv') # Carga los datos del archivo viento.csv en un DataFrame llamado datos_viento

#Ajustar tiempo (dato cada 5 segundos)
tiempo_humedad = datos_humedad['Tiempo'] * 5 #Ajuste del tiempo: cada unidad representa 5 segundos
tiempo_temperatura = datos_temperatura['Tiempo'] * 5 #Ajuste del tiempo: cada unidad representa 5 segundos
tiempo_viento = datos_viento['Tiempo'] * 5 #Ajuste del tiempo: cada unidad representa 5 segundos

#Aplicar análisis a cada señal
grafico_y_filtrado(tiempo_humedad, datos_humedad['Humedad_Relativa_%'], 'Humedad (%)', 'Humedad') #Genera un gráfico con esos datos.
grafico_y_filtrado(tiempo_temperatura, datos_temperatura['Temperatura_C'], 'Temperatura (°C)', 'Temperatura') #grafica y filtra datos de temperatura al largo del tiempo  
grafico_y_filtrado(tiempo_viento, datos_viento['Velocidad_Viento_mps'], 'Velocidad (m/s)', 'Velocidad del Viento') #  para visualizar y filtrar la velocidad del viento en función del tiempo
grafico_y_filtrado(tiempo_viento, datos_viento['Direccion_Viento_deg'], 'Dirección (°)', 'Dirección del Viento') #grafica y filtra la dirección del viento con la misma función grafico_y_filtrado
