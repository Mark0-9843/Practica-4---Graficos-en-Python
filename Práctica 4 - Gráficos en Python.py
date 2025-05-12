import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

frecuencia_muestreo = 0.2  # Frecuencia de muestreo (Hz)
intervalo_tiempo = 1 / frecuencia_muestreo  # Tiempo entre muestras (5 segundos)
muestras_promedio = 3  # Tamaño de la ventana para el promedio móvil
frecuencia_corte = 0.1  # Frecuencia de corte para filtro pasa bajas
banda_pasa_bandas = [0.02, 0.08]  # Frecuencia baja y alta para filtro pasa bandas

# Función principal de análisis y graficación
def grafico_y_filtrado(tiempo, datos_senal, etiqueta_eje_y, titulo_grafica):
    cantidad_muestras = len(datos_senal)
    frecuencias_fft = fftfreq(cantidad_muestras, d=intervalo_tiempo)[:cantidad_muestras // 2]

    # Promedio móvil
    senal_promediada = datos_senal.rolling(window=muestras_promedio, center=True).mean()

    # Filtro pasa bajas
    b_pasabajas, a_pasabajas = butter(N=4, Wn=frecuencia_corte, btype='low', analog=False)
    senal_filtrada_pasabajas = filtfilt(b_pasabajas, a_pasabajas, datos_senal)

    # Filtro pasa bandas
    b_pasabandas, a_pasabandas = butter(N=4, Wn=banda_pasa_bandas, btype='bandpass', analog=False)
    senal_filtrada_pasabandas = filtfilt(b_pasabandas, a_pasabandas, datos_senal)

    # FFT
    transformada_fft = np.abs(fft(datos_senal))[:cantidad_muestras//2]

    # Crear figura con 5 gráficas
    fig, ejes = plt.subplots(5, 1, figsize=(12, 15))
    fig.suptitle(f'{titulo_grafica}', fontsize=16)

    ejes[0].plot(tiempo, datos_senal)
    ejes[0].set_title('1. Señal Original')

    ejes[1].plot(tiempo, datos_senal, label='Original')
    ejes[1].plot(tiempo, senal_promediada, label='Promediado Móvil', linestyle='--')
    ejes[1].set_title('2. Promediado Móvil')
    ejes[1].legend()

    ejes[2].plot(tiempo, datos_senal, label='Original')
    ejes[2].plot(tiempo, senal_filtrada_pasabajas, label='Filtro Pasa Bajas', linestyle='--')
    ejes[2].set_title('3. Filtro Pasa Bajas')
    ejes[2].legend()

    ejes[3].plot(tiempo, datos_senal, label='Original')
    ejes[3].plot(tiempo, senal_filtrada_pasabandas, label='Filtro Pasa Bandas', linestyle='--')
    ejes[3].set_title('4. Filtro Pasa Bandas')
    ejes[3].legend()

    ejes[4].plot(frecuencias_fft, transformada_fft)
    ejes[4].set_title('5. Transformada Rápida de Fourier (FFT)')

    for eje in ejes:
        eje.set_xlabel('Tiempo (s)' if 'FFT' not in eje.get_title() else 'Frecuencia (Hz)')
        eje.set_ylabel(etiqueta_eje_y)
        eje.grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Cargar los datos desde los archivos CSV
datos_humedad = pd.read_csv('humedad.csv')
datos_temperatura = pd.read_csv('temperatura.csv')
datos_viento = pd.read_csv('viento.csv')

# Ajustar eje de tiempo (dato cada 5 segundos)
tiempo_humedad = datos_humedad['Tiempo'] * 5
tiempo_temperatura = datos_temperatura['Tiempo'] * 5
tiempo_viento = datos_viento['Tiempo'] * 5

# Aplicar análisis a cada señal
grafico_y_filtrado(tiempo_humedad, datos_humedad['Humedad_Relativa_%'], 'Humedad (%)', 'Análisis de Humedad Relativa')
grafico_y_filtrado(tiempo_temperatura, datos_temperatura['Temperatura_C'], 'Temperatura (°C)', 'Análisis de Temperatura')
grafico_y_filtrado(tiempo_viento, datos_viento['Velocidad_Viento_mps'], 'Velocidad (m/s)', 'Análisis de Velocidad del Viento')
grafico_y_filtrado(tiempo_viento, datos_viento['Direccion_Viento_deg'], 'Dirección (°)', 'Análisis de Dirección del Viento')
