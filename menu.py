import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Función para cada sección
def Modelo_logistico():
    st.title('Simulación del Modelo Logístico de Crecimiento Poblacional')

    r = st.slider('Tasa de crecimiento máximo (r)', 0.0, 1.0, 0.5, step=0.01)
    K = st.slider('Capacidad de carga (K)', 100, 1000, 500, step=50)
    P0 = st.slider('Población inicial (P0)', 1, 100, 10)
    periodos = st.slider('Número de períodos', 1, 100, 20)

    P = [P0]

    for t in range(1, periodos + 1):
        Pt = (K*P0*np.exp(r*t))/(K+P0*(np.exp(r*t)-1))
        P.append(Pt)

    st.subheader('Gráfica del Crecimiento Poblacional')
    fig, ax = plt.subplots()
    ax.plot(range(periodos + 1), P, marker='o')
    ax.set_title('Crecimiento poblacional usando el modelo logístico')
    ax.set_xlabel('Períodos')
    ax.set_ylabel('Población')
    ax.grid(True)
    # Mostrar la gráfica en la aplicación
    st.pyplot(fig)
    #Tabla
    st.subheader('Resultados')
    resultados = {'Población': P}
    st.write(resultados)


def Modelo_exponencial():
    st.title('Simulación del Modelo Exponencial de Crecimiento Poblacional')

    # Parámetros de entrada
    r = st.slider('Tasa de crecimiento (r)', 0.0, 1.0, 0.2, step=0.01)
    P0 = st.slider('Población inicial (P0)', 1, 100, 10)
    periodos = st.slider('Número de períodos', 1, 100, 20)

    # Lista para almacenar los valores de la población
    P = [P0]

    # Simulación de los períodos
    for t in range(1, periodos + 1):
        Pt = P[-1] * (1 + r)
        P.append(Pt)
    # Graficar los resultados
    st.subheader('Gráfica del Crecimiento Poblacional')
    fig, ax = plt.subplots()
    ax.plot(range(periodos + 1), P, marker='o')
    ax.set_title('Crecimiento poblacional usando el modelo exponencial')
    ax.set_xlabel('Períodos')
    ax.set_ylabel('Población')
    ax.grid(True)
    st.pyplot(fig)

    # Mostrar los resultados
    st.subheader('Resultados')
    resultados = {'Población': P}
    st.write(resultados)

def Modelo_mathusiano():
    st.title('Simulación del Modelo Malthusiano de Crecimiento Poblacional con Inmigración')

    # Parámetros de entrada
    r = st.slider('Tasa de crecimiento (r)', 0.0, 1.0, 0.2, step=0.01)
    I = st.slider('Tasa de inmigración (I)', 0, 100, 10, step=1)
    P0 = st.slider('Población inicial (P0)', 1, 100, 10)
    periodos = st.slider('Número de períodos', 1, 100, 20)

    # Lista para almacenar los valores de la población
    P = [P0]

    # Simulación de los períodos
    for t in range(1, periodos + 1):
        Pt = P[-1] + r * P[-1] + I
        P.append(Pt)

    # Graficar los resultados
    st.subheader('Gráfica del Crecimiento Poblacional con Inmigración')
    fig, ax = plt.subplots()
    ax.plot(range(periodos + 1), P, marker='o')
    ax.set_title('Crecimiento poblacional con inmigración usando el modelo Malthusiano')
    ax.set_xlabel('Períodos')
    ax.set_ylabel('Población')
    ax.grid(True)

    # Mostrar la gráfica en la aplicación
    st.pyplot(fig)

    # Mostrar los resultados en una tabla
    st.subheader('Resultados')
    resultados = {'Población': P}
    st.write(resultados)

def Modelo_rickert():
    # Título de la aplicación
    st.title('Simulación del Modelo de Crecimiento de Rickert')

    # Parámetros de entrada
    r = st.slider('Tasa de crecimiento (r)', 0.0, 1.0, 0.2, step=0.01)
    K = st.slider('Capacidad de carga (K)', 100, 1000, 500, step=50)
    # m = st.slider('Parámetro de saturación (m)', 0.1, 10.0, 2.0, step=0.1)
    P0 = st.slider('Población inicial (P0)', 1, 100, 10)
    periodos = st.slider('Número de períodos', 1, 100, 20)

    # Lista para almacenar los valores de la población
    P = [P0]

    # Simulación de los períodos
    for t in range(1, periodos + 1):
        Pt = P[-1] * np.exp(r*(1 - (P[-1] / K)))
        Pt = max(Pt, 0)  # La población no puede ser negativa
        P.append(Pt)

    # Graficar los resultados
    st.subheader('Gráfica del Crecimiento Poblacional')
    fig, ax = plt.subplots()
    ax.plot(range(periodos + 1), P, marker='o')
    ax.set_title('Crecimiento poblacional usando el modelo de Rickert')
    ax.set_xlabel('Períodos')
    ax.set_ylabel('Población')
    ax.grid(True)
    st.pyplot(fig)
    # Mostrar los resultados en una tabla
    st.subheader('Resultados')
    resultados = {'Población': P}
    st.write(resultados)


def Modelo_gompertz():
    st.title('Simulación del Modelo de Crecimiento de Gompertz')
    # Parámetros de entrada
    r = st.slider('Tasa de crecimiento (r)', 0.0, 1.0, 0.2, step=0.01)
    K = st.slider('Capacidad de carga (K)', 100, 1000, 500, step=50)
    P0 = st.slider('Población inicial (P0)', 1, 100, 10)
    periodos = st.slider('Número de períodos', 1, 100, 20)
    P = [P0]
    # Simulación de los períodos usando la fórmula de Gompertz
    for t in range(1, periodos + 1):
        Pt = K * np.exp(np.exp(-r * t) * np.log(P0/K))
        P.append(Pt)
    # Graficar los resultados
    st.subheader('Gráfica del Crecimiento Poblacional')
    fig, ax = plt.subplots()
    ax.plot(range(periodos + 1), P, marker='o')
    ax.set_title('Crecimiento poblacional usando el modelo de Gompertz')
    ax.set_xlabel('Períodos')
    ax.set_ylabel('Población')
    ax.grid(True)
    st.pyplot(fig)
    # Mostrar los resultados en una tabla
    st.subheader('Resultados')
    resultados = {'Población': P}
    st.write(resultados)

def Modelo_propio():
    st.title('Simulación del Modelo de Crecimiento Propio con 3 grupos')
    # Parámetros de entrada

   # Inputs del usuario
    st.sidebar.header("Parámetros del Modelo")

    # Tasas de natalidad y mortalidad para tres grupos de edad
    b0 = st.sidebar.slider("Tasa de natalidad para 0-4 años", 0.0, 0.1, 0.05)
    b1 = st.sidebar.slider("Tasa de natalidad para 5-9 años", 0.0, 0.1, 0.03)
    b2 = st.sidebar.slider("Tasa de natalidad para 10-14 años", 0.0, 0.1, 0.01)

    m0 = st.sidebar.slider("Tasa de mortalidad para 0-4 años", 0.0, 0.1, 0.02)
    m1 = st.sidebar.slider("Tasa de mortalidad para 5-9 años", 0.0, 0.1, 0.01)
    m2 = st.sidebar.slider("Tasa de mortalidad para 10-14 años", 0.0, 0.1, 0.005)

    # Vector de población inicial
    p0_0_4 = st.sidebar.number_input("Población inicial para 0-4 años", min_value=0, value=1000)
    p0_5_9 = st.sidebar.number_input("Población inicial para 5-9 años", min_value=0, value=800)
    p0_10_14 = st.sidebar.number_input("Población inicial para 10-14 años", min_value=0, value=600)

    # Número de períodos para la proyección
    t_max = st.sidebar.slider("Número de períodos para la proyección", 1, 50, 10)

    # Matriz de transición
    T = np.array([
        [1 - m0, b0, 0],
        [0, 1 - m1, b1],
        [0, 0, 1 - m2]
    ])

    # Vector de población inicial
    P_0 = np.array([p0_0_4, p0_5_9, p0_10_14])

    # Proyección de la población
    P_values = np.zeros((t_max + 1, 3))
    P_values[0] = P_0

    for t in range(1, t_max + 1):
        P_values[t] = np.dot(np.linalg.matrix_power(T, t), P_0)

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))

    # Periodos
    periodos = np.arange(t_max + 1)

    # Gráfica de la evolución de la población
    ax.plot(periodos, P_values[:, 0], label='0-4 años')
    ax.plot(periodos, P_values[:, 1], label='5-9 años')
    ax.plot(periodos, P_values[:, 2], label='10-14 años')

    ax.set_xlabel('Período')
    ax.set_ylabel('Población')
    ax.set_title('Evolución de la Población por Grupo de Edad')
    ax.legend()
    ax.grid(True)

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

def Modelo_De_Crecimiento_Multigrupal():


# Título
    st.title("Modelo de Crecimiento de la Población con grupos variables")

    # Input del usuario para el número de grupos de edad
    num_grupos = st.sidebar.slider("Número de grupos de edad", 2, 10, 3)

    # Inputs del usuario para tasas de natalidad y mortalidad
    b = []
    m = []
    for i in range(num_grupos):
        b.append(st.sidebar.slider(f"Tasa de natalidad para grupo {i+1}", 0.0, 0.1, 0.01))
        m.append(st.sidebar.slider(f"Tasa de mortalidad para grupo {i+1}", 0.0, 0.1, 0.01))

    # Inputs del usuario para la población inicial
    P_0 = []
    for i in range(num_grupos):
        P_0.append(st.sidebar.number_input(f"Población inicial para grupo {i+1}", min_value=0, value=1000))

    # Convertir listas a arrays
    b = np.array(b)
    m = np.array(m)
    P_0 = np.array(P_0)

    # Número de períodos para la proyección
    t_max = st.sidebar.slider("Número de períodos para la proyección", 1, 50, 10)

    # Crear matriz de transición
    T = np.zeros((num_grupos, num_grupos))

    # Rellenar la matriz de transición
    for i in range(num_grupos):
        if i < num_grupos - 1:
            T[i, i + 1] = b[i]  # Natalidad en el grupo i
        T[i, i] = 1 - m[i]  # Mortalidad en el grupo i

    # Proyección de la población
    P_values = np.zeros((t_max + 1, num_grupos))
    P_values[0] = P_0

    for t in range(1, t_max + 1):
        P_values[t] = np.dot(np.linalg.matrix_power(T, t), P_0)

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))

    # Periodos
    periodos = np.arange(t_max + 1)

    # Gráfica de la evolución de la población
    for i in range(num_grupos):
        ax.plot(periodos, P_values[:, i], label=f'Grupo {i+1}')

    ax.set_xlabel('Período')
    ax.set_ylabel('Población')
    ax.set_title('Evolución de la Población por Grupo de Edad')
    ax.legend()
    ax.grid(True)

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

def Modelo_Multigrupal_Con_Inmigracion():
    # Título
    st.title("Modelo de Crecimiento de la Población con Inmigración")

    # Input del usuario para el número de grupos de edad
    num_grupos = st.sidebar.slider("Número de grupos de edad", 2, 10, 3)

    # Inputs del usuario para tasas de natalidad y mortalidad
    b = []
    m = []
    for i in range(num_grupos):
        b.append(st.sidebar.slider(f"Tasa de natalidad para grupo {i+1}", 0.0, 0.1, 0.01))
        m.append(st.sidebar.slider(f"Tasa de mortalidad para grupo {i+1}", 0.0, 0.1, 0.01))

    # Input para la inmigración en el último grupo
    inmigracion = st.sidebar.number_input(f"Inmigración en el grupo {num_grupos}", min_value=0, value=100)

    # Inputs del usuario para la población inicial
    P_0 = []
    for i in range(num_grupos):
        P_0.append(st.sidebar.number_input(f"Población inicial para grupo {i+1}", min_value=0, value=1000))

    # Convertir listas a arrays
    b = np.array(b)
    m = np.array(m)
    P_0 = np.array(P_0)

    # Número de períodos para la proyección
    t_max = st.sidebar.slider("Número de períodos para la proyección", 1, 50, 10)

    # Crear matriz de transición
    T = np.zeros((num_grupos, num_grupos))

    # Rellenar la matriz de transición
    for i in range(num_grupos):
        if i < num_grupos - 1:
            T[i, i + 1] = b[i]  # Natalidad en el grupo i
        T[i, i] = 1 - m[i]  # Mortalidad en el grupo i

    print(T)

    # Proyección de la población
    P_values = np.zeros((t_max + 1, num_grupos))
    P_values[0] = P_0

    for t in range(1, t_max + 1):
        P_values[t] = np.dot(T, P_values[t - 1])
        P_values[t, -1] += inmigracion  # Agregar inmigración al último grupo

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))

    # Periodos
    periodos = np.arange(t_max + 1)

    # Gráfica de la evolución de la población
    for i in range(num_grupos):
        ax.plot(periodos, P_values[:, i], label=f'Grupo {i+1}')

    ax.set_xlabel('Período')
    ax.set_ylabel('Población')
    ax.set_title('Evolución de la Población por Grupo de Edad con Inmigración')
    ax.legend()
    ax.grid(True)

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)


# Crear el menú en la barra lateral
menu = st.sidebar.selectbox("Modelos", ["Modelo Logistico", "Modelo Exponencial", "Modelo Mathusiano", "Modelo Rickert", "Modelo Gompertz","Modelo 3 Grupos","Modelo multigrupal","Modelo multigrupal con inmigracion"])

# Mostrar la sección seleccionada
if menu == "Modelo Logistico":
    Modelo_logistico()
elif menu == "Modelo Exponencial":
    Modelo_exponencial()
elif menu == "Modelo Mathusiano":
    Modelo_mathusiano()
elif menu == "Modelo Rickert":
    Modelo_rickert()
elif menu == "Modelo Gompertz":
    Modelo_gompertz()
elif menu == "Modelo 3 Grupos":
    Modelo_propio()
elif menu == "Modelo multigrupal":
    Modelo_De_Crecimiento_Multigrupal()
elif menu == "Modelo multigrupal con inmigracion":
    Modelo_Multigrupal_Con_Inmigracion()
