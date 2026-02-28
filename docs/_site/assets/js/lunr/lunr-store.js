var store = [{
        "title": "Week 0 - Introduction to the Carla Simulator",
        "excerpt":"This is my first week of studies in the doctorate, where the work methodology, schedule and tools to use were specified. It was defined that the operating system to be used is Ubuntu 22.04 LTS, which was installed on a laptop equipped with an Nvidia RTX 4060 graphics card and...","categories": ["-weekly","Log"],
        "tags": ["CARLA 0.9.15"],
        "url": "/-weekly/log/week-0/",
        "teaser": null
      },{
        "title": "Week 1 - Instalación de Carla Simulator",
        "excerpt":"Construcción de la Infraestructura de Trabajo Como fase inicial del doctorado, se planteó el desarrollo de un auto autónomo en un entorno simulado mediante la técnica “end to end learning”, utilizando la arquitectura de red PilotNet. Para llevar a cabo este proyecto, se emplearon el Simulador CARLA 0.9.14 y el...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","Docker","ROS 2","ROS_BRIDGE"],
        "url": "/weekly%20log/week-1/",
        "teaser": null
      },{
        "title": "Week 2 - Captura y balanceo del dataset",
        "excerpt":"Generacion del dataset A continuación, se describe cómo fue elaborada la preparación de los datos para la implementación de la arquitectura PilotNet, con el objetivo de permitir que un automóvil, en este caso simulado, prediga los comandos o acciones requeridos para conducir el automóvil clonando el comportamiento del piloto automático...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","balanceo","ROS 2","ROS_BRIDGE","Dataset"],
        "url": "/weekly%20log/week-2/",
        "teaser": null
      },{
        "title": "Week 3 - Balanceo del dataset",
        "excerpt":"Exploracion de los datos El dataset generado consta de un conjunto de 15.700 imágenes con sus respectivas etiquetas. Al representar los valores correspondientes al ángulo de giro (steer) en un histograma con 50 intervalos, se observa una distribución altamente desequilibrada. histogram = plt.hist(df['steer_values'], bins=50) plt.xlabel(\"Ángulo de giro\") plt.ylabel(\"# de Conteos\")...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","balanceo","ROS 2","ROS_BRIDGE","Dataset"],
        "url": "/weekly%20log/week-3/",
        "teaser": null
      },{
        "title": "Week 4 - Modelo Nvidia -PilotNet",
        "excerpt":"Entrenamiento de la Red PilotNet La red neuronal PilotNet fue entrenada mediante la biblioteca TensorFlow 2.0 utilizando un conjunto de datos compuesto por 9000 imágenes recopiladas del simulador CARLA. Cada imagen fue etiquetada con su respectivo ángulo de giro. Para prevenir problemas de sobreajuste, se implementó un balanceo de datos...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","balanceo","ROS 2","ROS_BRIDGE","Dataset"],
        "url": "/weekly%20log/week4/",
        "teaser": null
      },{
        "title": "Week 5 - Pruebas del Piloto Automático Seguidor de Carril",
        "excerpt":"Pruebas del Piloto Automático Seguidor de Carril Para evaluar el desempeño del modelo entrenado, se implementó un piloto dummy en ROS 2 Humble. Este piloto utiliza el paquete ros_bridge para establecer la comunicación con el simulador CARLA. Durante las pruebas, se diseñó un nodo que se suscribe y publica a...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","ROS 2","ROS_BRIDGE","tensorFlow"],
        "url": "/weekly%20log/week-5/",
        "teaser": null
      },{
        "title": "Week 6 - Seguidor de linea PID Simple",
        "excerpt":"Seguidor de linea PID Simple Creación de un simple seguidor de línea PID para el robot Turtlebot2 en ROS 2 Humble. Se desarrolló un script utilizando OpenCV para detectar la línea a seguir. Este algoritmo realiza un recorte de la imagen en la región de interés, convierte la imagen a...","categories": ["weekly Log"],
        "tags": ["ROS 2","turtlebot2","openCV","PID"],
        "url": "/weekly%20log/week-6/",
        "teaser": null
      },{
        "title": "Week 7 - Pruebas del Piloto Automático Seguidor de línea con robot Turtlebot2",
        "excerpt":"Prueba del Turtlebot2 Para entrenar la red CNN PilotNet, se empleó un controlador PID seguidor de línea como experto para regular el ángulo de giro. Este control se basa en el cálculo del error de posición utilizando visión artificial, donde se detecta la línea y el centro de la imagen....","categories": ["weekly Log"],
        "tags": ["Turtlebot2","ROS 2","OpenCV","tensorFlow"],
        "url": "/weekly%20log/week-7/",
        "teaser": null
      },{
        "title": "Week 8 - Pruebas del Piloto Automático Seguidor de línea con robot Turtlebot2",
        "excerpt":"Prueba piloto automático La prueba de piloto automático involucró el uso de 130,000 imágenes junto con etiquetas de velocidad angular y lineal (ω, v). El conjunto de datos se generó utilizando el robot Turtlebot 2 y una cámara USB. El piloto experto consistió en un controlador PID estándar que permitía...","categories": ["weekly Log"],
        "tags": ["Turtlebot2","ROS 2","OpenCV","tensorFlow"],
        "url": "/weekly%20log/week-8/",
        "teaser": null
      },{
        "title": "Week 9 -  Detección de carriles y el cálculo del radio de curvatura",
        "excerpt":"Para la detección de carriles se emplearon las bibliotecas OpenCV y los métodos Sobel para resaltar características en la imagen que podrían indicar la presencia de bordes de carriles. Además, se aplicó la transformación de perspectiva para obtener una vista en “ojos de pájaro”, lo que facilita la detección de...","categories": ["weekly Log"],
        "tags": ["ROS 2","OpenCV","CARLA Simulator"],
        "url": "/weekly%20log/week-9/",
        "teaser": null
      },{
        "title": "Week 10 -  Mejora en la Detección del carríl",
        "excerpt":"Mejora en la detección del carril Se ha realizado una mejora significativa en el algoritmo de detección de carriles, enfocándose en la detección precisa de las líneas de demarcación de las vías. Esta mejora incluyó la capacidad de detectar tanto las líneas amarillas como las blancas, así como la mejora...","categories": ["weekly Log"],
        "tags": ["ROS 2","OpenCV","CARLA Simulator"],
        "url": "/weekly%20log/week-10/",
        "teaser": null
      },{
        "title": "Week 11 -  Etiquetado de imagenes",
        "excerpt":"Etiquetado manual Debido a que el enfoque de detección de carriles implementado utilizando visión computacional clásica no demostró la robustez deseada, se optó por generar un dataset de manera manual. Este proceso involucró el etiquetado manual de un conjunto de datos utilizando la herramienta Labelme. Se crearon etiquetas para tres...","categories": ["weekly Log"],
        "tags": ["labelme","pytorch","CARLA Simulator"],
        "url": "/weekly%20log/week11/",
        "teaser": null
      },{
        "title": "Week 12 - Deep Learning Lane Detection Test",
        "excerpt":"Deep Learning Lane Detection Test La prueba se realizó en tres pueblos distintos: Town04, Town02 y Town01. En las pruebas realizadas en Town04, se observó que el modelo es capaz de inferir bastante bien las líneas que demarcan el carril por el cual navega el automóvil (líneas de la señalética...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch"],
        "url": "/weekly%20log/week12/",
        "teaser": null
      },{
        "title": "Week 13 - Prueba de detección de curvatura",
        "excerpt":"Prueba de detección de curvatura   En algunos tramos funciona perfectamente, aún está pendiente mejorar la robustes para mejora la percepción del carril en diferentes condiciones y pueblos.    ","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch"],
        "url": "/weekly%20log/week13/",
        "teaser": null
      },{
        "title": "Week 14 - Prueba de especialistas",
        "excerpt":"se construyeron datasets especializados: uno para líneas rectas y otro para líneas curvas. Esto permitió el desarrollo de dos modelos expertos en cada tipo de trayectoria. Utilizamos un detector de curvatura para determinar el radio de las curvas y, mediante una condición lógica, seleccionar el modelo adecuado según el caso:...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","Tensorflow"],
        "url": "/weekly%20log/week14/",
        "teaser": null
      },{
        "title": "Week 15 - Prueba Visual del Detector de Carril",
        "excerpt":"Prueba Visual del Detector de Carril Para determinar el carril y su curvatura, implementamos la PythonAPI de CARLA para obtener los waypoints actuales del vehículo. Utilizamos el método draw_string para visualizar las predicciones de carril en el mapa global, donde se marcan las curvas en rojo y los trayectos rectos...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","ROS 2","ROS_BRIDGE"],
        "url": "/weekly%20log/week15/",
        "teaser": null
      },{
        "title": "Week 16 - Prueba Detector de Carril con MobileV3Small",
        "excerpt":"Prueba de modelo MobileV3Small Utilizamos el modelo de segmentación MobileNetV3Small para identificar diferentes clases: marca de carril izquierdo, marca de carril derecho y fondo. El entrenamiento se llevó a cabo con Fastai, una biblioteca de alto nivel basada en PyTorch que facilita la creación y entrenamiento de modelos de aprendizaje...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","ROS 2","ROS_BRIDGE","Pytorch"],
        "url": "/weekly%20log/week16/",
        "teaser": null
      },{
        "title": "Week 17 - Balanceo del dataset",
        "excerpt":"(Recuperación) Entrenamiento del Modelo PilotNet Modificado para Inferir Steering, Throttle y Brake Preparación de los Datos Los datos utilizados para el entrenamiento del modelo incluyen imágenes del simulador CARLA y etiquetas correspondientes a los valores de dirección, aceleración y freno. Se realizaron varias transformaciones y balanceo de datos para asegurar...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","balanceo","PilotNet","Dataset","Tensorflow"],
        "url": "/weekly%20log/week17/",
        "teaser": null
      },{
        "title": "Week 18 - Detección de Cruces",
        "excerpt":"Detección de Cruces con GNSS Se ha implementado la detección de coordenadas GNSS para abordar los problemas que el modelo PilotNet presentaba al enfrentar intersecciones. Cuando el sistema detecta que se aproxima a una intersección, se desactiva temporalmente el modelo entrenado de PilotNet y se activa el modo automático para...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","PilotNet","ROS_BRIDGE"],
        "url": "/weekly%20log/week18/",
        "teaser": null
      },{
        "title": "Week 19 - Construcción Dataset MoE",
        "excerpt":"Construcción de Dataset para MOE La siguiente fase del proyecto consiste en entrenar dos modelos expertos, uno especializado en rectas y otro en curvas, como preparación para la creación de un modelo final que integre ambas experticias para mejorar la navegación autónoma basada en PilotNet. Para llevar a cabo esta...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","PilotNet","Moe"],
        "url": "/weekly%20log/week19/",
        "teaser": null
      },{
        "title": "Week 20 - Entrenamiento expertos (Moe)",
        "excerpt":"Entrenamiento de Expertos (MoE) Se construyó un dataset llamado dataset_moe para el entrenamiento de dos expertos end-to-end utilizando el modelo PilotNet en el simulador CARLA. El dataset fue etiquetado con el estado de la curvatura del carril, dividiéndolo en dos clases: “recta” y “curva”. Con esta clasificación, se entrenaron dos...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","PilotNet","MoE"],
        "url": "/weekly%20log/week20/",
        "teaser": null
      },{
        "title": "Week 21 - Preparación del Entrenamiento con EfficientViT",
        "excerpt":"Preparación del Entrenamiento con EfficientViT Para mejorar la percepción del carril y su curvatura, se decidió construir un dataset implementando las imágenes segmentadas suministradas por el simulador CARLA como etiquetas de las imágenes RGB en crudo obtenidas. Estas imágenes segmentadas permiten diferenciar los diversos elementos que aparecen en la escena,...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","Segmentación Semántica","Dataset","MoE"],
        "url": "/weekly%20log/week-21/",
        "teaser": null
      },{
        "title": "Week 22 - Entrenamiento con EfficientNet_b0",
        "excerpt":"Entrenamiento con EfficientNet_b0 Para el entrenamiento se utilizó un dataset que contiene imágenes en formato RGB y segmentadas, cada una etiquetada con la información de curvatura de la carretera. El dataset se generó utilizando el simulador CARLA, donde se capturaron imágenes RGB y segmentadas de diferentes ciudades. Cada imagen segmentada...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","Segmentación Semántica","Dataset","MoE","PyTorch"],
        "url": "/weekly%20log/week22/",
        "teaser": null
      },{
        "title": "Week 23 - Entrenamiento del Modelo Selector de Calzada",
        "excerpt":"Entrenamiento del Modelo Selector de Calzada Para este experimento se utilizó transfer learning, aprovechando un modelo preentrenado: EfficientViT. Este modelo combina las características de EfficientNet y Vision Transformers (ViT) para el procesamiento de imágenes. EfficientViT integra la eficiencia computacional de EfficientNet con la capacidad de capturar relaciones globales que ofrece...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","Segmentación Semántica","Dataset","MoE","PyTorch"],
        "url": "/weekly%20log/week23/",
        "teaser": null
      },{
        "title": "Week 24 - Entrenamiento del Clasificador de Curvatura con EfficientNet",
        "excerpt":"EfficientNet: Entrenamiento del Clasificador de Curvatura Se eligió el modelo EfficientNet_B0 para entrenar un clasificador de curvatura, con el objetivo de detectar dos clases: curva y recta . Para este entrenamiento, se construyó un dataset balanceado (mismo número de ejemplos de curvas y rectas), compuesto por 36,000 imágenes RGB etiquetadas...","categories": ["weekly Log"],
        "tags": ["CARLA 0.9.14","EfficientNet","Dataset","MoE","PyTorch"],
        "url": "/weekly%20log/week24/",
        "teaser": null
      },{
        "title": "Week 25 - Entrenamiento de Expertos en Conducción Autónoma en CARLA",
        "excerpt":"Entrenamiento de expertos Se entrenaron dos expertos, uno para carreteras rectas y otro para carreteras curvas, y adicionlamente se desarrollo un clasificador de curvatura para distinguir entre segmentos rectos y curvos, permitiendo que el modelo adecuado tome el control. Detalles del Entrenamiento Experto en Carreteras Rectas: Entrenado con 70,000 imágenes...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week25/",
        "teaser": null
      },{
        "title": "Week 26 - Entrenamiento de Expertos y Clasificador de Curvatura",
        "excerpt":"Durante el proceso de desarrollo del sistema de conducción autónoma, se entrenaron dos expertos especializados: uno para carreteras rectas y otro para carreteras curvas. Además, se desarrolló un clasificador de curvatura basado en el modelo EfficientNet_B0, encargado de distinguir entre segmentos rectos y curvos, permitiendo que el modelo adecuado tome...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week26/",
        "teaser": null
      },{
        "title": "Week 27 - Nuevo Entrenamiento de Expertos en Conducción Autónoma en CARLA",
        "excerpt":"Este experimento se realizó con un dataset completamente renovado, empleando una metodología diferente para su construcción. La metodología consistió en programar una alternancia entre el piloto automático y un modo “piloto borracho” cada cinco segundos. Este modo simula un comportamiento errático, con el fin de enriquecer el dataset y generar...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Road Detection","pytorch","expertos"],
        "url": "/weekly%20log/week27/",
        "teaser": null
      },{
        "title": "Week 28 - Pruebas de Expertos y Clasificador de Curvatura",
        "excerpt":"Videos de pruebas de expertos         train_plot   Video de prueba en Town02  Este escenario no fue visto por el modelo durante el entrenamiento.    Town07 es y Town05 fuero usados en el dataset.      ","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week28/",
        "teaser": null
      },{
        "title": "Week 29 - Corrección modelos expertos",
        "excerpt":"Corrección Durante la simulación, se presentó un error en la predicción de los comandos de control, ya que el modelo infería valores idénticos de manera constante. Tras revisar el código, se identificó que el problema estaba en el procesamiento de las imágenes de entrada al modelo: los parámetros de normalización...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week-29/",
        "teaser": null
      },{
        "title": "Week 30 - Evaluación de Modelos de Conducción Autónoma en Situaciones Extremas",
        "excerpt":"En este experimento, se realizaron pruebas para evaluar el desempeño de los modelos de conducción autónoma, tanto en tramos rectos como curvos, bajo diversas condiciones. El objetivo principal fue analizar cómo el vehículo reaccionaba y recuperaba su rumbo tras ser ubicado en situaciones extremas. Recuperación del Rumbo en Condiciones Críticas...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week30/",
        "teaser": null
      },{
        "title": "Week 31 - Evaluación de Modelos de Conducción Autónoma con Data Mixta 20% DAgger + 80% Piloto Experto",
        "excerpt":"Para esta serie de pruebas, el modelo fue entrenado utilizando un dataset mixto compuesto por un 20% de datos obtenidos mediante el método DAgger y un 80% de datos provenientes de un piloto experto. El entrenamiento se enfocó en mejorar el desempeño en tramos rectos y curvos. Resultados Desempeño General:...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week31/",
        "teaser": null
      },{
        "title": "Week 32 - Evaluación de Modelos de Conducción Autónoma con Data Mixta 40% DAgger vs Data \"burbuja\"",
        "excerpt":"Se realizaron pruebas comparando el modelo DAgger (con un 40 %) frente al modelo burbuja (datos se obtenidos en condiciones ideales). Las evaluaciones con el modelo derivado de DAgger mostraron gran robustez para resolver los cuatro casos de prueba, superándolos con facilidad. En contraste, el modelo burbuja no logró superar...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","pytorch","expertos"],
        "url": "/weekly%20log/week32/",
        "teaser": null
      },{
        "title": "Week 33 - Comparación cualitativa del modelos DAgger 40% monolítico y mezcla de expertos.",
        "excerpt":"Se llevaron a cabo pruebas cualitativas para observar el comportamiento de los modelos de navegación entrenados, evaluando su desempeño en diversos escenarios. Para mejorar la calidad de las pruebas y obtener información más detallada, se está adaptando la herramienta de medición y evaluación Behavior Metrics al framework ROS 2, que...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","TensorFlow","expertos"],
        "url": "/weekly%20log/week33/",
        "teaser": null
      },{
        "title": "Week 34 - Escritura del Paper, Integración de ROS 2 en BehaviorMetrics e Investigación de Nuevos Modelos para Conducción Autónoma",
        "excerpt":"Actualmente estoy desarrollando una investigación titulada “Comparativa del modelo PilotNet utilizando diferentes enfoques: entrenamiento con un dataset convencional vs. un dataset DAgger, y evaluación de un modelo monolítico frente a una separación en expertos especializados”. Mi objetivo principal es demostrar que el uso del dataset DAgger (Dataset Aggregation) permite entrenar...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","TensorFlow","expertos"],
        "url": "/weekly%20log/week34/",
        "teaser": null
      },{
        "title": "Week 35 - Prueba del modelo modifiedDeepestLSTM",
        "excerpt":"Este modelo se basa en la arquitectura presentada en el paper “Transferring Vision-Based End-to-End Autonomous Driving Decision-Making from Simulation to Real-World Vehicles”, adaptado para mejorar su capacidad de generalización y robustez. Para entrenarlo, se construyó un dataset con 30,000 imágenes, donde se aplicó balanceo de datos en función del valor...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","TensorFlow","expertos"],
        "url": "/weekly%20log/week35/",
        "teaser": null
      },{
        "title": "Week 36 - Entrenamiento del modelo modifiedDeepestLSTM",
        "excerpt":"Resumen del Análisis y Ajustes Steering MSE: La validación es más alta y variable, lo que indica dificultades en la generalización. Brake MSE: El modelo aprende bien, con una convergencia estable en entrenamiento y validación. Throttle MSE: Se observa overfitting, ya que la validación se mantiene más alta y variable....","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","TensorFlow","expertos"],
        "url": "/weekly%20log/week36/",
        "teaser": null
      },{
        "title": "Week 37 - Entrenamiento del modelo modifiedDeepestLSTM",
        "excerpt":"Prueba del Modelo Monolítico con PyTorch En este experimento se evaluó el rendimiento de un modelo monolítico para conducción autónoma, implementado en PyTorch. El modelo se entrenó utilizando el dataset DAgger40, en el que las entradas corresponden a la máscara de calzada extraída de imágenes segmentadas (de tamaño 3 ×...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","TensorFlow","expertos"],
        "url": "/weekly%20log/week37/",
        "teaser": null
      },{
        "title": "Week 38 - Comparativa del modelo modifiedDeepestLSTM con dataset burbuja y DAgger40",
        "excerpt":"Comparación Cualitativa de Modelos en el Simulador de CARLA Este análisis compara cualitativamente el desempeño de distintos modelos en el simulador CARLA, observando su comportamiento en situaciones de conducción. Se evalúan modelos ModifiedDeepestLSTM entrenados con dos enfoques distintos: Dataset Convencional (“Burbuja”) Dataset DAGGER 40% Además, se comparan tanto modelos monolíticos...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Pytorch","Expertos"],
        "url": "/weekly%20log/week38/",
        "teaser": null
      },{
        "title": "Week 39 - Prueba de Robustez para el modelo modifiedDeepestLSTM",
        "excerpt":"Prueba de Robustez Se evaluó la robustez del modelo ModifiedDeepestLSTM, comparando su desempeño cuando es entrenado con el dataset convencional “burbuja” y con DAGGER 40. Para ello, se realizaron cuatro pruebas en el escenario Town02, analizando el comportamiento de ambos modelos. Los resultados mostraron que el modelo entrenado con el...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Lane Detection","TensorFlow","expertos"],
        "url": "/weekly%20log/week39/",
        "teaser": null
      },{
        "title": "Week 40 - Escritura Paper",
        "excerpt":"Progreso en el Paper Se ha completado la sección de análisis cualitativo sobre la comparación de ModifiedDeepestLST utilizando el dataset Burbuja y el método Dagger 40. Además, se incorporó una revisión detallada sobre simuladores de robótica, destacando sus características, ventajas y aplicaciones en el contexto de la navegación autónoma. Avance...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","MoE"],
        "url": "/weekly%20log/week40/",
        "teaser": null
      },{
        "title": "Week 41 - behaviorMetrics adaptación para ROS 2",
        "excerpt":"Para adaptar BehaviourMetrics a ROS 2 se adoptó la estrategia de crear una única instancia compartida del nodo al inicio de la aplicación y luego pasar dicha instancia como parámetro a todos los componentes que necesiten interactuar con ROS (ya sea para publicar, suscribirse o consultar tópicos). Esto centraliza la gestión...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week41/",
        "teaser": null
      },{
        "title": "Week 42 - BehaviorMetrics prueba de inferencia y comunicación ROS \"",
        "excerpt":"Prueba con modelo monolítico en CARLA sin DAgger en BehaviorMetrics Se realizó una prueba utilizando un modelo monolítico previamente entrenado (Bubble CARLA Model, sin la técnica DAgger). Durante el experimento, se implementó exitosamente el procesamiento previo de imágenes, lo que permitió al modelo generar predicciones válidas. Estas predicciones fueron enviadas...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week42/",
        "teaser": null
      },{
        "title": "Week 43 - BehaviorMetrics - Lectura de ROSBAG ROS2",
        "excerpt":"Esta semana avanzamos en la ampliación del script analyze_bag.py para que, además de procesar ‘bags’ de ROS 1, sea capaz de leer y deserializar automáticamente las ROS 2 bags generadas en CARLA. Para ello: Integramos la biblioteca rosbag2_py y creamos un mecanismo de mapeo tópico→tipo de mensaje, lo que nos permite extraer...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week43/",
        "teaser": null
      },{
        "title": "Week 44 - BehaviorMetrics - Lectura de ROSBAG ROS2 - Métricas espaciales",
        "excerpt":"Error de lectura en métricas espaciales Durante esta semana se avanzó en la implementación del flujo de grabación y análisis de métricas espaciales en simulaciones de conducción autónoma utilizando CARLA y ROS 2. El sistema registra correctamente los datos de control, guarda archivos .db3 y .json, y el vehículo se...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week44/",
        "teaser": null
      },{
        "title": "Week 45- BehaviorMetrics - Lectura de ROSBAG ROS2 - Métricas espaciales",
        "excerpt":"Registro y Análisis de Métricas de Simulación Una vez finalizado el experimento en CARLA, se recuperan métricas espaciales y de eventos directamente desde los datos grabados en ROS 2 (.db3). El sistema genera dos archivos clave: Archivo .json con todas las métricas cuantitativas del recorrido, incluyendo distancia recorrida, velocidad promedio,...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week45/",
        "teaser": null
      },{
        "title": "Week 46- BehaviorMetrics - (Actualización)",
        "excerpt":"Se abrió un pull request en GitHub para incorporar compatibilidad con ROS 2 en el análisis offline de archivos rosbag dentro del script metrics_carla.py. Con esta mejora, al finalizar una simulación se generan automáticamente un archivo .json con las métricas del experimento y un .png con la visualización del recorrido,...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week46/",
        "teaser": null
      },{
        "title": "Week 47 - Comparación de modelos monolíticos con Behavior Metrics",
        "excerpt":"Comparación de modelos monolíticos con Behavior Metrics En esta etapa se entrenaron y evaluaron dos modelos monolíticos de conducción autónoma utilizando el simulador CARLA y la herramienta de evaluación Behavior Metrics: PilotNet (bubble_deepest_model.pth) ModifiedDeepestLSTM (dagger_deepest_model_7x5_bts16_4.pth) Ambos fueron entrenados sobre datos recolectados mediante la técnica DAgger, la cual permite iterativamente refinar...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/keek47/",
        "teaser": null
      },{
        "title": "Week 48 - Prubea inicial conn modelo ResNet18 y BehaviorMetrics",
        "excerpt":"En esta primera prueba con ResNet18, el modelo fue capaz de aprender una política básica de seguimiento de carril. Sin embargo, durante la simulación en tiempo real, el vehículo se movió lentamente y terminó colisionando. Esto sugiere que es necesario ajustar los hiperparámetros, especialmente en la predicción de throttle, y...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week48/",
        "teaser": null
      },{
        "title": "Week 49 - Entrenamiento Resnet18 y EfficienNet_v2_s",
        "excerpt":"Comparación de modelos: EfficientNet V2 S vs ResNet18 Métrica EfficientNet V2 S ResNet18 Arquitectura EfficientNet V2 S ResNet18 Dataset balanced_data_70.csv balanced_data_70.csv Batch size 8 64 Épocas 61 (early stop) 100 División Train/Val 80% / 20% 80% / 20% MSE final (val) ~0.0010 ~0.0007 Resolución entrada (640, 240) (640, 240) Número...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week49/",
        "teaser": null
      },{
        "title": "Week 50 - Entrenamiento Resnet18 y PilotNet",
        "excerpt":"Comparativa de PilotNet y ResNet18 con Dataset Balanceado Se entrenaron dos modelos de control para conducción autónoma, PilotNet y ResNet18, utilizando un dataset balanceado en los extremos de steer (dirección) y throttle (aceleración). El objetivo fue mejorar la capacidad de generalización en situaciones críticas como curvas cerradas o cambios bruscos...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week50/",
        "teaser": null
      },{
        "title": "Week 51 - Prueba de modelo PilotNet y DAgger",
        "excerpt":"Prueba rápida del modelo Primero se probó el modelo PilotNet usando una pequeña muestra del dataset (aproximadamente 5%) para verificar que todo funcionara correctamente. La pérdida bajó como se esperaba y las predicciones de steer y throttle fueron razonables. Entrenamiento completo: aparece el overfitting Luego se entrenó el modelo con...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week51/",
        "teaser": null
      },{
        "title": "Week 52 - Test Offline",
        "excerpt":"Resultados Modelo Comportamiento Observado PilotNet Muy estable. Navega siempre por el carril derecho. Predicciones conservadoras. ResNet18 Completa el circuito, pero varía entre carril derecho e izquierdo. Toma más giros. EfficientNet Con el mejor modelo (epoch_67), navega de forma estable por el carril derecho. Velocidad promedio 24 km/h, hasta 35 km/h en rectas....","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week52/",
        "teaser": null
      },{
        "title": "Week 53 - Control Manual",
        "excerpt":"En busca de una mayor representatividad en los datos, se decidió utilizar el control manual del simulador CARLA. Para ello, se incorporó un mando de PlayStation como herramienta de conducción. Aunque esto puede parecer más cómodo en comparación con el teclado de una PC, manejar el vehículo manualmente no siempre...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week53/",
        "teaser": null
      },{
        "title": "Week 54 - Control Manual - Entrenamiento y Prueba",
        "excerpt":"Resultados del Entrenamiento PilotNet en CARLA Se entrenó un modelo PilotNet usando datos recogidos manualmente en CARLA con un mando de PlayStation, lo cual permitió capturar datos más realistas y suaves, en contraste con el piloto automático. Configuración del Experimento Parámetro Valor Dataset Size 85,942 imágenes (segmentadas) Imagen Shape (66,...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week54/",
        "teaser": null
      },{
        "title": "Week 55 - Evaluación de Modelos de Conducción Autónoma en CARLA: Métricas MSE/MAE y Behavior Metrics",
        "excerpt":"Tabla resumen de pruebas (con métricas MSE/MAE) Prueba Comparativa Modelo Dataset utilizado MSE Steer MAE Steer MSE Throttle MAE Throttle 1 Monolítico vs. Mixture of Experts (MoE) PilotNet (monolítico) DAgger 0.0311 0.0847 0.1698 0.4010 1 Monolítico vs. Mixture of Experts (MoE) ResNet18 (monolítico) DAgger 0.0466 0.1237 0.1961 0.4078 1 Monolítico...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week55/",
        "teaser": null
      },{
        "title": "Week 56 - Test offline para modelos conducción manual",
        "excerpt":"Correción Post anterior Para esta segunda ejecución se mantuvieron idénticas todas las condiciones de la simulación (mapa Town02, clima, vehículo Tesla Model 3, punto de partida y modelo DAgger), modificando únicamente dos parámetros en el archivo de configuración de Behavior Metrics: Parámetro Valor anterior Nuevo valor Efecto PilotTimeCycle 40 ms...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week56/",
        "teaser": null
      },{
        "title": "Week 57 - Análisis Cualitativo de Robustez",
        "excerpt":"Se utilizó Town02 como modelo entrenado, empleando el dataset obtenido a partir de la conducción manual. El modelo autónomo utiliza el piloto automático integrado mediante el ROS bridge. Su comportamiento es considerado ideal, ya que se basa en un controlador de trayectoria PID fundamentado en waypoints, lo cual facilita la...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week57/",
        "teaser": null
      },{
        "title": "Week 58 - Sincronización de tópicos",
        "excerpt":"Estadísticas de la inferencia Métrica Valor Aprox. Promedio ~3.6 ms Mediana ~3.1 ms Máximo ~8.1 ms Desviación estándar ~1.1 ms Posible causa de fallos En otros experimentos, se observaron picos de latencia como: Inferencia máxima: 713 ms Esto causó desincronización: los ticks se acumularon y el control se retrasó. ¿Qué...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week58/",
        "teaser": null
      },{
        "title": "Week 59 - Comparativa de modelos para conducción autónoma en CARLA",
        "excerpt":"Tiempo de inferencia promedio (usando API Python CARLA) Tiempo de inferencia promedio (usando API Python CARLA) Este indicador refleja el tiempo que tarda cada modelo en realizar una inferencia (predicción) sobre una imagen. Se evaluó directamente sobre la API de CARLA con los modelos torch *.pth para dataset teleoperado. Modelo...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week59/",
        "teaser": null
      },{
        "title": "Week 60 - PilotNet, MoE y Monolítico (CARLA)",
        "excerpt":"Resultados de comparación: Burbuja (autopiloto) vs Control Manual (teleoperado) usando modelo PilotNet Métrica Burbuja (Autopiloto) Control Manual (Humano) Distancia completada (m) 779.06 616.84 Distancia efectiva (m) 277.00 373.00 Velocidad promedio (km/h) 16.94 29.95 Velocidad máxima (km/h) 22.73 38.49 Velocidad mínima (km/h) 7.14 -0.72 Desviación posición media (m) 1.11 0.87 Desviación...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python","BehaviorMetrics"],
        "url": "/weekly%20log/week60/",
        "teaser": null
      },{
        "title": "Week 61 - Dataset, Autopilot vs. Teleoperado",
        "excerpt":"Al comparar los datasets, se observa que el autopilot (burbuja) genera valores de throttle muy repetitivos y discretos, casi en modo ON/OFF. En cambio, el teleoperado (control manual) muestra una distribución más amplia y transiciones suaves, propias de la conducción humana. Esto significa que el teleoperado aporta mayor variabilidad y...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week61/",
        "teaser": null
      },{
        "title": "Week 62 - Dataset teleoperado + DAgger",
        "excerpt":"Construcción del dataset y entrenamiento del modelo Se construyó un nuevo dataset, combinando la estrategia de teleoperación con la técnica DAgger (Dataset Aggregation). Esta última se implementó introduciendo perturbaciones aleatorias controladas sobre el vehículo (ruido gaussiano en dirección, aceleración y freno), lo que obliga al modelo a aprender a recuperarse...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week62/",
        "teaser": null
      },{
        "title": "Week 63 - Test offline, MoE (teleoperado) vs Moe (teleoperado + DAgger)",
        "excerpt":"Métrica Dagger (Teleoperado + DAgger) MC (Solo teleoperado) Δ (MC − Dagger) Δ% vs MC RMSE Steer 0.275 0.521 0.246 47.24% MAE Steer 0.163 0.477 0.314 65.77% RMSE Throttle 0.575 0.492 −0.084 −17.06% MAE Throttle 0.510 0.481 −0.029 −6.10% Nota (fórmulas) Col. 4 — Δ (MC − Dagger): Δ =...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week63/",
        "teaser": null
      },{
        "title": "Week 64 - Problemas en la obtención de métricas",
        "excerpt":"Resumen de incidencias en las pruebas con BehaviorMetrics y CARLA Durante las últimas pruebas con BehaviorMetrics sobre CARLA en modo ROS 2, se han identificado varias dificultades que impiden completar los experimentos de forma estable: Obtención de métricas incompleta El simulador se cierra de manera abrupta antes de finalizar los...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week64/",
        "teaser": null
      },{
        "title": "Week 65 - Corrección del problemas en la obtención de métricas",
        "excerpt":"En las pruebas recientes con CARLA + BehaviorMetrics (ROS 2) se presentaron varios problemas relacionados con la sincronización del simulador y la estabilidad de ejecución. 1. Problemas encontrados Bloqueo por sincronía Al lanzar Town02 en modo asíncrono, pero manteniendo el mundo en synchronous y el TrafficManager en espera de ticks,...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week65/",
        "teaser": null
      },{
        "title": "Week 66 - Dataset DAgger Teleoperado con estrategia agresiva",
        "excerpt":"Dataset DAgger + Teleoperado(Estrategia agresiva) Se generó un nuevo dataset con la técnica DAgger + teleoperación, esta vez con una agresividad mayor en la aceleración durante las curvas: se entra a una velocidad suficientemente alta y se mantiene o incrementa el throttle mientras se ejecuta la maniobra. Durante la corrección...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week66/",
        "teaser": null
      },{
        "title": "Week 67 - Monolítico vs expertos dataset teleoperado burbuja",
        "excerpt":"Métrica Monolítico MoE / Expertos completed_distance (m) ↑ 753.215 758.858 effective_completed_distance (m) ↑ 307.0 216.0 average_speed (m/s) ↑ 50.180 51.754 max_speed (m/s) 71.831 76.919 min_speed (m/s) → 0 mejor -6.51e-08 -5.91e-08 suddenness_distance_speed (Σdist) ↓ 0.2679 0.5268 suddenness_distance_control_commands (Σdist) ↓ 0.1197 0.1596 suddenness_distance_throttle (Σdist) ↓ 0.0970 0.1489 suddenness_distance_steer (Σdist) ↓ 0.0492...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week67/",
        "teaser": null
      },{
        "title": "Week 68 - soporte de BehaviorMetrics para PythonAPI",
        "excerpt":"Se realizaron modificaciones en BehaviorMetrics para permitir la ejecución directa sobre CARLA mediante la Python API, sin requerir ROS 1 ni ROS 2. Principales cambios: Adaptación de environment.py para lanzar CarlaUE4.sh y carla_world_generator.py automáticamente. Nuevo modo de operación detectado por ROS_VERSION=None. Actualización de controller_carla.py, pilot_carla.py y driver_carla.py para trabajar sin...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week68/",
        "teaser": null
      },{
        "title": "Week 69 - Mejoras en soporte de BehaviorMetrics para PythonAPI",
        "excerpt":"Estoy haciendo una revisión sobre el uso de Mixture of Experts (MoE) en conducción autónoma. Los papers que he leído muestran mejoras claras en seguridad, generalización y eficiencia, así que todo apunta a que esta arquitectura es una opción muy prometedora para sistemas de control y percepción en vehículos autónomos....","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week69/",
        "teaser": null
      },{
        "title": "Week 70 - Mejora del proceso de generación del dataset",
        "excerpt":"Mejora del proceso de generación del dataset Se optimizó el script de creación del dataset para resolver los problemas de desfase temporal que se presentaban al etiquetar imágenes en tiempo real, debido a la latencia en el guardado de cada fotograma. El nuevo flujo busca mejorar la sincronización entre las...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week70/",
        "teaser": null
      },{
        "title": "Week 71 - Reconstrucción del dataset y nuevo pipeline de sincronización",
        "excerpt":"Reconstrucción del dataset y nuevo pipeline de sincronización Esta semana reconstruí completamente el dataset para el modelo de conducción. Grabé escenarios variados en CARLA y dejé Town02 solo para test, para asegurar una prueba “real” sin contaminación del entrenamiento. Con la Python API generé unas 87.000 imágenes RGB y segmentadas....","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","ROS 2","Python"],
        "url": "/weekly%20log/week71/",
        "teaser": null
      },{
        "title": "Week 72 - Prueba y entrenamiento dataset (CM-CARLA-SyncDriving-RGB-Dataset)",
        "excerpt":"Sincronización del dataset Se corrigió el desfase temporal entre imágenes y comandos usando el flujo: Recorder → captura del estado completo de la simulación Replay → regeneración determinista de las imágenes Con esto se logró un dataset perfectamente alineado imagen–comando. Incorporación de imágenes RGB Además de segmentación, ahora se integran...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","Python","DAgger","Teleoperado"],
        "url": "/weekly%20log/week72/",
        "teaser": null
      },{
        "title": "Week 73 -  Prueba dataset Daniel",
        "excerpt":"Comparación de Métricas: ResNet-18 (30 épocas) vs MobileNet Se probaron ambos modelos utilizando el dataset de Daniel dentro de mi propio pipeline de entrenamiento. Las métricas mejoran de manera significativa respecto a ejecuciones anteriores, lo que indica que el flujo de entrenamiento está funcionando correctamente. Sin embargo, en pruebas online...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","Python","DAgger","Teleoperado"],
        "url": "/weekly%20log/week73/",
        "teaser": null
      },{
        "title": "Week 74 -  Balanceo estratificado (train, test offline y online)",
        "excerpt":"Balanceo del Dataset, Segmentación y Entrenamiento de Modelos DL Con el objetivo de mejorar el desempeño en conducción autónoma dentro del simulador CARLA, se rediseñó completamente el proceso de dataset y entrenamiento. Estratificación y Balanceo 5×4 (Steer–Throttle) Para evitar que el modelo aprendiera sesgos a conducir recto o a acelerar...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","Python","DAgger","Teleoperado"],
        "url": "/weekly%20log/week74/",
        "teaser": null
      },{
        "title": "Week 75 -  Burbuja vs DAgger(train, test offline y online)",
        "excerpt":"Tabla resumen – Métricas OFFLINE Modelos entrenados con volante (teleoperado) Comparación DAgger vs Burbuja Estrategia Steer MAE Steer RMSE Throttle MAE Throttle RMSE DAgger 0.0428 0.0883 0.0817 0.1546 Burbuja 0.0270 0.0456 0.0741 0.1327 ** Resultados preliminares: DAgger vs. Burbuja** En esta etapa del trabajo se esperaba que la estrategia DAgger...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","Python","DAgger","Teleoperado"],
        "url": "/weekly%20log/week75/",
        "teaser": null
      },{
        "title": "Week 76 -  Metricas de robustez por casos",
        "excerpt":"Caso Entrenamiento Distancia completada (m) Distancia efectiva (m) Desv. pos. media (m) Desv. pos. / km Colisiones Invasiones carril Suavidad ctrl / km 1 Burbuja 30.22 7.5 2.91 388.28 0 141 2.85 1 DAgger 31.52 6.5 1.83 280.87 0 146 2.00 2 Burbuja 192.26 44.0 2.33 52.96 303 497 0.31...","categories": ["weekly Log"],
        "tags": ["CARLA Simulator","Python","DAgger","Teleoperado"],
        "url": "/weekly%20log/week76/",
        "teaser": null
      },{
        "title": "Week 77 - Metricas de robustez por casos",
        "excerpt":"CASO CANÓNICO: Burbuja vs DAgger Caso Modelo Completed Distance (m) Collisions Lane Invasions Pos. Dev. Mean (m) Lane Inv./km C DAgger 767.99 995 1223 1.31 6224   Burbuja 761.69 31 1065 1.06 4819 Ambos modelos completan prácticamente la misma distancia. *Burbuja muestra una conducción más estable y segura. DAgger avanza...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week77/",
        "teaser": null
      },{
        "title": "Week 78 - Metricas de robustez por casos (burbuja + DAgger, 68/31)",
        "excerpt":"CASO CANÓNICO Modelo Completed Distance (m) Collisions Lane Invasions Pos. Dev. Mean (m) Lane Inv./km Burbuja + DAgger (45k + 15.6k) 753.36 0 1257 0.91 3416 Burbuja 45k 755.97 0 1325 1.00 4274 DAgger mejora la estabilidad lateral sin penalizar seguridad ni progreso en trayectos largos y continuos. POSICIONES DIFÍCILES...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week78/",
        "teaser": null
      },{
        "title": "Week 80 - Metricas de robustez por casos (burbuja + DAgger, 68/32)",
        "excerpt":"Caso canónico — Comparación Burbuja vs DAgger (burbuja 68% + 32% DAgger puro) Condiciones de la prueba Comparación de robustez entre modelos Burbuja y DAgger (EfficientNet-V2-S) Mismo mapa: Town02 Mismo punto inicial y final Circuito largo Condiciones equivalentes de simulación e inferencia Tabla resumen de métricas (Caso canónico) Métrica Burbuja...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week80/",
        "teaser": null
      },{
        "title": "Week 81 - Métrica *reward* (centrado a la calzada)",
        "excerpt":"Métrica reward (centrado a la calzada) — Validación fuera de BehaviorMetrics Antes de integrar la métrica de reward en BehaviorMetrics, primero se implementó y depuró por fuera del framework, usando directamente la API de Python de CARLA. El objetivo fue iterar más rápido: aislar el cálculo del reward, validar parámetros...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week81/",
        "teaser": null
      },{
        "title": "Week 82 - Experimento Burbuja vs DAgger (Town02, Efficientnet_v2-s, repeticiones consecutivas)",
        "excerpt":"Tabla resumen — Burbuja Run Completed distance (m) Effective distance (m) Avg speed (km/h) Pos. dev. mean (m) Lane invasions Collisions Reward mean Reward sum Offroad frames 1 758.20 277.0 64.22 1.039 1101 0 0.554 609.68 244 2 757.76 218.0 64.46 1.032 935 0 0.552 516.53 208 3 764.10 161.5...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week82/",
        "teaser": null
      },{
        "title": "Week 83 - Actualización Dataset Burbuja + DAgger",
        "excerpt":"Durante esta semana se construyó un nuevo dataset de Burbuja y un dataset mixto Burbuja + DAgger, y se reentrenó el modelo utilizando EfficientNet, ajustando el objetivo de control para seguir exclusivamente el carril derecho (a diferencia de experimentos previos centrados en el seguimiento general de la calzada). El dataset...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week83/",
        "teaser": null
      },{
        "title": "Week 84 - Noise Injection - Pruebas robustez",
        "excerpt":"Robustez en Conducción Autónoma Comparación: Dataset Burbuja vs Noise Injection Contexto Este estudio evalúa el impacto de incorporar Noise Injection en el dataset de entrenamiento de un modelo end-to-end de conducción autónoma en CARLA. Se analiza: Comportamiento nominal Capacidad de recuperación Robustez fuera de distribución Estabilidad dinámica a diferentes velocidades...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/keek84/",
        "teaser": null
      },{
        "title": "Week 85 - Noise Injection vs DAgger, pruebas de robustez y estabilidad en CARLA (PilotNet)",
        "excerpt":"Configuración Experimental Todos los experimentos fueron realizados en el simulador CARLA utilizando una arquitectura PilotNet modificada, entrenada para predecir comandos de dirección (steer) y aceleración (throttle). Se evaluaron cuatro estrategias de entrenamiento: Burbuja (Baseline): Dataset de 50k muestras de conducción humana natural. Noise Augmentation: Dataset Burbuja con perturbaciones artificiales controladas....","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week85/",
        "teaser": null
      },{
        "title": "Week 86 - Actualización de métricas en CARLA",
        "excerpt":"Se realizaron mejoras en la forma de reportar el desempeño del vehículo autónomo en simulación, con el objetivo de que las métricas reflejen con mayor fidelidad el comportamiento real durante la ejecución. Estas modificaciones aplican tanto al flujo basado en ROS como al flujo usando Python API, permitiendo comparar resultados...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week86/",
        "teaser": null
      },{
        "title": "Week 87 - Actualización de métricas en CARLA",
        "excerpt":"En esta semana consolidé el reporte comparativo de cinco variantes de entrenamiento (Burbuja, Noise injection, DAgger 16k, DAgger 27.6k y DAgger agresivo) usando las tablas resumen por escenario. El objetivo fue contrastar desempeño nominal y robustez bajo escenarios de perturbación y, adicionalmente, evaluar estabilidad al variar la velocidad objetivo. Notas...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week87/",
        "teaser": null
      },{
        "title": "Week 88 - Actualización de métricas en BehaviorMetrics (CARLA), comparación contra ruta ideal",
        "excerpt":"Actualización de métricas en BehaviorMetrics (CARLA) En esta actualización se modificó el sistema de métricas en BehaviorMetrics para CARLA con el objetivo de hacer la evaluación más robusta, reproducible y coherente entre distintos modelos. El cambio principal consiste en dejar de medir el progreso únicamente con heurísticas basadas en el...","categories": ["weekly Log"],
        "tags": ["Robustez","DAgger","Metrics Online","BehaviorMetrics"],
        "url": "/weekly%20log/week88/",
        "teaser": null
      },{
        "title": "Week 89 - Robustez del piloto, Caso Canónico, 15 Casos y Pruebas de Velocidad",
        "excerpt":"En esta entrada reporto los resultados de robustez en CARLA Town02 comparando cinco políticas/datasets bajo el mismo circuito de prueba, con 6 repeticiones por condición (valores reportados como promedio de las repeticiones). Circuito base Caso canónico: recorrido siguiendo la malla externa (outer loop) de Town02, manteniéndose en el carril derecho....","categories": ["weekly Log"],
        "tags": ["Robustez","Burbuja","Noise Augmentation","DAgger","CARLA","Town02"],
        "url": "/weekly%20log/week89/",
        "teaser": null
      }]
