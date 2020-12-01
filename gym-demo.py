#Importamos las librerias necesarias para el ejemplo Gym, Numpy y keras que incluye tensorflow
import gym #la libreria de los propios juegos que usaremos para entrenar
import random #Numeros Aleatorios
import numpy as np #Matematico
from keras.models import Sequential #Modelo de Keras
from keras.layers import Dense #Capas de Keras
from keras.optimizers import Adam #Optimizacion de Keras

env = gym.make('CartPole-v1') #creamos nuestro entorno de trabajo el juego que simularemos aunque no sea de atari cunple con el objetivo
env.reset()
goal_steps = 500 #definimos el numero de pasos para el entrenamiento, los episodios que tendra para entrenar mientras mas episodios mejor jugara la AI 
score_requirement = 60 #puntuacion requerida para que pase el siguiente episodio 
initial_games = 10000 #Entrenamiento inicial

#Función que ejecuta un bucle para hacer varias acciones para jugar el 
#juego.Por eso, intentar jugaremos hasta 500 pasos como máximo como anteriormente especificamos y donde podriamos cambiar la cifra
def play_a_random_game_first():
    try:
        for step_index in range(goal_steps):
            #env.render() #Para representar el juego
            action = env.action_space.sample() #Elegimos acción al azar
            #Acción aleatoria a través de la función que elige los 
            #los resultado del siguiente paso, según la acción pasada como
            #parametro
            observation, reward, done, info = env.step(action)
            print("Paso {}:".format(step_index)) #el paso del que va del paso 1 al paso 500
            print("Acción: {}".format(action)) #la accion que realizo en dicho paso/episodio
            print("Observacion: {}".format(observation)) #las observaciones que tuvo
            print("Recompensa: {}".format(reward))#recompensa dada o negada
            print("Done: {}".format(done))#verdadero o falso si termino de entrenar los 500 pasos
            print("Info: {}".format(info))#la nformacion que recolecto
            if done:#Si juego completado
                break
    finally:
        env.reset()

play_a_random_game_first()
#Preparando datos de Entrenamiento
def model_data_preparation():
    #inicializamos los arrays con los datos de entrenamiento y las puntuaciones que llevara la AI
    training_data = []   
    accepted_scores = [] 
    #Jugamos 10000 veces para obtener unos datos representativos suficientes para que jugue decentemente
    for game_index in range(intial_games):
        score = 0 #inicializamos variables
        game_memory = []
        previous_observation = []
        #inidicamos que se ejeccute 500 veces
        for step_index in range(goal_steps):
            action = random.randrange(0, 2)#Acción aleatoria.Iz=0 y De=1
            observation, reward, done, info = env.step(action)
            #almacenamos puntuacion
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])
        
        #resteamos entorno y lo mostramos por pantalla
        env.reset()

    print(accepted_scores)
    
    return training_data

training_data = model_data_preparation()

#Con esta función contruimos nuestros mmodelo.Nuestra red neuronal 
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model
 
#creamos la función que entrenará nuestro modelo
def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    
    model.fit(X, y, epochs=10)
    return model
 
trained_model = train_model(training_data)


scores = []#inicializamos puntuaciones y array de elecciones
choices = []
for each_game in range(1):#jugamos 100 partidas
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):#Jugamos 500 pasos por partida
        
        env.render()
        if len(prev_obs)==0:
            action = random.randrange(0,2)#en el primer paso elegimos movimiento al azar
        else:
            #A partir del segundo paso conocemos el estado actual del juego.
            #Entonces, tomaremos esa observación y se la daremos a nuestro 
            #modelo para predecir qué acción debemos tomar. 
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        #guardamos elección
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))