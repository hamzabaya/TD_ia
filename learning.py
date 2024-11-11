import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
  
    best_next_action = np.argmax(Q[sprime])
    # Équation de Bellman pour la mise à jour
    Q[s, a] += alpha * (r + gamma * Q[sprime, best_next_action] - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
   
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        return np.argmax(Q[s])  # Exploitation

if __name__ == "__main__":
    # Initialisation de l'environnement
    env = gym.make("Taxi-v3", render_mode="human")
    env.reset()
    env.render()

    # Initialisation de la Q-table avec des zéros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparamètres
    alpha = 0.01  # Taux d'apprentissage
    gamma = 0.8   # Facteur de discount
    epsilon = 0.2 # Taux d'exploration

    n_epochs = 200
    max_itr_per_epoch = 300
    rewards = []

    # Boucle d'entraînement
    for e in range(n_epochs):
        total_reward = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            # Choisir une action selon la stratégie epsilon-greedy
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)
            
            # Exécuter l'action et observer le résultat
            Sprime, R, done, _, _ = env.step(A)

            # Mise à jour de la Q-table
            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Mise à jour de l'état et cumul des récompenses
            S = Sprime
            total_reward += R

            if done:
                break

        print(f"Épisode #{e} : Récompense totale = {total_reward}")
        rewards.append(total_reward)

    print("Récompense moyenne =", np.mean(rewards))
    print("Entraînement terminé.")

    # Tracer les récompenses en fonction des épisodes
    plt.plot(rewards)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.title("Récompenses au fil des épisodes")
    plt.show()

    # Fermeture de l'environnement après l'entraînement
    env.close()
