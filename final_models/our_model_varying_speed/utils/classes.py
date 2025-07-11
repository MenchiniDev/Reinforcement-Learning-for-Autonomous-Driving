import numpy as np
from qqdm import qqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.image as mpimg
from collections import deque
import random
import os, time

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple

import random, math

import torch
from torch.linalg import solve as solve_matrix_system

import itertools
from multiprocessing import Pool, cpu_count

import sys


######## SET PARALLEL COMPUTING ##########
num_cores = cpu_count()

torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
##########################################


######## SET DEVICE ######################
# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     device = "cpu"
device = "cpu"
#########################################


######### SET SEED ######################
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
#########################################


################# SPEED UP ####################
torch.autograd.set_detect_anomaly(False);
torch.autograd.profiler.emit_nvtx(False);
torch.autograd.profiler.profile(False);
################################################


VectorScore = namedtuple('VectorScore', ('collision', 'distance', 'oob'))

class SequentialReplayBuffer:
    def __init__(self, capacity, seq_len, obs_dim, action_dim, reward_dim, device='cpu'):
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        # Modifica per memorizzare episodi completi
        self.episodes = deque(maxlen=capacity // seq_len) # Memorizza un certo numero di episodi
        self.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        self.total_transitions = 0 # Contatore per il numero totale di transizioni

    def add(self, state, action, reward, done):
        """Aggiunge una singola transizione all'episodio corrente."""
        self.current_episode['states'].append(torch.tensor(state, dtype=torch.float32))
        self.current_episode['actions'].append(torch.tensor(action, dtype=torch.long))
        self.current_episode['rewards'].append(torch.tensor(reward, dtype=torch.float32))
        self.current_episode['dones'].append(torch.tensor(done, dtype=torch.bool))
        self.total_transitions += 1

        if done:
            # Calcola RTG per l'episodio completato
            # Per il DT, di solito l'RTG è la somma dei reward futuri.
            # Qui usiamo la somma semplice dei reward dell'episodio.
            # Se hai un reward vettoriale, dovrai decidere come sommarlo per l'RTG.
            # Per semplicità, useremo la somma del primo componente del reward.
            episode_rewards = torch.stack(self.current_episode['rewards']) # (T, reward_dim)
            rtgs = []
            current_rtg = 0
            # Calcola RTG all'indietro
            for r_vec in reversed(episode_rewards):
                current_rtg = r_vec[0] + current_rtg # Usiamo il primo componente del reward
                rtgs.insert(0, current_rtg)
            self.current_episode['rtgs'] = torch.stack(rtgs).unsqueeze(-1) # (T, 1)

            self.episodes.append(self.current_episode)
            self.current_episode = { # Reset per il prossimo episodio
                'states': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }


    def can_sample(self, batch_size):
        # Assicurati di avere abbastanza episodi e che ogni episodio abbia almeno seq_len transizioni
        return len(self.episodes) >= batch_size and \
               all(len(ep['states']) >= self.seq_len for ep in self.episodes)

    def sample(self, batch_size):
        """
        Estrae batch di sequenze (rtgs, states, actions) dalla memoria.
        Ogni sequenza ha lunghezza `self.seq_len`.
        """
        batch_rtgs = []
        batch_states = []
        batch_actions = []

        # Seleziona casualmente `batch_size` episodi
        selected_episodes = random.sample(self.episodes, batch_size)

        for ep in selected_episodes:
            episode_len = len(ep['states'])
            # Seleziona un punto di inizio casuale all'interno dell'episodio
            # Assicurati che ci siano abbastanza passi rimanenti per formare una sequenza di seq_len
            start_idx = random.randint(0, episode_len - self.seq_len)
            end_idx = start_idx + self.seq_len

            # Estrai la sottosequenza
            sub_rtgs = ep['rtgs'][start_idx:end_idx].to(self.device)
            sub_states = torch.stack(ep['states'][start_idx:end_idx]).to(self.device)
            sub_actions = torch.stack(ep['actions'][start_idx:end_idx]).to(self.device)

            batch_rtgs.append(sub_rtgs)
            batch_states.append(sub_states)
            batch_actions.append(sub_actions)

        return (
            torch.stack(batch_rtgs),     # (batch, seq_len, 1)
            torch.stack(batch_states),   # (batch, seq_len, obs_dim)
            torch.stack(batch_actions)   # (batch, seq_len)
        )
    
    def __len__(self):
        return self.total_transitions # Conta il numero totale di transizioni aggiunte



def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


class Q_Network(nn.Module):

    def __init__(self, n_observations, hidden = 128, weights = None):
        
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.layer4 = nn.Linear(hidden, hidden)

        self.input_norm = nn.BatchNorm1d(n_observations)
        self.norm = nn.BatchNorm1d(hidden)

        self.criterion = nn.SmoothL1Loss()
        #self.optimizer = optim.AdamW(self.parameters(), lr = learning_rate, amsgrad=True)
        self.activation = F.relu

        self.weights = weights


    def instantiate_optimizer(self, learning_rate = 1e-4):
        self.optimizer = optim.SGD(self.parameters(), lr = learning_rate, momentum = 0.9, nesterov = True)

    
    def common_forward(self, x):

        x = self.input_norm(x)
        
        z = self.activation(self.layer1(x))
        
        y = self.activation(self.layer2(z))

        y = self.activation(self.layer3(y))

        z = z + self.activation(self.layer4(y))
        
        z = self.norm(z)

        return z
    


class Lex_Q_Network(Q_Network):

    def __init__(self, n_observations, n_actions, hidden = 128, learning_rate = 1e-4, weights = None):
        
        super().__init__(n_observations, hidden, weights)

        self.layerb1 = nn.Linear(hidden, n_actions)
        self.layerb2 = nn.Linear(hidden, n_actions)
        self.layerb3 = nn.Linear(hidden, n_actions)

        self.I = torch.eye(hidden, device = device)
        self.I_o1 = torch.eye(self.layerb1.out_features, device = device)
        self.I_o2 = torch.eye(self.layerb1.out_features + self.layerb2.out_features, device = device)

        self.instantiate_optimizer(learning_rate)


    def forward(self, x):

        z = self.common_forward(x)

        ort2, ort3, prj2, prj3 = self.project(z)

        o1 = F.sigmoid(self.layerb1(z)) - 1

        o2 = self.layerb2(ort2 + prj2)

        o3 = self.layerb3(ort3 + prj3)

        return torch.stack((o1, o2, o3), dim = 2)
    
    # Assumption: W is column full rank. 
    def project(self, z): # https://math.stackexchange.com/questions/4021915/projection-orthogonal-to-two-vectors

        W1 = self.layerb1.weight.clone().detach()
        W2 = self.layerb2.weight.clone().detach()
        ort2 = torch.empty_like(z)
        ort3 = torch.empty_like(z)
        
        zz = z.clone().detach()

        #mask = torch.heaviside(zz, self.v)
        #Rk = torch.einsum('ij, jh -> ijh', mask, self.I)
        #W1k = W1.matmul(Rk)
        #W2k_ = W2.matmul(Rk)
        #W2k = torch.cat((W1k, W2k_), dim = 1)
        W2k = torch.cat((W1, W2), dim = 0)

        ort2 = self.compute_orthogonal(z, W1, self.I_o1)
        ort3 = self.compute_orthogonal(z, W2k, self.I_o2)

        self.ort2 = ort2.clone().detach()
        self.ort3 = ort3.clone().detach()

        prj2 = zz - self.ort2
        prj3 = zz - self.ort3

        return ort2, ort3, prj2, prj3


    def compute_orthogonal(self, z, W, I_o):
        WWT = torch.matmul(W, W.mT)
        P = solve_matrix_system(WWT, I_o)
        P = torch.matmul(P, W)
        P = self.I - torch.matmul(W.mT, P)
        
        return torch.matmul(z, P)
        
    
    def learn(self, predict, target):
        self.optimizer.zero_grad()
        loss_f = self.criterion(predict[:,0], target[:,0])
        loss_i1 = self.criterion(predict[:,1], target[:,1])
        loss_i2 = self.criterion(predict[:, 2], target[:,2])
        
        loss_f.backward(retain_graph=True)
        loss_i1.backward(retain_graph=True)
        loss_i2.backward()

        self.optimizer.step()

        #return torch.tensor([loss_f, loss_i1, loss_i2]).clone().detach()
        

    def  __str__(self):
        return "Lex"




class Car:
    def __init__(self, position):
        # distanze dal centro delle route anteriori e posteriori
        self.lf = 1
        self.lr = 1

        self.max_speed = 20.0
        #self.prev_position = np.zeros(2)

        self.reset(position)


    def move(self, df, a):
        #velocità
        self.v += a
        
        self.v = np.maximum(-self.max_speed, np.minimum(self.max_speed, self.v)) # speed saturation
        
        #phi e besta sono angoli di sterzata e angolo di sterzata effettivo
        #self.beta = np.arctan((1 + self.lr / self.lf) * np.tan(df))
        self.beta += np.arctan((1 + self.lr / self.lf) * np.tan(df))

        arg = self.phi + self.beta

        self.prev_position = np.copy(self.position)
        self.position += self.v * array([np.cos(arg), np.sin(arg)])

        #self.phi += self.v/self.lr * np.sin(self.beta)     ## Stearing inertia disabled
        #ang = array([np.cos(self.phi), np.sin(self.phi)])

        ang = array([np.cos(self.beta), np.sin(self.beta)])

        self.prev_front = self.front
        self.front = self.position + self.lf * ang

        #self.prev_back = self.back
        self.back = self.position - self.lr * ang

    # azzera la posizione del veicolo e le velocità
    def reset(self, position):
        self.position = position
        self.prev_position = np.copy(position)

        self.v = 0
        self.phi = 0
        self.beta = 0

        self.front = array([self.position[0] + self.lf, self.position[1]])
        self.back = array([self.position[0], self.position[1] + self.lr])

        self.prev_front = np.copy(self.front)
        #self.prev_back = np.copy(self.back)



class Jaywalker:

    def __init__(self):

        # MODIFICHE MOVIMENTO JAYWALKER
        self.min_j_speed = 0.1
        self.max_j_speed = 0.5
        self.jaywalker_speed = 0.0 
        self.jaywalker_dir = 1 
        
        #modifiche per curriculum
        self.curriculum_stage = 1

        #modifiche per rallebntamento avversario
        self.completed_mean = 0.0

        if not hasattr(self, 'car_img'):
            self.car_img = mpimg.imread("../.car/carontop.png")  # ← metti il file nella stessa cartella


        self.reward_size = 3

        self.dim_x = 120
        self.dim_y = 10

        #modifiche per aggiunta dinamica di ostacoli
        self.n_lanes = 2
        self.lane_width = self.dim_y / self.n_lanes
        self.lanes_y = [self.lane_width/2 + i * self.lane_width for i in range(self.n_lanes)]
        self.obstacles = []     # verrà popolato in reset()
        self.max_obstacles = 1  # ad es., fino a 5 oggetti

        self.goal = array([self.dim_x, self.dim_y/4])

        self.jaywalker = array([self.dim_x/2, self.dim_y/4])
        self.jaywalker_r = 2

        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r

        #self.df = [-np.pi/9, -np.pi/18, 0, np.pi/18, np.pi/9]
        self.df = [-np.pi/18, 0, np.pi/18]
        self.a = [-2, 0, 2]
        self.actions = [p for p in itertools.product(self.df, self.a)]

        self.num_df_actions = len(self.df)
        self.num_a_actions = len(self.a)

        self.state_size = 8
        self.action_size = self.num_df_actions * self.num_a_actions #9 actions

        self.max_iterations = 15000

        self.car = Car(array([0.0,2.5]))

        self.counter_iterations = 0

        #self.prev_center_disance = np.abs(self.car.position[1] - self.goal[1])
        self.prev_target_distance = np.linalg.norm(self.car.front - self.goal)

        self.noise = 1e-5
        self.sight = 60
        self.sight_obstacle = 80

        self.scale_factor = 100


    def collision_with_jaywalker(self):
        front = np.maximum(self.car.front, self.car.prev_front)
        prev = np.minimum(self.car.front, self.car.prev_front)

        denom = front - prev + self.noise

        upper = (self.jaywalker_max - prev) / denom
        lower = (self.jaywalker_min - prev) / denom

        scalar_upper = np.min(upper)
        scalar_lower = np.max(lower)
        
        if scalar_upper >= 0 and scalar_lower <= 1 and scalar_lower <= scalar_upper:
                return True
        
        return False


    def collision_with_obstacle(self):
        car_r = 2.0  # same as jaywalker radius
        car_front_max = self.car.front + car_r
        car_front_min = self.car.front - car_r

        car_prev_front_max = self.car.prev_front + car_r
        car_prev_front_min = self.car.prev_front - car_r

        for obs in self.obstacles:
            obs_max = obs['pos'] + obs['r']
            obs_min = obs['pos'] - obs['r']

            front = np.maximum(car_front_max, car_prev_front_max)
            prev = np.minimum(car_front_min, car_prev_front_min)

            denom = front - prev + self.noise

            upper = (obs_max - prev) / denom
            lower = (obs_min - prev) / denom

            scalar_upper = np.min(upper)
            scalar_lower = np.max(lower)

            if scalar_upper >= 0 and scalar_lower <= 1 and scalar_lower <= scalar_upper:
                return True

        return False


    # return the inverse of the distance from the jaywalker and the angle w.r.t to it
    def vision(self):
        vector_to_jaywalker = self.jaywalker - self.car.position + self.noise
        distance = np.linalg.norm(vector_to_jaywalker)

        if self.car.position[0] >= self.jaywalker[0] or distance > self.sight:
            return 0, -np.pi, float('inf')

        angle = np.arctan2(vector_to_jaywalker[1], vector_to_jaywalker[0])
        inv_distance = 1 / distance

        return inv_distance, angle, distance


    def vision_obstacle(self):
        if not self.obstacles:
            return 0, -np.pi

        dists = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
        i_min = np.argmin(dists)
        obs = self.obstacles[i_min]

        vector_to_obs = obs['pos'] - self.car.position + self.noise
        distance = np.linalg.norm(vector_to_obs)

        if distance > self.sight_obstacle:
            return 0, -np.pi

        ref_angle = np.radians(15)  # inclined 15° left from forward
        angle_to_obs = np.arctan2(vector_to_obs[1], vector_to_obs[0])
        angle = angle_to_obs - ref_angle

        if np.abs(angle) > np.pi / 2:
            return 0, -np.pi  # outside 90° cone

        inv_distance = 1 / distance
        return inv_distance, angle



    def step(self, action):

        # MODIFICHE PER MOVIMENTO JAYWALKER
        self.jaywalker[1] += self.jaywalker_speed * self.jaywalker_dir
        # -- se il pedone esce dalla corsia sbuca da sotto  
        if self.jaywalker[1] < 0 or self.jaywalker[1] > self.dim_y:
            # riparti daccapo: nuova X e inverti direzione
            self.jaywalker[0] = random.uniform(self.dim_x/2, self.dim_x)
            self.jaywalker_dir *= -1
            # correggi Y all’interno
            self.jaywalker[1] = np.clip(self.jaywalker[1], 0, self.dim_y)
        # aggiorna bounding box del pedone
        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r


        #modifiche per aggiunta dinamica di ostacoli
        for obs in self.obstacles:
            obs['pos'][0] -= obs['v']  # si muovono verso sinistra

        df, a = self.actions[action]
        self.car.move(df, a) # default ripropaga l'accellerazione, altrimenti la modifico

        reward = np.zeros(self.reward_size)
        terminated = False
        completed = False

        if self.collision_with_jaywalker(): # collision with jaywalker
            reward[0] = -10
            terminated = True


        # distance from target
        reward[1] = (self.car.position[0] - self.car.prev_position[0])/self.scale_factor

        # accept surpassing the goal, terminate
        if self.car.front[0] >= self.goal[0]:
            if not terminated:
                completed = True
            terminated = True


        # collision with borders
        if self.car.front[1] > self.dim_y or self.car.front[1] < 0 or self.car.back[1] > self.dim_y or self.car.back[1] < 0 or self.car.position[0] < 0 or self.car.front[0] < 0:
            reward[2] -= 1000
            terminated = True 
        # distance from center of own lane
        else:
            # computes a distance-based penalty to encourage the car to stay centered in its lane
            reward[2] = -np.abs(self.car.position[1] - self.goal[1])

        reward[2] /= self.scale_factor * 10


        inv_distance, angle, jaywalker_distance = self.vision()

        # Find closest object
        obs_distance = float('inf')
        if self.obstacles:
            obs_distances = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
            obs_distance = min(obs_distances)
        
        # Call  vision_obstacle if jaywalker is detected or closest obstacle is closer than jaywalker
        if jaywalker_distance < float('inf') or obs_distance < jaywalker_distance:
            inv_distance_obs, angle_obs = self.vision_obstacle()
        else:
            inv_distance_obs, angle_obs = 0, -np.pi


        
        dists = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
        i_min = np.argmin(dists)
        obs = self.obstacles[i_min]
        inv_d_obs = 1/dists[i_min]
        angle_obs = np.arctan((obs['pos'][1]-self.car.position[1])/(obs['pos'][0]-self.car.position[0]+self.noise))
        # corsia attuale (indice)
        lane_idx = min(range(len(self.lanes_y)), key=lambda i: abs(self.car.position[1]-self.lanes_y[i]))
        # nuovo state
        state = array([
            self.car.position[1],
            inv_distance, angle,          # Jaywalker info
            self.car.v, self.car.beta,
            inv_distance_obs, angle_obs, # Obstacle info
            float(lane_idx)
        ])

        min_dist = 1 / inv_d_obs


        self.counter_iterations += 1
        truncated = False

        if self.counter_iterations >= self.max_iterations:
            truncated = True

        if self.collision_with_obstacle():
            reward[0] -= 10
            terminated = True

        return state, reward, terminated, truncated, completed


    def reset(self):
        self.obstacles = []

        # Alternanza scenari
        self.last_scenario = getattr(self, 'last_scenario', 1)
        current_scenario = 2 if self.last_scenario == 1 else 1
        self.last_scenario = current_scenario

        self.car.reset(array([0.0, 2.5]))
        self.counter_iterations = 0

        # --- Pedone fermo a metà strada, posizione fissa ---
        self.jaywalker = array([50+20, self.dim_y / 4])
        self.jaywalker_speed = 0.0
        self.jaywalker_dir = 0
        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r

        # Initialize pos_x with a default value
        pos_x = self.dim_x  # Default position
        speed = 0.0  # Default speed

        # Alternanza scenari in fase 1
        if self.curriculum_stage == 1:
            scenario = random.choice(["easy", "critical"])
        else:
            scenario = "critical"

        # Configura posizione ostacolo
        if scenario == "easy":
            pos_x = self.dim_x + 100  # lontano dal pedone
            speed = 0.0
        elif scenario == "critical":
            base_speed = 2.0
            if self.curriculum_stage == 2:  # fase dopo epsilon=0.01
                speed = base_speed - self.completed_mean
                # aggiunta varianza alla v di partenza dell'auto
                sigma = 1.5 # 1.5
                v_base = 2.0
                v_final = (v_base + np.random.normal(0, sigma)) * self.completed_mean
                v_final = max(0.0, v_final) # evito v negative
                self.car.v = v_final
            else:
                speed = base_speed
            pos_x = self.jaywalker[0] + 20  # Default position for critical scenario

        # Auto ostacolante nella corsia di sorpasso
        lane = self.lanes_y[1]
        self.obstacles.append({
            'type': 'car',
            'pos': array([pos_x, lane]),
            'r': 2.0,
            'v': speed
        })

        # Stato iniziale
        inv_distance, angle, jaywalker_distance = self.vision()
        dists = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
        i_min = np.argmin(dists)
        obs = self.obstacles[i_min]
        inv_d_obs = 1 / dists[i_min]
        angle_obs = np.arctan((obs['pos'][1] - self.car.position[1]) / (obs['pos'][0] - self.car.position[0] + self.noise))
        lane_idx = min(range(len(self.lanes_y)), key=lambda i: abs(self.car.position[1] - self.lanes_y[i]))


        # Find closest object
        obs_distance = float('inf')
        if self.obstacles:
            obs_distances = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
            obs_distance = min(obs_distances)
        
        # Call  vision_obstacle if jaywalker is detected or closest obstacle is closer than jaywalker
        if jaywalker_distance < float('inf') or obs_distance < jaywalker_distance:
            inv_distance_obs, angle_obs = self.vision_obstacle()
        else:
            inv_distance_obs, angle_obs = 0, -np.pi
        

        return array([
            self.car.position[1],
            inv_distance,
            angle,
            self.car.v,
            self.car.beta,
            inv_distance_obs,
            angle_obs,
            float(lane_idx)
        ])

    

    def random_action(self):
        return int(np.floor(random.random() * self.action_size))


    def __str__(self):
        return "jaywalker"
    
    #MODIFICHE PER VEDERE GLI OSTACOLI
    def render(self):
        plt.clf()
        ax = plt.gca()
        road = mpatches.Rectangle((0, 0), self.dim_x, self.dim_y,
                                  facecolor='black', edgecolor='none')
        ax.add_patch(road)

        gx = self.goal[0]
        ax.plot([gx, gx], [0, self.dim_y],
            color='lime', linewidth=2,
            linestyle=(0, (5, 5)),
            label='Finish')

        # 2) linee di bordo continue – bianche
        plt.plot([0, self.dim_x], [0, 0], color='white', linewidth=2)
        plt.plot([0, self.dim_x], [self.dim_y, self.dim_y], color='white', linewidth=2)

        # 3) linea centrale tratteggiata – bianca
        mid_y = self.dim_y / 2
        plt.plot([0, self.dim_x], [mid_y, mid_y],
                 color='white', linewidth=1,
                 linestyle=(0, (10, 10))) 

        #PEDONE COME CERCHIO ROSSO
        circle_j = plt.Circle(self.jaywalker, self.jaywalker_r, color='red', alpha=0.5)
        plt.gca().add_patch(circle_j)

       #GLI OSTACOLI POSSONO ESSERE MACCHINE (ARANCIONI)
        for obs in getattr(self, 'obstacles', []):
            c = 'orange' if obs['type']=='car' else 'green'
            circle_o = plt.Circle(obs['pos'], obs['r'], color=c, alpha=0.5)
            plt.gca().add_patch(circle_o)

        car = self.car
        car_length = 4.0
        car_width = 4.0
        arg = car.phi + car.beta

# Coordinate per posizionare l'immagine
        extent = [
            car.position[0] - car_length / 2,
            car.position[0] + car_length / 2,
            car.position[1] - car_width / 2,
            car.position[1] + car_width / 2
        ]

# Trasformazione per ruotare l'immagine
        img_transform = transforms.Affine2D().rotate_around(
            car.position[0], car.position[1], arg
        ) + plt.gca().transData

# Mostra immagine dell’auto
        plt.imshow(self.car_img, extent=extent, transform=img_transform, zorder=5)



        blue_patch = mpatches.Patch(color='blue', label='Your Car')
        red_patch = mpatches.Patch(color='red', label='Jaywalker')
        orange_patch = mpatches.Patch(color='orange', label='Obstacle Car')
        #plt.legend(handles=[blue_patch, red_patch, orange_patch])
        
        plt.xlim(-1, self.dim_x+1)
        plt.ylim(-1, self.dim_y+1)
        plt.pause(0.001)




class Weighted_Q_Network(Q_Network):

    def __init__(self, n_observations, n_actions, hidden = 128, learning_rate = 1e-4, weights = None):

        super().__init__(n_observations, hidden, weights)
        
        self.n_actions = n_actions
        self.reward_size = len(weights)

        self.weights = self.weights.to(device)

        self.final_layer = nn.Linear(hidden, n_actions*self.reward_size)

        self.instantiate_optimizer(learning_rate)

    
    def forward(self, x):
        
        z = self.common_forward(x)

        z = self.final_layer(z).view(-1, self.n_actions, self.reward_size)

        return z


    def learn(self, predict, target):

        self.optimizer.zero_grad()

        loss = self.criterion(torch.matmul(predict, self.weights), torch.matmul(target, self.weights))

        loss.backward()

        self.optimizer.step()


    def  __str__(self):
        return "Weight"



class Scalar_Q_Network(nn.Module):

    def __init__(self, n_observations, n_actions, hidden = 128, learning_rate = 1e-4):
        pass


    def  __str__(self):
        return "Scalar"



class QAgent():

    def __init__(self, network, env, learning_rate, batch_size, hidden, slack, \
                 epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, \
                 train_start, replay_frequency, target_model_update_rate, memory_length, mini_batches, weights):

        self.env = env

        self.batch_size = batch_size
        self.state_size = 8 #env.state_size
        self.action_size = env.action_size
        self.reward_size = env.reward_size
        self.slack = slack

        self.permissible_actions = torch.tensor(range(self.action_size)).to(device)

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.gamma = gamma
        self.replay_frequency = replay_frequency
        self.target_model_update_rate = target_model_update_rate
        self.mini_batches = mini_batches

        #modifiche per curriculum
        self.curriculum_stage = 1


        self.score = []
        self.epsilon_record = []
        self.completed = []
        self.num_actions = []

        self.train_start = train_start

        
        self.seq_len = 10  # ad esempio
        self.memory = SequentialReplayBuffer(
            capacity=memory_length,
            seq_len=self.seq_len,
            obs_dim=self.state_size,
            action_dim=1,  # perché l'azione è scalare
            reward_dim=self.reward_size,
            device=device
        )

        self.model = network(self.state_size, self.action_size, hidden, learning_rate, weights)
        self.target_model = network(self.state_size, self.action_size, hidden, learning_rate, weights)
        self.model.to(device)
        self.target_model.to(device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        self.target_model.eval()

        self.state_batch = torch.empty((self.batch_size, self.state_size), device = device)
        self.next_state_batch = torch.empty((self.batch_size, self.state_size), device = device)
        self.action_batch = torch.empty((self.batch_size), dtype = torch.long, device = device)
        self.reward_batch = torch.empty((self.batch_size,self.reward_size), device = device)
        self.not_done_batch = torch.empty(self.batch_size, dtype = torch.bool, device = device)
        self.next_state_values = torch.zeros((self.batch_size, self.reward_size), device = device)


    def update_epsilon(self):
        if len(self.memory) >= self.train_start and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
            
    def add_experience(self, state, action, reward, done):
        self.memory.add(state, action, reward, done)



    def act(self, current_state):
        # Estrai gli ultimi T passi dal buffer
        if len(self.memory.states) < self.seq_len:
            # fallback: pad con zeri se non abbastanza
            pad_len = max(0, self.seq_len - len(self.memory.states))
            start_idx = max(0, len(self.memory.states) - self.seq_len)

            states  = [torch.zeros_like(current_state)] * pad_len + list(itertools.islice(self.memory.states, start_idx, None))
            actions = [torch.zeros(1)] * pad_len + list(itertools.islice(self.memory.actions, start_idx, None))
            rewards = [torch.zeros(self.reward_size)] * pad_len + list(itertools.islice(self.memory.rewards, start_idx, None))

        else:
            start_idx = max(0, len(self.memory.states) - self.seq_len)

            states = list(itertools.islice(self.memory.states, start_idx, None))
            actions = list(itertools.islice(self.memory.actions, start_idx, None))
            rewards = list(itertools.islice(self.memory.rewards, start_idx, None))
        
        states = [s.squeeze(0) if s.dim() == 2 and s.size(0) == 1 else s for s in states]
        states = torch.stack(states).unsqueeze(0).to(device)  # (1, T, obs_dim)


        # Passa la sequenza al modello
        with torch.no_grad():
            q_values = self.model(states, actions=actions, rewards=rewards)  # (1, T, n_actions, reward_size)
            q_t = q_values[:, -1]  # prendiamo l'ultimo step della sequenza → (1, n_actions, reward_size)

        q_t = q_t.squeeze(0)  # (n_actions, reward_size)

        # Usa la tua policy arglexmax (o softmax, o max su reward pesato)
        selected_action = self.arglexmax(q_t)

        return selected_action



    def greedy_arglexmax(self, Q):
        permissible_actions = self.permissible_actions

        mask = (Q[:, 0] >= -0.7) # sceglie solo le azioni con Q[0] >= -0.7, cioè quelle con reward collisione accettabile

        if not torch.any(mask):
            permissible_actions = self.refine_actions(permissible_actions, Q[:,0])
        
        else:
            permissible_actions = permissible_actions[mask]

        permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,1])
        
        return permissible_actions[Q[permissible_actions,2].max(0)[1]]


    def select_action(self, state):
        if random.random() <= self.epsilon: #<= ?????????????????????????? 
            return self.env.random_action()
        
        with torch.no_grad():
            q_value = self.model(state).squeeze()
            action = self.arglexmax(q_value)
            return action


    def refine_actions(self, permissible_actions, q):
        lower_bound = q.max(0)[0]
        lower_bound -= self.slack * torch.abs(lower_bound)
        mask = q >= lower_bound
        return permissible_actions[mask.nonzero(as_tuple=True)[0]]


    def arglexmax(self, Q):
        # Inizializza tensore delle azioni permessibili
        permissible_actions = torch.tensor(self.permissible_actions, dtype=torch.long)

        # Step 1: filtro sul reward 0
        q0 = Q[permissible_actions, 0]
        mask = q0 >= -0.7

        if not torch.any(mask):
            permissible_actions = self.refine_actions(permissible_actions, q0)
        else:
            permissible_actions = permissible_actions[mask.nonzero(as_tuple=True)[0]]

        # Step 2: filtro sul reward 1
        q1 = Q[permissible_actions, 1]
        permissible_actions = self.refine_actions(permissible_actions, q1)

        # Step 3: filtro sul reward 2
        q2 = Q[permissible_actions, 2]
        permissible_actions = self.refine_actions(permissible_actions, q2)

        return random.choice(permissible_actions.tolist())


    
    def update_target_model(self, tau):
        weights = self.model.state_dict()
        target_weights = self.target_model.state_dict()
        for i in target_weights:
            target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
        self.target_model.load_state_dict(target_weights)
    

    def save_arglexmax(self, i):
        self.actions[i] = self.arglexmax(self.q_values[i,:])


    def q_value_arrival_state(self, states):
        self.q_values = self.model(states)
        self.actions = torch.empty(len(states), device=device, dtype=torch.int64)

        for i in range(len(states)):
            #self.actions[i] = self.arglexmax(self.q_values[i,:])
            self.actions[i] = self.greedy_arglexmax(self.q_values[i,:])
        #p.map(self.save_arglexmax, np.arange(0, len(states)))

        actions = torch.vstack((self.actions, self.actions, self.actions)).T.unsqueeze(1)
        # evaluate a' according wih target network
        return self.target_model(states).gather(1,actions).squeeze()

    def experience_replay(self):
        if not self.memory.can_sample(self.batch_size):
            return

        states, actions, rewards, dones = self.memory.sample(self.batch_size)
        # states: (B, T, obs_dim)
        # actions: (B, T)
        # rewards: (B, T, reward_size)

        # Calcolo target Q values
        with torch.no_grad():
            next_states = states[:, 1:, :]  # da t+1 a T
            rewards_seq = rewards[:, :-1, :]  # da t a T-1
            actions_seq = actions[:, :-1]  # da t a T-1

            target_q = self.target_model(next_states)  # (B, T-1, A, R)
            best_next_actions = target_q.max(dim=2)[0]  # max_a Q(s_{t+1}, a)
            target_values = rewards_seq + self.gamma * best_next_actions

        # Predicted Q values per azione presa
        pred_q = self.model(states[:, :-1, :])  # (B, T-1, A, R)

        # Estrai i valori corrispondenti all'azione presa
        action_indices = actions_seq.unsqueeze(-1).unsqueeze(-1)  # (B, T−1, 1, 1)
        action_indices = action_indices.expand(-1, -1, 1, self.reward_size)  # (B, T−1, 1, R)

        predicted_values = pred_q.gather(2, action_indices).squeeze(2)  # (B, T−1, R)

        self.model.train()
        self.model.learn(predicted_values, target_values)
        self.model.eval()



    def learn(self):
        bar = qqdm(np.arange(self.episodes), desc="Learning")

        best_completed = 0.0 # Track the best completition score
        consecutive_successes = 0 # counter for consecutive completed episodes
        compl_mean = 0.0 # Initialize completion mean

        for e in bar:
        
            state = self.env.reset()
            state = torch.tensor(state).to(device)
            episode_score = np.zeros(self.reward_size)
            step = 0
            done = False
                        
            while not done:
                #action = self.select_action(state.unsqueeze(0))
                action = self.act(state.unsqueeze(0))
                next_state, reward, terminated, truncated, completed = self.env.step(action)

                self.env.render()
                done = terminated or truncated          
                next_state = torch.tensor(next_state).to(device)
                episode_score += reward
                reward = torch.tensor(reward)
                self.add_experience(state.cpu(), action, reward.cpu(), terminated)
                state = next_state
                
                if (step & self.replay_frequency) == 0:
                    for i in np.arange(self.mini_batches):
                        self.experience_replay()
                        self.update_target_model(self.target_model_update_rate)
                
                step += 1

                if done:                                
                    self.update_epsilon()
                    # Passa alla fase 2 quando epsilon ≈ epsilon_min
                    if self.epsilon <= 0.02 and self.curriculum_stage == 1: #0.01 HARDCODED
                        self.curriculum_stage = 2
                        self.env.curriculum_stage = 2

                    self.score.append(episode_score)
                    self.epsilon_record.append(self.epsilon)
                    self.completed.append(completed)
                    self.num_actions.append(step)

                # Update best completion score and check conditions
                if completed:
                    # Calculate moving averages over last 31 episodes (or all available if fewer)
                    window_size = min(31, len(self.completed))
                    
                    # Collision score (index 0 in self.score)
                    current_collisions = np.mean([s[0] for s in self.score[-window_size:]]) if self.score else 0
                    
                    # Completion rate
                    current_completed = np.mean(self.completed[-window_size:]) if self.completed else 0
                    
                    # Update best completion score
                    if current_completed > best_completed:
                        best_completed = current_completed
                        print(f"New best completion score: {best_completed:.2f} at episode {e}")
                    
                    # Check for model saving condition
                    if current_completed > 0.70:
                        save_path = f"best_model_episode_{e}.pt"
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Model saved at episode {e}: Collision=0, Completed={current_completed:.2f}")
                    
                    # Update consecutive successes counter
                    if current_completed > 0.70:
                        consecutive_successes += 1
                    else:
                        if consecutive_successes > 0:
                            print(f"{consecutive_successes} consecutive successes reset at episode {e}.")
                            consecutive_successes = 0
                    
                    # Early stopping condition
                    if consecutive_successes >= 100:
                        print(f"Early stopping achieved at episode {e} with {consecutive_successes} consecutive successes.")
                        break
                if e >= 31:
                    rew_mean = sum(self.score[-31:]) / 31
                    compl_mean = np.mean(self.completed[-31:])
                    act_mean = np.mean(self.num_actions[-31:])
                    current_eps = self.epsilon_record[-1] if self.epsilon_record else self.epsilon

                    bar.set_infos({
                            'Speed_': f'{(time.time() - bar.start_time) / (bar.n + 1):.2f}s/it',
                            'Collision': f'{rew_mean[0]:.2f}',
                            'Forward': f'{rew_mean[1]:.2f}',
                            'OOB': f'{rew_mean[2]:.2f}',
                            'Completed': f'{compl_mean:.2f}',
                            'Actions': f'{act_mean:.2f}',
                            'Epsilon': f'{current_eps:.4f}'
                        })
                self.env.completed_mean = compl_mean

                    
        #self.env.close()


    def plot_score(self, score, start, end, N, title, filename):
        plt.plot(score);
        mean_score = np.convolve(array(score), np.ones(N)/N, mode='valid')
        plt.plot(np.arange(start,end), mean_score)

        if title is not None:
            plt.title(title);

        plt.savefig(filename)
        plt.clf()


    def plot_learning(self, N, title, filename):
        vs = VectorScore(*zip(*self.score))
        time = len(vs.oob)
        start = math.floor(N/2)
        end = time-start
        self.plot_score(vs.collision, start, end, N, title + " collision", filename + str(self.env) + "_collision_graph")
        self.plot_score(vs.oob, start, end, N, title + " oob", filename + str(self.env) + "_oob_graph")
        self.plot_score(vs.distance, start, end, N, title + " distance", filename + str(self.env) + "_distance_graph")
        self.plot_score(self.completed, start, end, N, title + " completed", filename + str(self.env) + "_completed_graph")
        self.plot_score(self.num_actions, start, end, N, title + " actions", filename + str(self.env) + "_actions_graph")
        

    def plot_epsilon(self, filename = ""):
        plt.plot(self.epsilon_record);
        plt.title("Epsilon decay");
        plt.savefig(filename + str(self.env) + "_epsilon");
        plt.clf()
        
    
    def save_model(self, path=""):
        torch.save(self.model.state_dict(), path+str(self.env)+"_"+str(self)+".pt")

    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path+str(self.env)+"_"+str(self)+".pt", map_location=torch.device(device)))
        self.model.eval()


    def __str__(self):
        return "QAgent"


    def simulate(self, number = 0, path = "", verbose = False):
        done = False
        self.epsilon = -1
        state = self.env.reset()
        state = torch.tensor(state).to(device)
        position_x = []
        position_y = []
        position_x.append(self.env.car.position[0])
        position_y.append(self.env.car.position[1])

        #print(state)
        #print(self.env.car.position)
        #print("")

        while not done:
            if verbose:
                print(state)
                print(self.model(state.unsqueeze(0)))

            #action = self.select_action(state.unsqueeze(0))
            action = self.act(state.unsqueeze(0))

            if verbose:
                print(action)
                print("")
            
            next_state, _, terminated, truncated, _  = self.env.step(action)

            done = terminated or truncated

            state = torch.tensor(next_state).to(device)
            #print(state)
            #print(self.env.car.position)
            #print("")

            position_x.append(self.env.car.position[0])
            position_y.append(self.env.car.position[1])

        plt.plot(position_x, position_y);

        # lanes
        plt.plot([0.0, self.env.dim_x], [0.0, 0.0]);
        plt.plot([0.0, self.env.dim_x], [self.env.dim_y, self.env.dim_y]);

        # jaywalker
        jaywalker_position_x = [self.env.jaywalker[0] - self.env.jaywalker_r,\
                                self.env.jaywalker[0] - self.env.jaywalker_r,\
                                self.env.jaywalker[0] + self.env.jaywalker_r,\
                                self.env.jaywalker[0] + self.env.jaywalker_r,\
                                self.env.jaywalker[0] - self.env.jaywalker_r]
        
        jaywalker_position_y = [self.env.jaywalker[1] - self.env.jaywalker_r,\
                                self.env.jaywalker[1] + self.env.jaywalker_r,\
                                self.env.jaywalker[1] + self.env.jaywalker_r,\
                                self.env.jaywalker[1] - self.env.jaywalker_r,\
                                self.env.jaywalker[1] - self.env.jaywalker_r]
        
        plt.plot(jaywalker_position_x, jaywalker_position_y);

        plt.savefig(path + str(self.env) + "_simulation_" + str(number) + "_render.png");
        plt.clf();



class DTAgent(): # Rinominato da QAgent per chiarezza
    def __init__(self, network, env, learning_rate, batch_size, hidden, slack, \
                 epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, \
                 train_start, replay_frequency, target_model_update_rate, memory_length, mini_batches, weights): # Alcuni args potrebbero non servire più
        
        self.env = env

        self.batch_size = batch_size
        self.state_dim = env.state_size # Usiamo state_dim per coerenza con DT
        self.action_dim = env.action_size # Usiamo action_dim
        self.reward_dim = env.reward_size # Reward dim dal tuo ambiente

        # Per DT, epsilon non è usato per l'esplorazione, ma la si può gestire nell'ambiente o in fase di simulazione
        self.epsilon = epsilon_start # Potrebbe non essere usato per DT training
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.episodes = episodes
        self.gamma = gamma # Gamma è ancora rilevante per calcolare RTG se non usi solo somma semplice
        self.replay_frequency = replay_frequency # Replay frequency è ancora utile
        # Target model update rate non è usato per DT classico
        self.target_model_update_rate = target_model_update_rate # Rimuovere o ignorare
        self.mini_batches = mini_batches # Ancora utile per gli update

        #modifiche per curriculum
        self.curriculum_stage = 1

        self.score = []
        self.epsilon_record = []
        self.completed = []
        self.num_actions = []

        self.train_start = train_start # Quando iniziare l'addestramento

        self.seq_len = 10  # Lunghezza della sequenza per il Decision Transformer
        self.memory = SequentialReplayBuffer(
            capacity=memory_length,
            seq_len=self.seq_len,
            obs_dim=self.state_dim,
            action_dim=1,  # L'azione è scalare
            reward_dim=self.reward_dim,
            device=device
        )

        # Il "network" qui sarà il DecisionTransformer
        # Nota: La funzione network_constructor nel tuo main.py deve restituire un DecisionTransformer
        self.model = network(
            # state_dim=self.state_dim,
            act_dim=self.action_dim,
            hidden_size=hidden,
            max_length=self.seq_len,
            reward_dim=1 # Se RTG è sempre scalare
        ).to(device)

        # Per il DT, di solito non c'è un target network separato come in DQN.
        # self.target_model = network(self.state_dim, self.action_dim, hidden, learning_rate, weights)
        # self.target_model.to(device)
        # self.target_model.load_state_dict(self.model.state_dict())
        # self.target_model.eval()
        self.model.train() # Il DT è quasi sempre in modalità train per l'addestramento

        # Optimizer e Loss Function sono interni al DecisionTransformer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = self.model.loss_fn # Usiamo la loss function definita nel DT

    # epsilon update non è più rilevante per la selezione delle azioni in DT
    def update_epsilon(self):
        # Questo metodo potrebbe essere rimosso o modificato per un diverso tipo di esplorazione
        # o per la gestione del curriculum
        if len(self.memory) >= self.train_start and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def add_experience(self, state, action, reward, done):
        self.memory.add(state, action, reward, done)

    def act(self, current_state, target_return_scalar=0.0):
        """
        Per il Decision Transformer, l'azione è condizionata dal target return.
        current_state: lo stato corrente (1, obs_dim)
        target_return_scalar: il return desiderato per l'episodio (scalare)
        """
        self.model.eval() # Per l'inferenza, mettiamo il modello in eval mode
        
        # Prepara gli input per il DT
        # States: ultimi seq_len stati + stato corrente (padded se necessario)
        # Actions: ultime seq_len azioni (padded)
        # RTGs: ultimi seq_len RTG (padded), il più recente è il target_return_scalar

        # Se il buffer è vuoto o troppo piccolo, si usano padding iniziali
        states_list = list(self.memory.current_episode['states']) + [current_state.cpu()]
        actions_list = list(self.memory.current_episode['actions'])
        rewards_list = list(self.memory.current_episode['rewards'])

        # Calcolo RTG per la predizione:
        # Per la predizione, il DT ha bisogno dell'RTG *futuro desiderato*.
        # Qui usiamo un target_return_scalar come input.
        # Se la lunghezza dell'episodio corrente supera seq_len, prendiamo gli ultimi seq_len.
        
        # Per la simulazione, creiamo una sequenza di input di lunghezza self.seq_len
        # Il RTG al tempo t è il return desiderato da t in poi.
        # Lo stato al tempo t è lo stato osservato a t.
        # L'azione al tempo t è l'azione presa a t-1 (se si predice l'azione corrente).
        # Per predire l'azione corrente, il contesto è (RTG corrente, stato corrente, azione precedente).

        # Costruiamo gli input per il DT:
        # states: (1, seq_len, obs_dim)
        # actions: (1, seq_len)
        # rtgs: (1, seq_len, 1)

        # Padding iniziale se la sequenza è più corta di seq_len
        pad_len = max(0, self.seq_len - len(states_list))
        
        current_states_padded = ([torch.zeros(self.state_dim, dtype=torch.float32)] * pad_len + states_list[-self.seq_len:]).to(device)
        current_actions_padded = ([torch.tensor(0, dtype=torch.long)] * pad_len + actions_list[-self.seq_len:]).to(device)
        
        # Per RTG, l'ultimo RTG nella sequenza deve essere il target_return_scalar.
        # Gli RTG precedenti possono essere interpolati o impostati a target_return_scalar.
        # Un approccio comune è replicare il target_return_scalar per tutta la sequenza.
        current_rtgs_padded = (torch.full((pad_len + len(states_list[-self.seq_len:]), 1), target_return_scalar, dtype=torch.float32)).to(device)


        # Unstack the list of tensors for concatenation if needed
        # current_states_padded = torch.stack(current_states_padded).unsqueeze(0).to(device)
        # current_actions_padded = torch.stack(current_actions_padded).unsqueeze(0).to(device)
        # current_rtgs_padded = torch.stack(current_rtgs_padded).unsqueeze(0).to(device)
        # No need to stack lists after creating tensors directly
        
        # In this implementation, states_list[-self.seq_len:] etc. are already tensors.
        # If they are lists of tensors, you need to stack them first.
        # Assuming current_states_padded etc. are already tensors of correct shape.

        # Correctly stack lists of tensors if they are coming from current_episode (which are lists)
        states_for_dt = torch.stack(states_list[-self.seq_len:]).unsqueeze(0)
        actions_for_dt = torch.stack(actions_list[-self.seq_len:]).unsqueeze(0)
        rtgs_for_dt = torch.full((1, states_for_dt.shape[1], 1), target_return_scalar, dtype=torch.float32)


        # Pad if needed
        if states_for_dt.shape[1] < self.seq_len:
            pad_len = self.seq_len - states_for_dt.shape[1]
            states_for_dt = F.pad(states_for_dt, (0, 0, pad_len, 0))
            actions_for_dt = F.pad(actions_for_dt, (pad_len, 0), "constant", 0)
            rtgs_for_dt = F.pad(rtgs_for_dt, (0, 0, pad_len, 0), "constant", target_return_scalar)

        states_for_dt = states_for_dt.to(device)
        actions_for_dt = actions_for_dt.to(device)
        rtgs_for_dt = rtgs_for_dt.to(device)


        with torch.no_grad():
            # Il DT predice i logits per tutte le azioni per ogni timestep della sequenza.
            # Vogliamo l'azione per l'ultimo timestep (lo stato corrente).
            logits = self.model(rtgs_for_dt, states_for_dt, actions_for_dt) # (1, seq_len, act_dim)
            
            # Prendiamo i logits dell'ultimo timestep della sequenza
            action_logits = logits[:, -1, :] # (1, act_dim)
            
            # Applica softmax e campiona o prende argmax
            probs = F.softmax(action_logits, dim=-1)
            
            # Esplorazione (se desiderata durante la simulazione/valutazione)
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = torch.argmax(probs).item()

        self.model.train() # Torna in train mode dopo l'inferenza
        return action

    # Non serve più greedy_arglexmax e arglexmax per DT puro, a meno che non si voglia usarle come baseline
    # per la selezione delle azioni in un contesto non-DT o per il testing.
    # Se il DT è il tuo unico modello, puoi rimuoverli.

    def update_target_model(self, tau):
        # Questo metodo non è rilevante per Decision Transformer classico
        pass

    def experience_replay(self):
        if not self.memory.can_sample(self.batch_size):
            return

        # Campiona (rtgs, states, actions)
        # rtgs: (B, T, 1)
        # states: (B, T, obs_dim)
        # actions: (B, T)
        batch_rtgs, batch_states, batch_actions = self.memory.sample(self.batch_size)

        # Il DT prende come input (rtgs, states, actions) e predice le azioni successive
        # Quindi, se l'input è (rtgs[t], states[t], actions[t-1]), l'output è pred_actions[t]
        # In una configurazione più comune, il DT predice actions[t] dato (rtgs[t], states[t], actions[t-1])
        # Per addestrare:
        # Input del modello: (batch_rtgs[:, :-1], batch_states[:, :-1], batch_actions[:, :-1])
        # Target: batch_actions[:, 1:] (azioni future)

        # Nota: La tua attuale implementazione del forward in DecisionTransformer prende actions=None
        # se non è fornito e crea zeros_like. Qui dobbiamo passarlo correttamente.
        # La predizione di un DT è per l'azione *successiva* nella sequenza.
        # Quindi, se il token `i` è `(rtg_i, state_i, action_{i-1})`, il DT dovrebbe predire `action_i`.
        # Di solito, il primo elemento di `actions` viene ignorato (o è un token dummy)
        # e le predizioni `logits[i]` corrispondono a `target_actions[i]`.

        # Input per il DT: (rtgs, states, actions) per i passi da 0 a T-1
        # Target per la loss: actions per i passi da 1 a T
        # Questo significa che actions[:, 0] viene usato come "azione precedente" per predire actions[:, 1].
        
        # Per la perdita, i logits predicono action_i, dato (rtg_i, state_i, action_{i-1})
        # I target sono le azioni vere (batch_actions)
        
        # Input: rtgs, states, actions (passati direttamente)
        # L'output logits avrà dimensione (B, T, act_dim)
        # Il target_actions sarà (B, T)
        
        self.optimizer.zero_grad()
        predicted_action_logits = self.model(batch_rtgs, batch_states, batch_actions)
        
        # La loss deve essere calcolata sui logits del DT e sulle azioni vere.
        # Il DT predice l'azione al timestep `t` dato il contesto fino al timestep `t`.
        # Target Actions: `batch_actions`
        # Loss è tra `predicted_action_logits` e `batch_actions`
        loss = self.loss_fn(predicted_action_logits, batch_actions) # Assicurati che predicted_action_logits sia (B*T, act_dim) e batch_actions sia (B*T) al momento del passaggio alla loss_fn
        
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def learn(self):
        bar = qqdm(np.arange(self.episodes), desc="Learning")

        best_completed = 0.0 # Track the best completition score
        consecutive_successes = 0 # counter for consecutive completed episodes
        compl_mean = 0.0 # Initialize completion mean

        for e in bar:
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_score = np.zeros(self.reward_size)
            step = 0
            terminated = False # Flag per terminazione ambiente
            truncated = False  # Flag per troncamento ambiente

            # Target Return: Definisci un target return desiderato per l'episodio
            # Questo è un iperparametro cruciale per il DT.
            # Potresti iniziare con un valore fisso (es. 100 per un buon ritorno)
            # o campionarlo da una distribuzione, o basarlo su target di performance passati.
            # Per ora, usiamo un valore fisso per esempio.
            target_return = 100.0 # Esempio: un valore alto per incoraggiare performance elevate.

            # Reset the current_episode buffer in memory for the new episode
            self.memory.current_episode = {
                'states': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
                        
            while not terminated and not truncated:
                # Per il Decision Transformer, l'azione è selezionata in base allo stato corrente
                # e al return-to-go desiderato.
                action = self.act(state.unsqueeze(0), target_return_scalar=target_return)
                
                next_state, reward_vec, terminated, truncated, completed = self.env.step(action)
                
                self.env.render() # Rendering dell'ambiente

                # Aggiorna il target_return per il prossimo passo.
                # L'RTG al tempo t è (target_return del passo precedente) - (reward al passo precedente).
                # Questo è il modo in cui il DT "persegue" il suo obiettivo.
                # Se il reward è un vettore, devi decidere quale componente usare per l'RTG.
                target_return -= reward_vec[0] # Usiamo il primo componente del reward per aggiornare l'RTG

                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
                episode_score += reward_vec # Accumula il reward vettoriale

                # Aggiungi l'esperienza al replay buffer episodico
                self.add_experience(state.cpu().numpy(), action, reward_vec, terminated or truncated)
                
                state = next_state_tensor
                
                if (step % self.replay_frequency) == 0 and len(self.memory) >= self.train_start:
                    for i in np.arange(self.mini_batches):
                        self.experience_replay()
                
                step += 1

            # Dopo la fine dell'episodio
            self.update_epsilon() # Puoi ancora usarlo per curriculum learning o altre logiche
            
            self.score.append(episode_score)
            self.epsilon_record.append(self.epsilon)
            self.completed.append(completed)
            self.num_actions.append(step)

            # Logica di logging e salvataggio del modello
            # ... (la tua logica di logging e salvataggio rimane simile) ...
            if e >= 31: # Aggiorna la barra di progresso e i messaggi dopo un certo numero di episodi
                    rew_mean = sum(self.score[-31:]) / 31
                    compl_mean = np.mean(self.completed[-31:])
                    act_mean = np.mean(self.num_actions[-31:])
                    current_eps = self.epsilon_record[-1] if self.epsilon_record else self.epsilon

                    bar.set_infos({
                            'Speed_': f'{(time.time() - bar.start_time) / (bar.n + 1):.2f}s/it',
                            'Collision': f'{rew_mean[0]:.2f}',
                            'Forward': f'{rew_mean[1]:.2f}',
                            'OOB': f'{rew_mean[2]:.2f}',
                            'Completed': f'{compl_mean:.2f}',
                            'Actions': f'{act_mean:.2f}',
                            'Epsilon': f'{current_eps:.4f}'
                        })
            self.env.completed_mean = compl_mean
            
        #self.env.close()

    def plot_score(self, score, start, end, N, title, filename):
        plt.plot(score);
        mean_score = np.convolve(array(score), np.ones(N)/N, mode='valid')
        plt.plot(np.arange(start,end), mean_score)

        if title is not None:
            plt.title(title);

        plt.savefig(filename)
        plt.clf()


    def plot_learning(self, N, title, filename):
        vs = VectorScore(*zip(*self.score))
        time = len(vs.oob)
        start = math.floor(N/2)
        end = time-start
        self.plot_score(vs.collision, start, end, N, title + " collision", filename + str(self.env) + "_collision_graph")
        self.plot_score(vs.oob, start, end, N, title + " oob", filename + str(self.env) + "_oob_graph")
        self.plot_score(vs.distance, start, end, N, title + " distance", filename + str(self.env) + "_distance_graph")
        self.plot_score(self.completed, start, end, N, title + " completed", filename + str(self.env) + "_completed_graph")
        self.plot_score(self.num_actions, start, end, N, title + " actions", filename + str(self.env) + "_actions_graph")
        

    def plot_epsilon(self, filename = ""):
        plt.plot(self.epsilon_record);
        plt.title("Epsilon decay");
        plt.savefig(filename + str(self.env) + "_epsilon");
        plt.clf()
        
    
    def save_model(self, path=""):
        torch.save(self.model.state_dict(), path+str(self.env)+"_"+str(self)+".pt")

    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path+str(self.env)+"_"+str(self)+".pt", map_location=torch.device(device)))
        self.model.eval()


    def __str__(self):
        return "DTAgent" # Aggiornato il nome


    def simulate(self, number = 0, path = "", verbose = False):
        done = False
        self.epsilon = 0.0 # Durante la simulazione, nessuna esplorazione
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        position_x = []
        position_y = []
        position_x.append(self.env.car.position[0])
        position_y.append(self.env.car.position[1])

        # Per la simulazione, devi anche specificare un target return desiderato.
        # Questo può essere lo stesso usato in addestramento, o un valore specifico.
        target_return = 100.0 # Ad esempio, cerca di raggiungere un ritorno di 100

        # Reset current episode buffer for simulation (important to maintain seq_len history)
        self.memory.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        while not done:
            if verbose:
                print(f"State: {state.cpu().numpy()}")

            action = self.act(state.unsqueeze(0), target_return_scalar=target_return)

            if verbose:
                print(f"Action: {action}")
            
            next_state, reward_vec, terminated, truncated, _  = self.env.step(action)

            # Aggiorna il target_return per il prossimo passo nella simulazione
            target_return -= reward_vec[0] # Usa la stessa logica di aggiornamento dell'addestramento

            done = terminated or truncated

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Aggiungi la transizione all'episodio corrente nel buffer della simulazione
            # Questo è cruciale perché `act` ha bisogno della storia per costruire la sequenza.
            self.memory.current_episode['states'].append(state.cpu())
            self.memory.current_episode['actions'].append(torch.tensor(action, dtype=torch.long).cpu())
            self.memory.current_episode['rewards'].append(torch.tensor(reward_vec, dtype=torch.float32).cpu())
            self.memory.current_episode['dones'].append(torch.tensor(terminated, dtype=torch.bool).cpu())


            state = next_state_tensor

            position_x.append(self.env.car.position[0])
            position_y.append(self.env.car.position[1])

        plt.plot(position_x, position_y);

        # lanes
        plt.plot([0.0, self.env.dim_x], [0.0, 0.0]);
        plt.plot([0.0, self.env.dim_x], [self.env.dim_y, self.env.dim_y]);

        # jaywalker
        jaywalker_position_x = [self.env.jaywalker[0] - self.env.jaywalker_r,\
                                self.env.jaywalker[0] - self.env.jaywalker_r,\
                                self.env.jaywalker[0] + self.env.jaywalker_r,\
                                self.env.jaywalker[0] + self.env.jaywalker_r,\
                                self.env.jaywalker[0] - self.env.jaywalker_r]
        
        jaywalker_position_y = [self.env.jaywalker[1] - self.env.jaywalker_r,\
                                self.env.jaywalker[1] + self.env.jaywalker_r,\
                                self.env.jaywalker[1] + self.env.jaywalker_r,\
                                self.env.jaywalker[1] - self.env.jaywalker_r,\
                                self.env.jaywalker[1] - self.env.jaywalker_r]
        
        plt.plot(jaywalker_position_x, jaywalker_position_y);

        plt.savefig(path + str(self.env) + "_simulation_" + str(number) + "_render.png");
        plt.clf();