from utils.classes import *

#replay buffer




# partiamo con azioni discrete
# reward = [collision, distance from target, centered in own lane]
# speed is saturated to 50 NO MORE




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerQNetwork(nn.Module):
    def __init__(
        self,
        n_observations,
        n_actions,
        hidden=64,
        learning_rate=1e-4,
        weights=None,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        reward_size=3,
        include_action_reward=False  # <--- nuovo argomento
    ):
        super().__init__()
        self.reward_size = reward_size
        self.n_actions = n_actions
        self.include_action_reward = include_action_reward

        # input dimension
        self.input_dim = n_observations
        if self.include_action_reward:
            self.input_dim += 1 + reward_size  # 1 per azione scalare, reward vettoriale

        d_model = hidden

        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, n_actions * reward_size)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        print("Transformer con input_dim =", self.input_dim, "caricato!")

    def forward(self, states, actions=None, rewards=None, attention_mask=None):
        """
        Args:
            x: Tensor of shape (B, T, obs_dim)
            attention_mask: optional tensor of shape (B, T)
                            1 = keep token, 0 = mask it (like padding)
        Returns:
            q_values: Tensor of shape (B, T, n_actions, reward_size)
        """
        if self.include_action_reward:
            assert actions is not None and rewards is not None, "actions and rewards must be provided"
            x = torch.cat([states, actions, rewards], dim=2)  # (B, T, obs + 1 + R)
            batch_size, seq_len, _ = x.shape
        else:
            x = states  # solo osservazioni

        

        # Proiezione degli input nello spazio d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Aggiunta del positional encoding
        x = self.pos_encoder(x)

        # Costruzione attention mask (opzionale)
        if attention_mask is not None:
            # Trasforma mask booleana in float (0 = maschera, -inf = maschera)
            # PyTorch usa float mask: 0.0 = keep, -inf = mask
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            extended_mask = extended_mask.to(dtype=torch.float32)
            extended_mask = (1.0 - extended_mask) * -1e9  # 0 -> 0, 1 -> -inf
            # TransformerEncoder in PyTorch accetta src_key_padding_mask invece
            src_key_padding_mask = (attention_mask == 0)  # (B, T)
        else:
            src_key_padding_mask = None

        # Trasformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Head: produce Q-values
        x = self.head(x)  # (B, T, n_actions * reward_size)
        return x.view(batch_size, seq_len, self.n_actions, self.reward_size)


    def learn(self, predict, target):
        # compute loss and backprop
        self.optimizer.zero_grad()
        loss = self.criterion(predict, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()




def main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, img_filename, simulations_filename, num_simulations, version = ""):

    agent = QAgent(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights)
    

    agent.learn()
    agent.plot_learning(31, title = "Jaywalker", filename = img_filename + str(agent.model) + "_" + version)
    agent.plot_epsilon(img_filename + str(agent.model) + "_" + version)
    
    for i in np.arange(num_simulations):
        agent.simulate(i, simulations_filename + str(agent.model) + "_" + version)
    
    agent.save_model(str(agent.model) + "_" + version)


if __name__ == "__main__":
    
    env = Jaywalker()
    episodes = 10000
    replay_frequency = 3
    gamma = 0.95
    learning_rate = 1e-2 #5e-4
    epsilon_start = 1
    epsilon_decay = 0.997 #0.995
    epsilon_min = 0.01
    batch_size = 256
    train_start = 1000
    target_model_update_rate = 1e-3
    memory_length = 1500000 #100000
    mini_batches = 4
    branch_size = 256
    slack = 0.1
    hidden = 128
    num_simulations = 1
    img_filename = "imgs/"
    simulations_filename = "imgs/simulations/"
    simulations = 0

    network_type = sys.argv[1]

     # print used device
    print(f"Device: {device}")

    

    #p = Pool(32)
    if network_type == "lex":
        network = Lex_Q_Network
        weights = None
    
    elif network_type == "transformer":
        def network_constructor(obs, act, hidden, lr, weights):
            return TransformerQNetwork(
                n_observations=obs,
            n_actions=act,
            hidden=hidden,
            learning_rate=lr,
            reward_size=3
        )
        network = network_constructor # in questo modo sto dicendo che il mio network
                                        # è una funzione che prende in ingresso 5 parametri
                                        # che alloco dentro la init del Qagent
        weights = None




    elif network_type == "weighted":
        network = Weighted_Q_Network
        weights = torch.tensor([1.0, 0.1, 0.01])

        simulations = int(sys.argv[2])

        if simulations > 1:
            weights_list = [weights ** i for i in np.arange(1, simulations+1)]
            img_filename = "weighted_simulations/" + img_filename
            simulations_filename = "weighted_simulations/simulations/"

    elif network_type == "sclar":
        network = Scalar_Q_Network

    else:
        raise ValueError("Network type" + network_type + "unknown")

    if simulations > 1:
        for i in np.arange(simulations):
            w = weights_list[i]

            main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, w, img_filename, simulations_filename, num_simulations, "v" + str(i) + "_")
    else:
        main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, img_filename, simulations_filename, num_simulations)
