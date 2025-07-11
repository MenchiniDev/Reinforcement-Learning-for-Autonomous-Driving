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


        batch_size, seq_len, _ = states.shape

        if self.include_action_reward and actions is not None and rewards is not None:
            x = torch.cat([states, actions.unsqueeze(-1), rewards], dim=2)
        else:
            x = states

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.head(x)

        return x.view(batch_size, seq_len, self.n_actions, self.reward_size)


    def learn(self, predict, target):
        # compute loss and backprop
        self.optimizer.zero_grad()
        loss = self.criterion(predict, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    


# ora implemento un Decision transformer

import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size=128,
        n_layer=3,
        n_head=4,
        dropout=0.1,
        max_length=100,
        action_embedding_dim=16
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.embed_dim = 1 + state_dim + action_embedding_dim  # RTG + state + action
        self.hidden_size = hidden_size

        # Proiezioni individuali
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Embedding(act_dim, action_embedding_dim)
        self.action_proj = nn.Linear(action_embedding_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.predict_action = nn.Linear(hidden_size, act_dim)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, rtgs, states, actions=None):
        """
        rtgs: (B, T, 1)
        states: (B, T, state_dim)
        actions: (B, T) – previous actions as input (int), optional
        """
        B, T, _ = states.shape

        # embeddings
        rtg_embed = self.embed_return(rtgs)  # (B, T, H)
        state_embed = self.embed_state(states)  # (B, T, H)

        if actions is not None:
            act_embed = self.embed_action(actions)  # (B, T, A_emb)
            act_embed = self.action_proj(act_embed)  # (B, T, H)
        else:
            act_embed = torch.zeros_like(state_embed)  # per inferenza

        token_embeddings = rtg_embed + state_embed + act_embed  # (B, T, H)

        # Aggiungi embedding posizionali
        token_embeddings = token_embeddings + self.pos_embedding[:, :T]
        token_embeddings = self.embed_ln(token_embeddings)

        x = self.transformer(token_embeddings)  # (B, T, H)

        logits = self.predict_action(x)  # (B, T, act_dim)

        return logits

    def compute_loss(self, logits, target_actions):
        """
        logits: (B, T, act_dim)
        target_actions: (B, T)
        """
        return self.loss_fn(
            logits.reshape(-1, self.act_dim),
            target_actions.reshape(-1)
        )



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
    gamma = 0.95 # Ancora rilevante per calcolare RTG nel buffer
    learning_rate = 1e-4 # Il learning rate per AdamW
    epsilon_start = 1
    epsilon_decay = 0.997
    epsilon_min = 0.01
    batch_size = 256
    train_start = 1000
    target_model_update_rate = 1e-3 # Non usato per DT, ma può restare per compatibilità
    memory_length = 1500000
    mini_batches = 4
    # branch_size = 256 # Non sembra usato nel codice
    slack = 0.1 # Non usato per DT
    hidden = 128
    num_simulations = 1
    img_filename = "imgs/"
    simulations_filename = "imgs/simulations/"
    simulations = 0

    network_type = sys.argv[1] if len(sys.argv) > 1 else None # Gestisce il caso senza argomenti

    # print used device
    print(f"Device: {device}")

    # Rimuovi il blocco 'if network_type is None:'
    # e gestisci direttamente 'transformer'
    
    # if network_type is None: # Questo blocco è superfluo ora
    #     def network_constructor(obs, act, hidden, lr, weights):
    #         return TransformerQNetwork(
    #             n_observations=obs,
    #         n_actions=act,
    #         hidden=hidden,
    #         learning_rate=lr,
    #         reward_size=3
    #     )
    #     network = network_constructor
    #     weights = None

    if network_type == "lex":
        network = Lex_Q_Network
        weights = None
        agent_class = QAgent # Usa l'agente Q per reti Q-like
    
    elif network_type == "transformer":
        def network_constructor(obs, act, hidden_size, lr, weights_unused): # weights_unused per compatibilità
            # Adatta la reward_dim se il tuo RTG non è scalare
            return DecisionTransformer(
                state_dim=obs,
                act_dim=act,
                hidden_size=hidden_size,
                max_length=10, # Deve corrispondere a self.seq_len in DTAgent
                reward_dim=1 # Se il tuo RTG è scalare
            )
        network = network_constructor
        weights = None # Non usato dal DT
        agent_class = DTAgent # Usa il nuovo agente Decision Transformer

    elif network_type == "weighted":
        network = Weighted_Q_Network
        weights = torch.tensor([1.0, 0.1, 0.01])
        simulations = int(sys.argv[2]) if len(sys.argv) > 2 else 1 # Assicurati che sys.argv[2] esista
        agent_class = QAgent

        if simulations > 1:
            weights_list = [weights ** i for i in np.arange(1, simulations+1)]
            img_filename = "weighted_simulations/" + img_filename
            simulations_filename = "weighted_simulations/simulations/"

    elif network_type == "scalar": # Corretto da "sclar" a "scalar"
        network = Scalar_Q_Network
        agent_class = QAgent

    else:
        raise ValueError("Network type " + str(network_type) + " unknown")

    # Inizializza l'agente con la classe corretta (QAgent o DTAgent)
    agent = agent_class(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights) # slack e weights non usati da DT

    # Aggiungi questa funzione di supporto per il main_body
    def run_main_body(agent_instance, network_type_str, version_str=""):
        agent_instance.learn()
        agent_instance.plot_learning(31, title = "Jaywalker", filename = img_filename + network_type_str + "_" + version_str)
        agent_instance.plot_epsilon(img_filename + network_type_str + "_" + version_str)
        
        for i in np.arange(num_simulations):
            agent_instance.simulate(i, simulations_filename + network_type_str + "_" + version_str)
        
        agent_instance.save_model(network_type_str + "_" + version_str)

    if simulations > 1 and network_type == "weighted": # Questa parte è specifica per "weighted"
        for i in np.arange(simulations):
            w = weights_list[i]
            # Ricrea l'agente per ogni set di pesi, perché QAgent è legato ai pesi
            current_agent = agent_class(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, w)
            run_main_body(current_agent, network_type, "v" + str(i) + "_")
    else:
        run_main_body(agent, network_type) # Usa l'agente già creato