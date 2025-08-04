import torch
from torch import nn, optim, Tensor
from typing import Optional
import json
from transformers import AutoTokenizer
import time

# Initialize torch device to use cuda if we have a gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# print(f"Using device: {device}")


class SentenceDataset:
    def __init__(self, a):
        with open(a) as f:
            data = json.load(f)
            data = [torch.tensor(seq) for seq in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # changed
        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)

        # See here for PyTorch activation functions
        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.activation = nn.ReLU()

    def forward(self, input: Tensor, hidden_state: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input at timestep t
                - shape: (batch_size, input_dim,)
            hidden_state (Tensor): Hidden state from timestep t-1
                - shape: (batch_size, hidden_dim,)

        Returns:
            Tensor: Next hidden state at timestep t
                - shape: (batch_size, hidden_dim)
        """
        # changed
        emb_out = self.i2h(input)
        prev_out = self.h2h(hidden_state)
        out = self.activation(emb_out+prev_out)

        return out


class SelfAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Self-Attention.
    """

    def __init__(self, num_head, hidden_dim, key_dim, value_dim):
        super(SelfAttention, self).__init__()

        self.num_head = num_head
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim

        assert key_dim * num_head <= hidden_dim, "key_dim × num_head must not exceed hidden_dim"
        assert value_dim * num_head <= hidden_dim, "value_dim × num_head must not exceed hidden_dim"

        # Projections: hidden_dim → num_head × (key_dim/value_dim)
        self.query_proj = nn.Linear(hidden_dim, num_head * key_dim)
        self.key_proj   = nn.Linear(hidden_dim, num_head * key_dim)
        self.value_proj = nn.Linear(hidden_dim, num_head * value_dim)

        # Final output projection back to hidden_dim
        self.output_proj = nn.Linear(num_head * value_dim, hidden_dim)

    def forward(self, y_all):
        """
        Compute multi-head self-attention for all timesteps.

        Args:
            y_all (Tensor): shape (B, T, hidden_dim)

        Returns:
            Tensor: shape (B, T, hidden_dim)
        """
        B, T, _ = y_all.shape
        H = self.num_head
        Dk = self.key_dim
        Dv = self.value_dim

        # Project input to queries, keys, values
        Q = self.query_proj(y_all)  # (B, T, H*Dk)
        K = self.key_proj(y_all)    # (B, T, H*Dk)
        V = self.value_proj(y_all)  # (B, T, H*Dv)

        # Reshape to (B, H, T, Dk or Dv)
        Q = Q.view(B, T, H, Dk).transpose(1, 2)  # (B, H, T, Dk)
        K = K.view(B, T, H, Dk).transpose(1, 2)  # (B, H, T, Dk)
        V = V.view(B, T, H, Dv).transpose(1, 2)  # (B, H, T, Dv)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Dk ** 0.5)  # (B, H, T, T)

        # Apply causal mask to prevent attending to future tokens
        mask = torch.tril(torch.ones(T, T, device=y_all.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, Dv)

        # Merge heads: (B, T, H * Dv)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * Dv)

        # Final linear projection
        output = self.output_proj(attn_output)  # (B, T, hidden_dim)

        return output


class RNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        """
        input_dim (int): Input dimension of RNN
        hidden_dim (int): Hidden dimension of RNN
        """
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # changed
        self.cell = RNNCell(input_dim,hidden_dim)

        # changed
        self.out = nn.Linear(hidden_dim,hidden_dim)

    def step(self, input: Tensor, hidden_prev: Optional[Tensor] = None) -> Tensor:
        """
        Compute hidden and output states for a single timestep

        Args:
            input (Tensor): input at current timestep t
                - shape: (batch_size, input_dim,)
            hidden_prev (Tensor): hidden states of preceding timesteps [1, t-1]
                If there are no previous hidden states (i.e. we are at t=1), then
                this may be None and we will initialize the previous hidden state
                to all zeros.
                - shape: (batch_size, t-1, hidden_dim)

        Returns:
            Tensor: RNN hidden state at current timestep t
                - shape: (batch_size, hidden_dim,)
            Tensor: RNN output at current timestep t.
                RNN output state at current timestep t
                - shape: (batch_size, hidden_dim,)
        """
        if hidden_prev is None:
            # If this is the first timestep and there is no previous hidden state,
            # create a dummy hidden state of all zeros

            # changed
            batch_size = input.shape[0]
            last_hidden_state = torch.zeros(batch_size, self.hidden_dim).to(input)
        else:
            # changed
            last_hidden_state = hidden_prev[:, -1, :]  # (B, hidden_dim)

        # Call the RNN cell and apply the transform to get a prediction
        next_hidden_state = self.cell(input, last_hidden_state)
        next_output_state = self.out(next_hidden_state)

        return next_hidden_state, next_output_state

    def forward(self, sequence: Tensor) -> Tensor:
        """
        Compute hidden and output states for all timesteps over input sequence

        Args:
            sequence (Tensor): inputs to RNN over t timesteps
                - shape (batch_size, t, input_dim)

        Returns:
            Tensor: hidden states over t timesteps
                - shape (batch_size, t, hidden_dim)
            Tensor: output states over t timesteps
                - shape (batch_size, t, hidden_dim)
        """
        hidden_states = None
        output_states = []
        _, t, _ = sequence.shape

        for i in range(t):
            # changed -> extract current input sequence
            inp = sequence[:, i, :]  # (B, 1, input_dim)

            # changed
            next_hidden_state, next_output_state = self.step(inp,hidden_states)
            next_hidden_state = next_hidden_state.unsqueeze(1)

            # changed
            if hidden_states is None:
                hidden_states = next_hidden_state
            else:
                hidden_states = torch.cat([hidden_states, next_hidden_state], dim=1)

            # changed
            output_states.append(next_output_state)  # (B, hidden_dim)

        # changed -> torch.stack all of the output states over the timestep dim
        output_states = torch.stack(output_states, dim=1)  # (B, T, hidden_dim)

        return hidden_states, output_states


class RNNLanguageModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        key_dim=None,
        value_dim=None,
    ):
        """
        embed_dim (int): Dimension of word embeddings
        hidden_dim (int): Dimension of RNN hidden states
        vocab_size (int): Number of (sub)words in model vocabulary
        """
        super(RNNLanguageModel, self).__init__()

        # changed
        self.embeddings = nn.Embedding(vocab_size,embed_dim)
        self.rnn = RNN(embed_dim,hidden_dim)
        self.attention = SelfAttention(hidden_dim,key_dim,value_dim)
        self.lm_head = nn.Linear(hidden_dim,vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Computes next-token logits and hidden states for each token in tokens

        Args:
            tokens (Tensor): Input tokens IDs
                - shape (batch_size, t,)

        Returns:
            Tensor: Next-token logits for each token from the LM head
                - shape (batch_size, t, vocab_size)
            Tensor: RNN hidden states for each token
                - shape (batch_size, t, hidden_dim)
            Tensor: RNN output states for each token
                - shape (batch_size, t, hidden_dim)
        """
        # changed
        i = self.embeddings(tokens) # (B, t, embed_dim)
        h, y_hats = self.rnn(i)  # both (B, t, hidden_dim)
        o = self.attention(y_hats)  # (B, t, hidden_dim)
        g = self.lm_head(o)  # (B, t, vocab_size)

        return g, h, y_hats

    def select_token(self, token_logits: Tensor, temperature: float) -> int:
        """
        Selects (or samples) next token from token_logits

        Args:
            token_logits (Tensor): Next token logits
                - shape (batch_size, vocab_size,)
            temperature (float): Sampling temperature. If 0, do greedy decoding.

        Returns:
            index (int): ID of next token selected
        """
        if temperature == 0:
            # Greedy Decoding
            return torch.argmax(token_logits, dim=-1)
        else:
            # Temperature Sampling
            token_logits = token_logits / temperature
            token_probs = torch.softmax(token_logits, dim=-1)
            index = torch.multinomial(token_probs, 1)[0]
            return index

    def generate(self, tokens: Tensor, max_tokens=10, temperature=0.0) -> Tensor:
        """
        Generates new tokens given `tokens` as a prefix.

        Args:
            tokens (Tensor): Input tokens
                - shape: (1, input_length,)
            max_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature

        Returns:
            Tensor: generated tokens
                - shape: (max_tokens,)
        """
        # Get hidden states for input tokens by calling forward
        token_logits, hidden_states, attn_inputs = self(tokens)
        next_token_logits = token_logits[0, -1]

        new_tokens = []
        step = 0

        # Now, start generating new tokens
        # While we could in theory repeatedly call self(tokens) here, we don't since
        # that's an order of magnitude more inefficient as we would be repeatedly re-encoding
        # the prefix. Instead, here, we repeatedly compute the hidden state and next token
        # logits for the *latest* token.
        while True:
            step += 1

            # Select next token based on next_token_logits
            next_token = self.select_token(next_token_logits, temperature)
            new_tokens.append(next_token.item())

            # Stop generating once we reach max_tokens
            if step >= max_tokens:
                break

            # Get next input embedding
            embed = self.embeddings(next_token)

            # Get next hidden state and next attn input state from RNN
            next_hidden_state, next_attn_input = self.rnn.step(embed, hidden_states)

            # Update hidden states
            hidden_states = torch.cat(
                [hidden_states, next_hidden_state.unsqueeze(1)], dim=1
            )

            # Update attention inputs
            attn_inputs = torch.cat([attn_inputs, next_attn_input.unsqueeze(1)], dim=1)

            # Call attention
            next_output_state = self.attention.step(attn_inputs)

            # Generate the token to be used in the next step of generation
            next_token_logits = self.lm_head(next_output_state)

        return torch.tensor(new_tokens)


def train(lm, train_data, valid_data, loss_fn, optimizer, num_sequences, batch_size):
    """
    Run one epoch of language model training

    Args:
        lm (RNNLanguageModel): RNN language model
        dataset (list[Tensor]): Train dataset
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function
        optimizer: PyTorch Adam optimizer
        num_sequences: The total number of sequences to train on
        batch_size: Number of sequences we process in one step

    Returns:
        List: Training losses
        List: Validation Losses
    """
    # Set the model to training model
    lm.train()
    max_grad_norm = 1.0

    train_batch_losses = []
    train_batch_loss = 0.0
    valid_batch_losses = []

    # DO NOT change the next line
    dataset = train_data
    start_time = time.time()

    # Run validation everytime we process around 10% of the training data
    val_frequency = 0.1
    val_index = int(num_sequences * val_frequency) // batch_size
    if val_index == 0:
        val_index = 1

    # Loop over the dataset
    for idx, sequence in enumerate(dataset):
        time_elapsed = round((time.time() - start_time) / 60, 6)

        # Move the sequence to the device
        sequence = sequence.to(device)

        # Stop training when we hit the num_sequences limit
        if idx == num_sequences // batch_size:
            break

        # changed -> to clear stored gradients from previous iteration of loop
        optimizer.zero_grad()

        # changed
        token_logits, hidden_states, attn_inputs = lm(sequence) # token_logit shape: (batch_size, t, vocab_size)

        # remove very last prediction since we cant compute loss
        logits = token_logits[:, :-1, :] # (batch_size, t-1, vocab_size)

        # remove first actual data since we cant make such prediction
        targets = sequence[:, 1:] # (batch_size, t-1)

        # flatten to turn both into 2D and 1D vector (googled how to use reshape here)
        logits = logits.reshape(-1, token_logits.shape[-1])
        targets = targets.reshape(-1)

        # changed, make sure to check for shape matching ((batch_size * (t - 1)) * (vocab_size) and (batch_size * (t - 1)))
        loss = loss_fn(logits, targets)

        # changed
        loss.backward()

        # DO NOT change this - clip gradient norm to avoid exploding gradients
        nn.utils.clip_grad_norm_(lm.parameters(), max_grad_norm)

        # changed -> updates params given in RNN using Adam's
        optimizer.step()

        # DO NOT change any of the code below
        train_batch_loss += loss.detach().cpu().item()

        if idx % val_index == 0:
            # Calculate train/val loss as normal
            train_batch_loss = (
                round(train_batch_loss / val_index, 6)
                if idx != 0
                else round(train_batch_loss, 6)
            )

            # Append to the batch loss and reset to 0
            train_batch_losses.append(train_batch_loss)
            train_batch_loss = 0.0

            print(
                f"Batch: {idx} | Sequence Length: {(sequence.shape[1])} | Elapsed time (minutes): {time_elapsed}"
            )

            # Append to the validation loss
            valid_loss = round(validate(lm, valid_data, loss_fn), 6)
            valid_batch_losses.append(valid_loss)

    print(f"Train Batch Losses: {train_batch_losses}")

    return train_batch_losses, valid_batch_losses


@torch.no_grad()
def validate(lm, dataset, loss_fn):
    """
    Args:
        lm (RNNLanguageModel):
        dataset (list[Tensor]): Validation dataset
        loss_fn: PyTorch cross entropy loss function

    Returns:
        float: Average validation loss
    """
    # Set the model to eval mode
    lm.eval()

    mean_loss = 0.0
    num_batches = 1

    for i, sequence in enumerate(dataset):
        if i < num_batches:
            # Move the sequence to the device
            sequence = sequence.to(device)

            # TODO: Perform forward pass through the model
            token_dists, _, _ = lm(sequence)  # (batch_size, t, vocab_size)

            # changed -> Compute loss (Same as in train)
            targets = sequence[:, 1:]            
            logits = token_dists[:, :-1, :]    

            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)

            loss = loss_fn(logits, targets)

            # DO NOT change this line
            mean_loss += loss.detach().cpu().item()

    return mean_loss / num_batches


@torch.no_grad()
def complete(prefix: str, num_tokens=64, temperature=0.0):
    """
    Generates text completion from language model given text prefix.
    This function has been implemented for you.

    Args:
        prefix (str):
        num_tokens (int): Number of new tokens to generate
        temperature (float): Sampling temperature

    Returns:
        str: Text completion
    """
    lm.eval()

    input = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")
    input = input.to(device)
    output = lm.generate(input, max_tokens=num_tokens, temperature=temperature)

    return tokenizer.decode(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--metrics_out", type=str)
    parser.add_argument("--train_losses_out", type=str)
    parser.add_argument("--val_losses_out", type=str)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dk", type=int)
    parser.add_argument("--dv", type=int)
    parser.add_argument("--num_sequences", type=int)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
    vocab_size = tokenizer.vocab_size

    # Initialize LM
    lm = RNNLanguageModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        key_dim=args.dk,
        value_dim=args.dv,
    )
    lm = lm.to(device)

    print(lm)
    print(
        "Number of Parameters: ",
        sum(p.numel() for p in lm.parameters() if p.requires_grad),
    )
    print("Loading data")

    train_data = SentenceDataset(args.train_data)

    valid_data = SentenceDataset(args.val_data)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
    )

    print("Finished Loading Dataset")

    # Initialize PyTorch cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.parameters(), lr=1e-3)

    ### BEGIN: Training Loop
    start = time.time()
    train_loss, valid_loss = train(
        lm,
        train_dataloader,
        valid_dataloader,
        loss_fn,
        optimizer,
        args.num_sequences,
        args.batch_size,
    )
    end = time.time()
    time_taken = end - start
    ### END: Training Loop

    results = {
        "Train Losses": train_loss,
        "Valid Losses": valid_loss,
        "Final Train Loss": train_loss[-1],
        "Final Valid Loss": valid_loss[-1],
        "Time": time_taken,
    }

    for key, value in results.items():
        print(key, value)

    print("Final Train Loss: ", train_loss[-1])
    print("Final Valid Loss: ", valid_loss[-1])

    # Saves trained model weights
    torch.save(lm, "model.pt")

    # You can later load back in your model in a separate Python file by running:
    # >>> from rnn import *
    # >>> lm = torch.load("model.pt")

    """
    # Example code for generating text with your LM

    test_str = ["Once upon a time there was a"]
    
    for ts in test_str:
        completion = complete(ts, num_tokens=64, temperature=0.3)
        print("  Test prefix:", ts)
        print("  Test output:", completion)
    """

    # Greedy Sampling(Please comment out when submitting to gradescope)
    test_strs = ["Once upon a time there was a "]
    for ts in test_strs:
        completion = complete(ts, num_tokens=128, temperature=0.0)
        print("  Test prefix:", ts)
        print("  Test output:", completion)

    # # Looping through all temperature values for empirical questions
    # # Please comment out when submitting to gradescope
    print("----------------")
    samples_per_setting = 5
    for temperature in [0, 0.3, 0.8]:
        for _ in range(samples_per_setting):
            completion = complete(test_strs[0], num_tokens=128, temperature=temperature)
            print("  Test prefix:", ts)
            print("  Test output:", completion)
            print("----------------")

    # Save your metrics
    with open(args.train_losses_out, "w") as f:
        for loss in train_loss:
            f.write(str(loss) + "\n")

    with open(args.val_losses_out, "w") as f:
        for loss in valid_loss:
            f.write(str(loss) + "\n")

    with open(args.metrics_out, "w") as f:
        f.write("Final Train Loss: " + str(train_loss[-1]) + "\n")
        f.write("Final Valid Loss: " + str(valid_loss[-1]) + "\n")

