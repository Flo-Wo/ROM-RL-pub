import torch
import torch.nn as nn


class SindySurrogate(nn.Module):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        state_hidden_dim: int,
        control_hidden_dim: int,
        state_latent_dim: int,
        control_latent_dim: int,
        dict_dim: int,
        act_func,
        **kwargs
    ):
        super(SindySurrogate, self).__init__()
        # safe the state to split the xu vector
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.state_latent_dim = state_latent_dim
        self.control_latent_dim = control_latent_dim
        # define phi_1, psi_1
        self.state_encoder = Encoder(
            in_dim=state_dim,
            hidden_dim=state_hidden_dim,
            latent_dim=state_latent_dim,
            act_func=act_func,
        )
        self.state_decoder = Decoder(
            out_dim=state_dim,
            hidden_dim=state_hidden_dim,
            latent_dim=state_latent_dim,
            act_func=act_func,
        )
        # define phi_2, psi_2
        self.control_encoder = Encoder(
            in_dim=control_dim,
            hidden_dim=control_hidden_dim,
            latent_dim=control_latent_dim,
            act_func=act_func,
        )
        self.control_decoder = Decoder(
            out_dim=control_dim,
            hidden_dim=control_hidden_dim,
            latent_dim=control_latent_dim,
            act_func=act_func,
        )
        # the xi matrix is just a linear layer without a bias
        # matrix has the shape (out_feat, in_feat)
        self.xi = nn.Linear(
            in_features=state_latent_dim,
            out_features=dict_dim,
            bias=False,
        )

    def count_parameters(self):
        """Count the total number of parameters, including the Xi matrix."""
        return sum(
            p.numel()
            for p in self.parameters()
            if isinstance(p, torch.Tensor) and p.requires_grad
        )

    def forward(self, xu: torch.tensor) -> tuple[torch.tensor]:
        """
        Simple forward, return the vec of the latent space, the decoded state and control.
        """
        x, u = torch.split(xu, [self.state_dim, self.control_dim], dim=1)
        x_enc = self.state_encoder(x)
        u_enc = self.control_encoder(u)

        vec_latent = torch.cat((x_enc, u_enc), dim=1)

        x_dec = self.state_decoder(x_enc)
        u_dec = self.control_decoder(u_enc)
        return vec_latent, x_dec, u_dec

    def predict(
        self,
        xu: torch.tensor,
        theta: callable,
    ) -> torch.tensor:
        """Predict the next step"""
        x, u = torch.split(xu, [self.state_dim, self.control_dim], dim=0)
        # print("x.shape", x.shape)
        # print("u.shape", u.shape)
        x_enc = self.state_encoder(x)
        u_enc = self.control_encoder(u)
        vec_latent = torch.cat((x_enc, u_enc), dim=0)
        return self.state_decoder(
            torch.linalg.matmul(
                torch.as_tensor(
                    theta(vec_latent.detach().numpy()), dtype=torch.float32
                ),
                self.xi.weight,
            )
        )


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        act_func: nn.Module,
    ):
        super(Encoder, self).__init__()
        # the input layer is not using an activation function
        self.l1 = _dense_layer(in_dim, hidden_dim, act_func=None)
        self.l2 = _dense_layer(hidden_dim, latent_dim, act_func=act_func)
        self.apply(_init_weights)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.l1(x)
        return self.l2(x)


class Decoder(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        latent_dim: int,
        act_func: nn.Module,
    ):
        super(Decoder, self).__init__()
        # the last layer is not using an activation function
        self.l1 = _dense_layer(latent_dim, hidden_dim, act_func=act_func)
        self.l2 = _dense_layer(hidden_dim, out_dim, act_func=None)
        self.apply(_init_weights)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.l1(x)
        return self.l2(x)


def _dense_layer(
    input_shape: int,
    output_shape: int,
    act_func: torch.nn = None,
    bias: bool = True,
):
    """Create a generic dense layer, which consists of:
        - linear layer with the shape (input_shape, output_shape)
       (- batch normalization layer, see BONUS)
        - activation function (optional)

    Parameters
    ----------
    input_shape : int
        Size of the input vectors.
    output_shape : int
        Size the output vector should have.
    act_func : torch.nn, optional
        Activation function used after the linear layer, by default None
        as we might want to define layers without an activation function.
    bias : bool, optional
        Use a bias, by default True.

    Returns
    -------
    nn.Sequential
        Sequential containing the modules listed above.
    """
    steps = [nn.Linear(input_shape, output_shape, bias=bias)]

    if act_func is not None:
        steps.append(act_func())
    return nn.Sequential(*steps)


def _init_weights(layer: torch.nn):
    """Generically initialize the weights of a layer based on its type.

    Can be used inside the constructor via
    ```python
    self.apply(_init_weights)
    ```
    **after** you defined all layers.

    Parameters
    ----------
    layer : torch.nn
        Generic layer of our network.
    """
    # TODO: check the gain flag
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(
            layer.weight,  # gain=nn.init.calculate_gain('relu')
        )
        torch.nn.init.zeros_(layer.bias)
