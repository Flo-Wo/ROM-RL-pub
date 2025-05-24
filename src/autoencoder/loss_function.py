import torch
import torch.nn as nn


class GradFreeRefinementLoss(nn.Module):
    def __init__(
        self,
        theta: callable,
        lambda_0: float,
        lambda_1: float,
    ):
        """Loss for the SINDy-C case, i.e. z = x_{t+1}.

        Parameters
        ----------
        theta : callable
            Library functions of pysindy.
        lambda_0 : float
            Multiplier for the prediction of the next state.
        lambda_1 : float
            Multiplier for the AE loss.
        """
        super(GradFreeRefinementLoss, self).__init__()
        self.theta = theta
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1

    def forward(
        self,
        # input: (state, control) & PDE param
        xu: torch.tensor,
        mu: torch.tensor,
        # target: next state
        x_next: torch.tensor,
        model: nn.Module,
    ):
        vec_latent, x_dec, u_dec = model.forward(xu)

        thetaZ_times_Xi = torch.linalg.matmul(
            # no need to transpose, since in torch batches are stacked row-wise
            torch.as_tensor(
                self.theta(vec_latent.detach().numpy()), dtype=torch.float32
            ),  # , mu),
            model.xi.weight,
        )
        loss_pred = (
            torch.linalg.norm(model.state_decoder(thetaZ_times_Xi) - x_next, ord="fro")
            ** 2
        )
        # classical AutoEncoder loss
        loss_AE = (
            torch.linalg.norm(torch.cat((x_dec, u_dec), dim=1) - xu, ord="fro") ** 2
        )
        return (
            self.lambda_0 * loss_pred + self.lambda_1 * loss_AE,
            loss_pred / torch.numel(x_next),
            loss_AE / torch.numel(xu),
        )


class GradFreeFullLoss(GradFreeRefinementLoss):
    def __init__(
        self,
        theta: callable,
        lambda_0: float,
        lambda_1: float,
        lambda_2: float,
    ):
        """See Documentation of GradFreeRefinementLoss.

        Additional Parameters
        ----------
        lambda_2 : float
            Multiplier for the sparsity of Xi, i.e. L1-norm.
        """
        super(GradFreeFullLoss, self).__init__(
            theta=theta, lambda_0=lambda_0, lambda_1=lambda_1
        )
        self.lambda_2 = lambda_2

    def forward(
        self,
        # input: (state, control) & PDE param
        xu: torch.tensor,
        mu: torch.tensor,
        # target: next state
        x_next: torch.tensor,
        model: nn.Module,
    ):
        refinement_loss, loss_pred, loss_AE = super().forward(xu, mu, x_next, model)
        # regularization loss to promote sparsity
        loss_reg = torch.linalg.matrix_norm(model.xi.weight, ord=1)
        return (
            refinement_loss + self.lambda_2 * loss_reg,
            loss_pred,
            loss_AE,
            loss_reg / torch.numel(model.xi.weight),
        )
