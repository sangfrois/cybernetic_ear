import torch
from torch.autograd import Variable

def ewc_penalty(model, fisher_matrix, prev_weights, lambda_ewc):
    """
    Calculates the EWC penalty.

    Args:
        model (nn.Module): The model to penalize.
        fisher_matrix (dict): A dictionary containing the Fisher Information Matrix for each parameter.
        prev_weights (dict): A dictionary containing the weights of the model from the previous task.
        lambda_ewc (float): The EWC regularization strength.

    Returns:
        torch.Tensor: The EWC penalty.
    """
    penalty = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher = fisher_matrix[name]
            prev_weight = prev_weights[name]
            penalty += (fisher * (param - prev_weight) ** 2).sum()
    return lambda_ewc * penalty

def compute_fisher(model, dataset):
    """
    Computes the Fisher Information Matrix for a policy network.

    Args:
        model (nn.Module): The policy network.
        dataset (torch.Tensor): A tensor of states of shape (batch_size, feature_dim).

    Returns:
        dict: A dictionary containing the Fisher Information Matrix for each parameter.
    """
    fisher_matrix = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_matrix[name] = torch.zeros_like(param.data)

    model.eval()
    for inputs in dataset:
        model.zero_grad()

        # Ensure input has batch dimension
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        policy = model(inputs)
        log_likelihood = policy.log()

        # Sample an action from the policy
        action = torch.multinomial(policy, 1).item()

        # Calculate the negative log-likelihood for the sampled action
        # Handle both batched and single inputs
        if log_likelihood.dim() == 2:
            loss = -log_likelihood[0, action]
        else:
            loss = -log_likelihood[action]

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_matrix[name] += param.grad.data ** 2

    model.train()

    for name, param in fisher_matrix.items():
        param /= len(dataset)

    return fisher_matrix