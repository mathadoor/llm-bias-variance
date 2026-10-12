from torch import Tensor
def coerce_numeric(text: str, strict=False, placeholder=-1) -> float | int:
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            if strict:
                raise ValueError(f"The input value could not be coerced")
            else:
                return placeholder

def prep_messages(system_message: str, user_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user_message", "content": user_message}
    ]

def compute_brier_decomposition(
    P: Tensor | list | float,
    O: Tensor | list | float
) -> tuple[float, float]:
    # enforce type
    tensor_types = {}
    tensor_dict = {"p": P, "o": O}
    var_list = [Tensor, list, float]
    for var_name, var in tensor_dict.items():
        for idx, _type in enumerate(var_list):
            if isinstance(var, _type):
                tensor_types[var_name] = idx
                break 
        var_type = tensor_types.get(var_name, None)
        if var_type==1:
            tensor_dict[var_name] = Tensor(var)
        elif var_type==2:
            tensor_dict[var_name] = Tensor([var])
        elif var_type is None:
            raise TypeError(
                f"type not supported for {var_name}.\n"
                f"only tensor, list and float supported. "
                f"Provided type: {_type}"
            )

    # validate tensor dimension 
    P, O = tensor_dict["p"], tensor_dict["o"]
    P, O = P.squeeze(), O.squeeze()
    dim_p, dim_o = len(P.size()), len(O.size())
    if  dim_p > 2 or dim_o > 1:
        raise ValueError(
            "Required: dim(P) <= 2 and dim(O) <= 1\n"
            f"provided: dim(P)={dim_p}, dim(O)={dim_o}"
        )
    if dim_p == 0:
        P = P.reshape(1, 1)
    elif dim_p == 1:
        P = P.unsqueeze(-1)
    
    if dim_o == 0:
        O = O.reshape(1, 1)

    # enforce value range
    if ((0 > P) | (P > 1)).any() or ((O != 0.0) & (O != 1.0)).any():
        raise ValueError(
            "Values of P must be in the range of 0 and 1. "
            "Values of O must be 0 or 1"
        )
    
    p_mean = P.mean(axis=1)
    variance = ((P - p_mean.unsqueeze(-1))**2).mean()
    bias_sq = ((O - p_mean)**2).mean()
    return variance.item(), bias_sq.item()