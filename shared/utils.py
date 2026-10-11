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

def compute_brier_score(
    P: Tensor,
    O: Tensor
) -> float:
    if isinstance(P, list):
        P = Tensor(P)
    if isinstance(O, list):
        O = Tensor(O)
    P, O = P.squeeze(), O.squeeze()
    
    dim_p, dim_o = len(P.size()), len(O.size())
    if  dim_p > 2 or dim_o != 1:
        ValueError(
            f"Required: dim(P) <= 2 and dim(O) == 1, 
            provided: dim(P): {dim_p}, dim(O): {dim_o}"
        )
    
    if len(P.size()) == 1:
        P = P.unsqueeze(-1)

    p_mean = P.mean(axis=1)
    variance = ((P - p_mean)**2).mean()
    bias_sq = ((O - p_mean)**2).mean()
    return bias_sq