import yaml
from pathlib import Path
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
        {"role": "user", "content": user_message}
    ]

def load_config(config_path: str):
    path = Path(__file__).parent.parent / "configs" / config_path
    return yaml.safe_load(path.read_bytes())

def _numbered_dirs(path: Path) -> list[Path]:
    return sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))

def load_prompts(prompts_path: str) -> dict[int, list[str]]:
    root = Path(__file__).parent.parent / "prompts" / prompts_path
    if not root.is_dir():
        raise FileNotFoundError(f"Prompts folder not found: {root}")
    return {
        int(size_dir.name): [
            (run_dir / "prompts.txt").read_text(encoding="utf-8").strip()
            for run_dir in _numbered_dirs(size_dir)
        ]
        for size_dir in _numbered_dirs(root)
    }

def compute_brier_decomposition(
    P: Tensor | list,
    O: Tensor | list
) -> tuple[float, float]:
    # enforce type
    tensor_dict = {"p": P, "o": O}
    for var_name, var in tensor_dict.items():
        if isinstance(var, list):
            tensor_dict[var_name] = Tensor(var)
        if not isinstance(tensor_dict[var_name], Tensor):
            raise TypeError(
                f"type not supported for {var_name}.\n"
                f"only tensor, list are supported. "
                f"Provided type: {type(var)}"
            )

    # validate tensor dimension 
    P, O = tensor_dict["p"], tensor_dict["o"]
    dim_p, dim_o = len(P.size()), len(O.size())

    if  dim_p != 2 or dim_o != 1:
        raise ValueError(
            "Required: dim(P) = 2 and dim(O) = 1\n"
            f"provided: dim(P)={dim_p}, dim(O)={dim_o}"
        )
 
    # valiate number of samples is the same
    if P.size(0) != O.size(0):
        raise ValueError(
            "First dimension of P and O must match\n"
            f"P size: {P.size()}\n"
            f"O size: {O.size()}\n"
        )

    # enforce value range
    if ((0 > P) | (P > 1)).any() or ((O != 0.0) & (O != 1.0)).any():
        raise ValueError(
            "Values of P must be in the range of 0 and 1. "
            "Values of O must be 0 or 1"
        )
    
    # do the actual computation
    p_mean = P.mean(axis=1)
    variance = ((P - p_mean.unsqueeze(-1))**2).mean()
    bias_sq = ((O - p_mean)**2).mean()
    return variance.item(), bias_sq.item()