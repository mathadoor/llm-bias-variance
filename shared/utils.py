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