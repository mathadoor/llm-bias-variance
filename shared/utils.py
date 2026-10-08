def coerce_numeric(text: str, strict=False, placeholder=-1):
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