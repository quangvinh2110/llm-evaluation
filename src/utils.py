from string import whitespace

def normalize_name(text: str):
    replace_str = "/"+whitespace
    return text.translate((str.maketrans(replace_str, '_'*len(replace_str))))
