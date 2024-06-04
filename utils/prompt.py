import re


class Lora:
    value: str
    scale: float

    def __init__(self, value: str, scale: float):
        self.value = value
        self.scale = scale


class Prompt:
    value: str
    lora: Lora

    def __init__(self, value: str, lora: Lora | None = None):
        self.value = value
        self.lora = lora


def parser(prompt: str) -> Prompt:
    a = re.findall(r'<([^>]+)>', prompt)
    for i in range(len(a)):
        b = a[i].split(":")
        if len(b) == 3:
            if b[0].lower() == "lora":
                return Prompt(value=prompt.replace("<{}>".format(a[i]), ""), lora=Lora(value=b[1], scale=float(b[2])))
    return Prompt(value=prompt)


if __name__ == "__main__":
    print(parser("<LORA:example:1>,running"))
    print(parser("1girl,<lora:example:1>"))
    print(parser("1girl,<lora:example:1>,running,<lora2:example:1>,running"))
