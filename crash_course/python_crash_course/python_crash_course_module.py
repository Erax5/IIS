"""
This is an example module.

A module is anything you import into your script by writing `import <module name>`.
It can be a local file (like this one), a local folder, or something you have installed in your system.
For example, the statement `import random` in this file imports a module from the Python standard library called `random`.

The text you're reading right now is the module's "docstring".
It can be used to give some general information about the module itself.
To make a docstring, just add a string like this one as the very first statement in a module, class, or function.
By convention, multi-line strings are used. In Python, you can make a multi-line string by using triple quotes (\"\"\").
"""


import random


# List of Unicode diacritic characters. For details, see: https://en.wikipedia.org/wiki/Combining_Diacritical_Marks
diacritics = [ chr(768 + x) for x in range(112) ]


def cursed_text(text, curse_level=4):
    """
    Add a random amount of diacritics to each letter in `text`.
    `curse_level` sets the maximum number of diacritics added to a single letter.
    """

    characters = []

    for original in text:
        num_additions = random.randint(0, curse_level)
        for _ in range(num_additions):
            addition = random.choice(diacritics)
            characters.append(addition)
        characters.append(original)

    return "".join(characters)
