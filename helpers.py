import os


def validate_path(p):
    p = os.path.abspath(p)
    if not os.path.exists(p):
        print("Path '{}' does not exist. Please provide a valid image path to annotate.".format(p))
        exit(1)
