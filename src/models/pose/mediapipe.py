try:
    import mediapipe
    assert(mediapipe.__version__ == "0.10.0")
    print(mediapipe.__version__)
except:
    raise Exception("mediapipe dependency was not found. Make sure you pip installed mediapipe-requirements.txt")

