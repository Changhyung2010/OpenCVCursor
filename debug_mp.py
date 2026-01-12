import mediapipe as mp
print("File:", mp.__file__)
print("Dir:", dir(mp))
try:
    import mediapipe.solutions
    print("Explicit import succeeded")
except Exception as e:
    print("Explicit import failed:", e)
