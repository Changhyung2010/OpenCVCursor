import cv2 as cv

def check_cameras():
    print("Checking cameras...")
    for i in range(5):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera Index {i}: Found")
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  - Frame read successfully: {frame.shape}")
            else:
                print(f"  - Could not read frame.")
            cap.release()
        else:
            print(f"Camera Index {i}: Not found")

if __name__ == "__main__":
    check_cameras()
