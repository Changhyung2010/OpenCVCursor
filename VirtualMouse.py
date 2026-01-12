import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui
import argparse


class KalmanFilter2D:
    """
    2D Kalman Filter for smooth cursor tracking.
    Predicts and corrects X,Y positions for MacBook-trackpad-level smoothness.
    """
    def __init__(self):
        # State: [x, y, dx, dy] (position + velocity)
        self.kalman = cv.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure x, y)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (lower = smoother, but more lag)
        # Tuned for maximum smoothness
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        
        # Measurement noise covariance (higher = trust prediction more)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        # Initial state covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
    
    def update(self, x, y):
        """Update filter with new measurement and return smoothed position."""
        measurement = np.array([[x], [y]], dtype=np.float32)
        
        if not self.initialized:
            # Initialize state with first measurement
            self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
            return x, y
        
        # Predict
        self.kalman.predict()
        
        # Correct with measurement
        corrected = self.kalman.correct(measurement)
        
        return float(corrected[0]), float(corrected[1])


class DoubleExponentialSmoother:
    """
    Double exponential smoothing (Holt's method) for trend-aware smoothing.
    Better than simple exponential for tracking moving targets.
    """
    def __init__(self, alpha=0.15, beta=0.1):
        self.alpha = alpha  # Data smoothing factor
        self.beta = beta    # Trend smoothing factor
        self.level_x = None
        self.level_y = None
        self.trend_x = 0
        self.trend_y = 0
    
    def smooth(self, x, y):
        if self.level_x is None:
            self.level_x, self.level_y = x, y
            return x, y
        
        # Update level and trend for X
        prev_level_x = self.level_x
        self.level_x = self.alpha * x + (1 - self.alpha) * (self.level_x + self.trend_x)
        self.trend_x = self.beta * (self.level_x - prev_level_x) + (1 - self.beta) * self.trend_x
        
        # Update level and trend for Y
        prev_level_y = self.level_y
        self.level_y = self.alpha * y + (1 - self.alpha) * (self.level_y + self.trend_y)
        self.trend_y = self.beta * (self.level_y - prev_level_y) + (1 - self.beta) * self.trend_y
        
        return self.level_x, self.level_y


class MovingAverageSmoother:
    """Simple moving average for additional smoothing layer."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = []
        self.y_buffer = []
    
    def smooth(self, x, y):
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        if len(self.x_buffer) > self.window_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
        
        return np.mean(self.x_buffer), np.mean(self.y_buffer)


def calculate_hand_size(lmList):
    """
    Calculate hand size based on wrist to middle finger MCP distance.
    Used to normalize pinch threshold for different hand orientations/distances.
    """
    if len(lmList) < 10:
        return 100  # Default fallback
    
    # Distance from wrist (0) to middle finger MCP (9)
    wrist = lmList[0]
    mcp = lmList[9]
    distance = np.sqrt((wrist[1] - mcp[1])**2 + (wrist[2] - mcp[2])**2)
    return max(distance, 50)  # Minimum to avoid division issues


def calculate_normalized_pinch_distance(lmList, hand_size):
    """
    Calculate pinch distance normalized by hand size.
    Works regardless of hand orientation or distance from camera.
    """
    # Index tip (8) and Thumb tip (4)
    x1, y1 = lmList[8][1], lmList[8][2]
    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
    
    # Raw pixel distance
    raw_distance = np.sqrt((x1 - x_thumb)**2 + (y1 - y_thumb)**2)
    
    # Normalize by hand size (returns ratio)
    normalized = raw_distance / hand_size
    
    return normalized, raw_distance


def main():
    parser = argparse.ArgumentParser(description="Virtual Mouse with Hand Tracking")
    parser.add_argument("--camera", type=int, default=0, help="Camera Index (default: 0)")
    parser.add_argument("--smoothing", type=str, default="max", 
                        choices=["kalman", "double", "max"],
                        help="Smoothing: kalman, double (exp), max (all layers)")
    parser.add_argument("--pinch-threshold", type=float, default=0.25,
                        help="Pinch sensitivity 0.1-0.5 (lower=need closer pinch, default: 0.25)")
    args = parser.parse_args()

    # Camera settings
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction (active tracking area margin)
    
    # Smoothing parameters - tuned for maximum smoothness
    DEAD_ZONE = 2  # Smaller dead zone for responsiveness
    MIN_MOVE_INTERVAL = 0.008  # ~120Hz cursor updates
    
    pTime = 0
    last_move_time = 0
    prev_screen_x, prev_screen_y = 0, 0
    
    # Tracking state
    is_tracking_paused = False
    
    # Initialize camera
    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, hCam)
    cap.set(cv.CAP_PROP_FPS, 30)
    
    # Hand detector with optimized settings
    detector = htm.handDetector(maxHands=1, modelComplexity=0, detectionCon=0.7, trackCon=0.7)
    
    # Get screen size
    wScr, hScr = pyautogui.size()
    
    # Disable pyautogui's built-in pause
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True
    
    # Initialize all smoothing layers
    kalman_filter = KalmanFilter2D()
    double_exp_smoother = DoubleExponentialSmoother(alpha=0.15, beta=0.1)
    moving_avg_smoother = MovingAverageSmoother(window_size=4)
    
    # Click state
    click_cooldown = 0
    CLICK_COOLDOWN_TIME = 0.25
    
    # Pinch threshold (normalized ratio)
    pinch_threshold = args.pinch_threshold
    
    print("=" * 50)
    print("Virtual Mouse - Maximum Smoothness Mode")
    print("=" * 50)
    print(f"Smoothing: {args.smoothing}")
    print(f"Pinch threshold: {pinch_threshold} (normalized)")
    print(f"Screen: {wScr}x{hScr}, Camera: {wCam}x{hCam}")
    print("-" * 50)
    print("Controls:")
    print("  SPACE  - Pause/Resume tracking")
    print("  +/-    - Adjust pinch sensitivity")
    print("  ESC    - Exit")
    print("=" * 50)
    
    while True:
        # 1. Capture frame
        success, img = cap.read()
        if not success:
            break
        
        # Check for keyboard input (non-blocking)
        key = cv.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Space - toggle pause
            is_tracking_paused = not is_tracking_paused
            status = "PAUSED" if is_tracking_paused else "RESUMED"
            print(f"Tracking {status}")
        elif key == ord('+') or key == ord('='):  # Increase threshold (less sensitive)
            pinch_threshold = min(0.5, pinch_threshold + 0.02)
            print(f"Pinch threshold: {pinch_threshold:.2f} (less sensitive)")
        elif key == ord('-') or key == ord('_'):  # Decrease threshold (more sensitive)
            pinch_threshold = max(0.1, pinch_threshold - 0.02)
            print(f"Pinch threshold: {pinch_threshold:.2f} (more sensitive)")
        
        # Find hands
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        current_time = time.time()
        
        # Draw tracking area
        color = (0, 0, 255) if is_tracking_paused else (255, 0, 255)
        cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), color, 2)
        
        # Show pause status
        if is_tracking_paused:
            cv.putText(img, "PAUSED (Space to resume)", (wCam//2 - 150, hCam//2), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Process if not paused and hand detected
        if not is_tracking_paused and len(lmList) != 0:
            # Get fingertip positions
            x1, y1 = lmList[8][1:]  # Index Finger Tip
            
            # Check which fingers are up
            try:
                fingers = detector.getFingers(img)
            except:
                fingers = [0, 0, 0, 0, 0]
            
            # Index finger up, middle down = Move mode
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert to screen coordinates
                raw_x = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                raw_y = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                # Apply smoothing layers based on mode
                if args.smoothing == "max":
                    # Layer 1: Kalman filter
                    smooth_x, smooth_y = kalman_filter.update(raw_x, raw_y)
                    # Layer 2: Double exponential
                    smooth_x, smooth_y = double_exp_smoother.smooth(smooth_x, smooth_y)
                    # Layer 3: Moving average
                    smooth_x, smooth_y = moving_avg_smoother.smooth(smooth_x, smooth_y)
                elif args.smoothing == "kalman":
                    smooth_x, smooth_y = kalman_filter.update(raw_x, raw_y)
                else:  # double
                    smooth_x, smooth_y = double_exp_smoother.smooth(raw_x, raw_y)
                
                # Dead zone check
                dx = abs(smooth_x - prev_screen_x)
                dy = abs(smooth_y - prev_screen_y)
                
                if dx > DEAD_ZONE or dy > DEAD_ZONE:
                    if current_time - last_move_time >= MIN_MOVE_INTERVAL:
                        # Mirror X and clamp
                        final_x = max(0, min(wScr - 1, wScr - smooth_x))
                        final_y = max(0, min(hScr - 1, smooth_y))
                        
                        pyautogui.moveTo(final_x, final_y)
                        
                        prev_screen_x, prev_screen_y = smooth_x, smooth_y
                        last_move_time = current_time
                
                # Draw cursor indicator
                cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                
                # Check for click with normalized pinch detection
                if current_time - click_cooldown >= CLICK_COOLDOWN_TIME:
                    hand_size = calculate_hand_size(lmList)
                    normalized_dist, raw_dist = calculate_normalized_pinch_distance(lmList, hand_size)
                    
                    # Use normalized threshold - works regardless of hand orientation
                    if normalized_dist < pinch_threshold:
                        cv.circle(img, (x1, y1), 15, (0, 255, 0), cv.FILLED)
                        pyautogui.click()
                        click_cooldown = current_time
                        
        # FPS display
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv.putText(img, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.putText(img, f"Mode: {args.smoothing}", (20, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.putText(img, f"Pinch: {pinch_threshold:.2f}", (20, 110), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        cv.imshow("Virtual Mouse", img)
            
    cap.release()
    cv.destroyAllWindows()
    print("Virtual Mouse stopped")


if __name__ == "__main__":
    main()
