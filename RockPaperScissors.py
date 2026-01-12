import cv2 as cv
import HandTrackingModule as htm
import time

def get_gesture(fingers):
    # fingers[0] is thumb, ignore it.
    # fingers[1:] are [Index, Middle, Ring, Pinky]
    fingers_no_thumb = fingers[1:]
    
    # Rock: All closed (except maybe thumb) -> [0, 0, 0, 0]
    if fingers_no_thumb == [0, 0, 0, 0]:
        return "Rock"
    # Paper: All open -> [1, 1, 1, 1]
    elif fingers_no_thumb == [1, 1, 1, 1]:
        return "Paper"
    # Scissors: Index and Middle open, others closed -> [1, 1, 0, 0]
    elif fingers_no_thumb == [1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"

def determine_winner(g1, g2):
    if g1 == g2:
        return "Draw"
    elif g1 == "Rock":
        return "Player 1 Wins" if g2 == "Scissors" else "Player 2 Wins"
    elif g1 == "Paper":
        return "Player 1 Wins" if g2 == "Rock" else "Player 2 Wins"
    elif g1 == "Scissors":
        return "Player 1 Wins" if g2 == "Paper" else "Player 2 Wins"
    return "Error"

def main():
    cap = cv.VideoCapture(0)
    detector = htm.handDetector(maxHands=2)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.findHands(img)
        
        # Check if hands are detected
        # We need to access detector.results which is set in findHands
        # But detector.results is 'self.results' in the class. 
        # We can access it if we didn't modify the class to hide it.
        # Based on previous view_file, it is self.results.
        
        # However, it's cleaner to check how many hands via findPosition maybe?
        # findPosition returns a list. If list is empty, no hand for that index.
        
        # Let's check results directly if possible or try/except getFingers
        
        hands_detected = 0
        fingers1 = []
        fingers2 = []
        
        # Check Hand 1
        try:
            fingers1 = detector.getFingers(img, 0)
            hands_detected += 1
        except:
            pass
            
        # Check Hand 2
        try:
            fingers2 = detector.getFingers(img, 1)
            hands_detected += 1
        except:
            pass
            
        if hands_detected == 2:
            g1 = get_gesture(fingers1)
            g2 = get_gesture(fingers2)
            
            # Display Gestures and Fingers
            cv.putText(img, f"P1: {g1}", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv.putText(img, f"{fingers1}", (50, 80), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            
            cv.putText(img, f"P2: {g2}", (400, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv.putText(img, f"{fingers2}", (400, 80), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            
            if g1 != "Unknown" and g2 != "Unknown":
                result = determine_winner(g1, g2)
                cv.putText(img, result, (200, 150), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        elif hands_detected == 1:
             cv.putText(img, "Waiting for Player 2...", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        else:
             cv.putText(img, "Waiting for Players...", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        cv.imshow("Rock Paper Scissors", img)
        if cv.waitKey(1) & 0xFF == 27: # Esc to exit
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
