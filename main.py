import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import random
import winsound

class AIBirdGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.w, self.h = 1280, 720
        self.cap.set(3, self.w)
        self.cap.set(4, self.h)
        
        # Load the Hand Brain
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.game_started = False 
        self.birds = [] 
        self.particles = []
        self.score = 0
        self.last_shot_time = 0
        self.flash_timer = 0
        self.ix, self.iy = 0, 0

    def draw_bird(self, img, x, y, color, size, is_fat=False):
        # Body
        cv2.circle(img, (int(x), int(y)), size, color, -1)
        cv2.circle(img, (int(x), int(y)), size+2, (0, 0, 0), 2)
        # Eyes
        e_off, e_r = int(size*0.35), int(size*0.22)
        cv2.circle(img, (int(x-e_off), int(y-e_off)), e_r, (255, 255, 255), -1)
        cv2.circle(img, (int(x+e_off), int(y-e_off)), e_r, (255, 255, 255), -1)
        cv2.circle(img, (int(x-e_off), int(y-e_off)), int(e_r*0.4), (0, 0, 0), -1)
        cv2.circle(img, (int(x+e_off), int(y-e_off)), int(e_r*0.4), (0, 0, 0), -1)
        # Beak
        b_w = int(size*0.3)
        pts = np.array([[x-b_w, y+5], [x+b_w, y+5], [x, y+size-10]], np.int32)
        cv2.fillPoly(img, [pts], (0, 165, 255))
        if is_fat: 
            # FIXED: Changed FONT_HERSHEY_BOLD to FONT_HERSHEY_DUPLEX
            cv2.putText(img, "10", (int(x-18), int(y+10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)

    def process_shot(self):
        winsound.Beep(1000, 50) 
        self.flash_timer = 3
        for bird in self.birds[:]:
            bx, by = bird['pos']
            dist = np.linalg.norm(np.array([self.ix, self.iy]) - np.array([bx, by]))
            if dist < bird['size'] + 15:
                winsound.Beep(2000, 30)
                reward = 10 if bird['is_fat'] else 1
                for _ in range(30):
                    self.particles.append({'x': bx, 'y': by, 'color': bird['color'], 
                                         'vx': random.uniform(-15, 15), 'vy': random.uniform(-15, 15), 
                                         'life': 25})
                self.birds.remove(bird)
                self.score += reward

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # Enter Key
                self.game_started = True
                winsound.Beep(1200, 100)
            elif key == ord('q'): break

            # AI Hand Tracking
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = self.detector.detect(mp_image)

            if detection_result.hand_landmarks:
                landmarks = detection_result.hand_landmarks[0]
                itip, ttip = landmarks[8], landmarks[4]
                self.ix, self.iy = int(itip.x * self.w), int(itip.y * self.h)
                tx, ty = int(ttip.x * self.w), int(ttip.y * self.h)
                cv2.drawMarker(frame, (self.ix, self.iy), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 25, 2)

                dist = np.linalg.norm(np.array([self.ix, self.iy]) - np.array([tx, ty]))
                if dist < 45 and (time.time() - self.last_shot_time) > 0.4:
                    if self.game_started:
                        self.last_shot_time = time.time(); self.process_shot()

            # --- BRANDING ---
            cv2.rectangle(frame, (10, 10), (450, 110), (0,0,0), -1)
            cv2.putText(frame, "PROJECT: AI BIRD POPPER", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, "DEV: Vivek Dangwal", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

            if not self.game_started:
                cv2.rectangle(frame, (self.w-320, 10), (self.w-10, 110), (0, 255, 0), 2)
                cv2.putText(frame, "PRESS ENTER TO START", (self.w-300, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            else:
                # SPAWN & MOVE BIRDS
                if random.random() < 0.04:
                    is_fat = random.random() < 0.15
                    size = 70 if is_fat else 35
                    color = (0, 0, 255) if is_fat else (random.randint(50,200), random.randint(50,200), 255)
                    self.birds.append({'pos': [0, random.randint(150, 600)], 'size': size, 
                                       'color': color, 'is_fat': is_fat, 'speed': random.randint(6, 14)})

                for bird in self.birds[:]:
                    bird['pos'][0] += bird['speed']
                    if bird['pos'][0] > self.w: self.birds.remove(bird)
                    else: self.draw_bird(frame, bird['pos'][0], bird['pos'][1], bird['color'], bird['size'], bird['is_fat'])

                # PARTICLES
                for p in self.particles[:]:
                    p['x'] += p['vx']; p['y'] += p['vy']; p['vy'] += 0.6; p['life'] -= 1
                    if p['life'] > 0: cv2.circle(frame, (int(p['x']), int(p['y'])), 4, p['color'], -1)
                    else: self.particles.remove(p)

                cv2.putText(frame, f"SCORE: {self.score}", (self.w-280, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

            if self.flash_timer > 0:
                cv2.circle(frame, (self.ix, self.iy), 120, (255, 255, 255), -1)
                self.flash_timer -= 1

            cv2.imshow("Vivek Dangwal - AI Bird Popper", frame)

        self.cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    game = AIBirdGame(); game.run()