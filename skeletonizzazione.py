import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# Inizializzazione della fotocamera RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Avvio della pipeline
pipeline.start(config)

# Inizializzazione di Mediapipe per rilevamento e segmentazione del corpo e delle mani
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose()
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Funzione per salvare le coordinate in un file
def save_coordinates(filename, body_points, hand_points, face_points):
    with open(filename, 'a') as file:
        file.write("Punti corpo (x, y, z):\n")
        for point in body_points:
            file.write(f"{point}\n")
        
        file.write("Punti mani (x, y, z):\n")
        for point in hand_points:
            file.write(f"{point}\n")
        
        file.write("Punti faccia (x, y, z):\n")
        for point in face_points:
            file.write(f"{point}\n")
        
        file.write("\n\n")

def get_skeleton_and_depth(frame, depth_frame):
    """
    Funzione per rilevare lo scheletro, le dita e ottenere la profondità dei punti articolari.
    """
    # Conversione del frame in RGB per Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    segmentation_results = segmenter.process(rgb_frame)
    
    # Creazione della maschera per la segmentazione
    body_mask = segmentation_results.segmentation_mask
    mask = (body_mask > 0.5).astype(np.uint8)  # Soglia per separare corpo e sfondo
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    skeleton_points = []
    hand_points = []
    face_points = []

    # Disegno delle connessioni del corpo
    if pose_results.pose_landmarks:
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = pose_results.pose_landmarks.landmark[start_idx]
            end_landmark = pose_results.pose_landmarks.landmark[end_idx]
            
            # Converti le coordinate normalizzate in pixel
            x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
            x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

            # Verifica se i punti sono all'interno dell'immagine e della maschera
            if (0 <= x1 < depth_frame.shape[1] and 0 <= y1 < depth_frame.shape[0] and
                0 <= x2 < depth_frame.shape[1] and 0 <= y2 < depth_frame.shape[0] and
                mask[y1, x1] > 0 and mask[y2, x2] > 0):

                # Estrai le profondità
                depth1 = depth_frame[y1, x1]
                depth2 = depth_frame[y2, x2]
                
                # Aggiungi le coordinate (x, y, z)
                skeleton_points.append((x1, y1, depth1))
                skeleton_points.append((x2, y2, depth2))
                
                # Disegno della linea tra i punti
                cv2.line(masked_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Disegno delle mani
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]
                
                # Converti le coordinate normalizzate in pixel
                x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

                # Verifica se i punti sono all'interno dell'immagine e della maschera
                if (0 <= x1 < depth_frame.shape[1] and 0 <= y1 < depth_frame.shape[0] and
                    0 <= x2 < depth_frame.shape[1] and 0 <= y2 < depth_frame.shape[0] and
                    mask[y1, x1] > 0 and mask[y2, x2] > 0):

                    # Estrai le profondità
                    depth1 = depth_frame[y1, x1]
                    depth2 = depth_frame[y2, x2]
                    
                    # Aggiungi le coordinate (x, y, z)
                    hand_points.append((x1, y1, depth1))
                    hand_points.append((x2, y2, depth2))
                    
                    # Disegno della linea tra i punti della mano
                    cv2.line(masked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Rilevamento del viso (FaceMesh)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                z = landmark.z  # Profondità relativa

                # Verifica se il punto è all'interno dell'immagine
                if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                    depth = depth_frame[y, x]
                    face_points.append((x, y, depth))

                    # Disegno del punto del viso
                    #cv2.circle(masked_frame, (x, y), 1, (0, 0, 255), -1)

    return masked_frame, skeleton_points, hand_points, face_points

try:
    while True:
        # Acquisizione dei frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Conversione dei frame in array NumPy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Rilevamento dello scheletro, segmentazione e profondità
        segmented_image, skeleton_points, hand_points, face_points = get_skeleton_and_depth(color_image, depth_image)

        # Salvataggio delle coordinate su file di testo
        save_coordinates("coordinates.txt", skeleton_points, hand_points, face_points)

        # Visualizzazione dei risultati
        cv2.imshow('Segmented Body with Skeleton and Depth', segmented_image)
        
        # Esci premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Rilascio della pipeline e chiusura delle finestre
    pipeline.stop()
    cv2.destroyAllWindows()
