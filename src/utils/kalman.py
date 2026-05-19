import cv2
import numpy as np

class TrajectoryKalmanFilter:
    def __init__(self, dt=0.05):
        """
        Inizializza il Filtro di Kalman 2D per prevedere la traiettoria di un pedone.
        
        Args:
            dt (float): Delta time (secondi) tra un frame e l'altro. 
                        Di default 0.05s presuppone circa 20 FPS.
        """
        self.dt = dt
        
        # Inizializza il filtro di OpenCV:
        # dinamica a 4 stati [X, Z, Vx, Vz]
        # 2 misurazioni in input [X, Z]
        # 2 variabili di controllo [V_ego_x, V_ego_z]
        self.kf = cv2.KalmanFilter(4, 2, 2)
        
        # 1. Matrice di Transizione di Stato (A o F)
        # Descrive la fisica del sistema (moto rettilineo uniforme):
        # X_nuovo = X_vecchio + Vx * dt
        # Z_nuovo = Z_vecchio + Vz * dt
        self.kf.transitionMatrix = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], np.float32)
        
        # 2. Matrice di Misurazione (H)
        # Mappa come le misurazioni si relazionano agli stati.
        # Noi misuriamo solo X e Z (i primi due stati), non le velocità.
        self.kf.measurementMatrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], np.float32)
        
        # 3. Covarianza del Rumore di Processo (Q)
        # Quanto ci aspettiamo che il pedone cambi improvvisamente velocità o direzione.
        # Valori più alti rendono il filtro più "reattivo" ai cambi di direzione, 
        # ma meno fluido. Le velocità (Vx, Vz) hanno solitamente un'incertezza maggiore.
        self.kf.processNoiseCov = np.array([
            [1e-2, 0.0, 0.0, 0.0],
            [0.0, 1e-2, 0.0, 0.0],
            [0.0, 0.0, 5e-2, 0.0],
            [0.0, 0.0, 0.0, 5e-2]
        ], np.float32)
        
        # 4. Covarianza del Rumore di Misurazione (R)
        # Quanto ci fidiamo del sensore Depth di CARLA e della Bounding Box di YOLO.
        # Se la Bounding Box balla molto, alza questi valori (es. 1e-1 o 5e-1).
        self.kf.measurementNoiseCov = np.array([
            [1e-1, 0.0],
            [0.0, 1e-1]
        ], np.float32)
        
        # 5. Covarianza dell'Errore Iniziale (P)
        # Incertezza alta all'avvio perché non conosciamo la velocità iniziale.
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        
        # 6. Matrice di Controllo (B)
        # Sottrae lo spostamento causato dal movimento del veicolo (ego-motion)
        # X_nuovo = X_vecchio + Vx*dt - V_ego_x*dt
        # Z_nuovo = Z_vecchio + Vz*dt - V_ego_z*dt
        self.kf.controlMatrix = np.array([
            [-self.dt, 0.0],
            [0.0, -self.dt],
            [0.0, 0.0],
            [0.0, 0.0]
        ], np.float32)
        
        self.initialized = False

    def predict_update(self, measured_x, measured_z, ego_vx=0.0, ego_vz=0.0):
        """
        Esegue i passi di Previsione e Correzione del Filtro di Kalman.
        
        Args:
            measured_x (float): Posizione laterale misurata al frame corrente (metri).
            measured_z (float): Profondità misurata al frame corrente (metri).
            ego_vx (float): Velocità laterale del veicolo (m/s).
            ego_vz (float): Velocità longitudinale del veicolo (m/s).
            
        Returns:
            tuple: (est_x, est_z, vel_x, vel_z)
                   est_x: Posizione laterale filtrata (relativa).
                   est_z: Profondità filtrata (relativa).
                   vel_x: Velocità laterale stimata ASSOLUTA (m/s).
                   vel_z: Velocità longitudinale stimata ASSOLUTA (m/s).
        """
        # OpenCV richiede rigorosamente input in formato matrice colonna float32
        m_x = np.float32(measured_x)
        m_z = np.float32(measured_z)
        
        if not self.initialized:
            # Al primo avvistamento, impostiamo la posizione misurata come stato iniziale
            # e assumiamo una velocità assoluta di 0 m/s in entrambe le direzioni.
            self.kf.statePost = np.array([[m_x], [m_z], [0.0], [0.0]], np.float32)
            self.kf.statePre = np.array([[m_x], [m_z], [0.0], [0.0]], np.float32)
            self.initialized = True
            return float(m_x), float(m_z), 0.0, 0.0
            
        # FASE 1: PREDICT
        # Il filtro ipotizza la nuova posizione applicando le velocità assolute e 
        # sottraendo lo spostamento del veicolo (ego-motion)
        control = np.array([[np.float32(ego_vx)], [np.float32(ego_vz)]], np.float32)
        self.kf.predict(control)
        
        # FASE 2: CORRECT (UPDATE)
        # Correggiamo l'ipotesi usando le nuove misurazioni reali di YOLO e del Depth Sensor
        measurement = np.array([[m_x], [m_z]], np.float32)
        estimated_state = self.kf.correct(measurement)
        
        # Estraiamo i 4 valori dallo stato aggiornato
        est_x = float(estimated_state[0, 0])
        est_z = float(estimated_state[1, 0])
        vel_x = float(estimated_state[2, 0])
        vel_z = float(estimated_state[3, 0])
        
        return est_x, est_z, vel_x, vel_z