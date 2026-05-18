import utils.kalman as kalman

# Simulazione rapida
filtro = kalman.TrajectoryKalmanFilter(dt=0.1) # Ipotizziamo 10 FPS per la simulazione

# Il pedone è a 2 metri di lato (X=2) e parte da 50 metri di distanza (Z=50)
# Noi avanziamo di 1 metro a frame (10 m/s * 0.1s)
distanza_z = 50.0 
pos_x = 2.0

print(f"{'Misura X':<10} | {'Misura Z':<10} || {'Stima X':<10} | {'Stima Z':<10} | {'Vel X':<10} | {'Vel Z':<10}")
print("-" * 75)

for _ in range(15):
    # La X misurata resta uguale, la Z misurata scende perché noi ci avviciniamo
    ex, ez, vx, vz = filtro.predict_update(pos_x, distanza_z)
    
    print(f"{pos_x:<10.2f} | {distanza_z:<10.2f} || {ex:<10.2f} | {ez:<10.2f} | {vx:<10.2f} | {vz:<10.2f}")
    
    distanza_z -= 8.0 # Ci avviciniamo di 1 metro a frame