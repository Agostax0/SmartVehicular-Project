def check_collision_risk(est_x, est_z, vel_x, vel_z, car_half_width=1.0, safety_margin=0.5):
    """
    Evaluate if the pedestrian will enter the car's trajectory.
    
    Returns:
        (bool, str, float): (Risk, State Message, TTC calculated)
    """
    if vel_z >= -0.5:
        return False, "Nessun pericolo di avvicinamento", float('inf')
    
    # Time-to-Collision
    ttc = est_z / abs(vel_z)
    
    # Too far, false
    if ttc > 3.0: 
        return False, "Pedone lontano", ttc
        
    # Lateral position at impact
    x_at_impact = est_x + (vel_x * ttc)
    
    # Hitbox check
    hitbox_limit = car_half_width + safety_margin
    
    if abs(x_at_impact) <= hitbox_limit:
        if ttc <= 1.5:
            return True, f"Impatto a X={x_at_impact:.2f}m in {ttc:.2f}s", ttc
        else:
            return False, f"Pedone in rotta a X={x_at_impact:.2f}m", ttc
    else:
        return False, f"Pedone a X={x_at_impact:.2f}m", ttc