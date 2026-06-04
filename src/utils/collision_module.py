def check_collision_risk(est_x, est_z, vel_x, vel_z, car_half_width=1.0, safety_margin=0.5):
    """
    Evaluate if the pedestrian will enter the car's trajectory.
    
    Returns:
        (bool, str, float): (Risk, State Message, TTC calculated)
    """
    if vel_z >= -0.5:
        return False, "There is no risk of collision", float('inf')
    
    # Time-to-Collision
    ttc = est_z / abs(vel_z)
    
    # Too far, false
    if ttc > 15.0: 
        return False, "Pedestrian is far away", ttc
        
    # Lateral position at impact
    x_at_impact = est_x + (vel_x * ttc)
    
    # Hitbox check
    hitbox_limit = car_half_width + safety_margin
    
    if abs(x_at_impact) <= hitbox_limit:
        if ttc <= 5.0:
            return True, f"Impact at X={x_at_impact:.2f}m in {ttc:.2f}s", ttc
        else:
            return False, f"Pedestrian in the trajectory at X={x_at_impact:.2f}m", ttc
    else:
        return False, f"Walker at X={x_at_impact:.2f}m", ttc