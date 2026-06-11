import pygame


class KeyboardController:
    """Translates pygame key state into vehicle control values.

    Key bindings:
        W / Up    — throttle
        S / Down  — brake / reverse
        A / Left  — steer left
        D / Right — steer right
        Space     — hand-brake
    """

    THROTTLE_STEP = 0.8
    BRAKE_STEP = 1.0
    STEER_STEP = 0.5

    def parse(self, keys) -> dict:
        """Return a control dict from the current pygame key state.

        Args:
            keys: Result of ``pygame.key.get_pressed()``.

        Returns:
            dict with keys *throttle*, *steer*, *brake*, *hand_brake*.
        """
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        hand_brake = False

        # Throttle
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            throttle = self.THROTTLE_STEP

        # Brake / reverse
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            brake = self.BRAKE_STEP

        # Steer left
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer = -self.STEER_STEP

        # Steer right
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer = self.STEER_STEP

        # Hand-brake
        if keys[pygame.K_SPACE]:
            hand_brake = True

        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "hand_brake": hand_brake,
        }
