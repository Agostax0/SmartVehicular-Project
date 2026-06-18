"""Steering-wheel controller for manual driving.

Wraps a USB steering wheel + pedal set (Logitech G25/G27/G29, Thrustmaster,
Fanatec, etc.) exposed through pygame's joystick API, and translates it into
the same control dict shape produced by ``KeyboardController.parse`` — so the
engine can swap input sources with no other change.

Axis layout (configurable via constructor args / config.yaml):
    * steering   — wheel rotation, range [-1, +1]  (left negative)
    * throttle   — gas pedal,  analog [0, +1]
    * brake      — brake pedal, analog [0, +1]

Pedals in SDL2 rest at -1 and go to +1 when fully pressed, so each pedal is
remapped with ``(raw + 1) / 2``. Steering is used as-is (linear mapping).

If no joystick is detected at construction time the controller is inert
(``is_connected`` is False) and the engine falls back to the keyboard.
"""

import pygame

from utils.logger import get_logger

logger = get_logger(__name__)


class SteeringWheelController:
    """Translates a USB steering wheel + pedals into vehicle control values.

    Produces a dict identical in shape to ``KeyboardController.parse``:
    ``{throttle, steer, brake, hand_brake}`` so it is a drop-in replacement.

    Default axis indices match the most common Logitech/Thrustmaster layout
    (0 = wheel, 1 = brake, 2 = throttle) but can be overridden, since
    different firmwares expose axes in different orders.
    """

    def __init__(
        self,
        steer_axis=0,
        throttle_axis=2,
        brake_axis=1,
        deadzone=0.02,
        hand_brake_button=None,
    ):
        """Initialise the wheel, must be called after ``pygame.init()``.

        Args:
            steer_axis: Index of the steering axis (wheel rotation).
            throttle_axis: Index of the analog throttle (gas) pedal axis.
            brake_axis: Index of the analog brake pedal axis.
            deadzone: Absolute dead-zone applied to the steering axis to
                suppress jitter when the wheel is centered.
            hand_brake_button: Optional joystick button index mapped to the
                hand-brake. ``None`` disables hand-brake from the wheel.
        """
        self.steer_axis = steer_axis
        self.throttle_axis = throttle_axis
        self.brake_axis = brake_axis
        self.deadzone = deadzone
        self.hand_brake_button = hand_brake_button

        self.joystick = None
        self.is_connected = False

        try:
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                logger.warning(
                    "No steering wheel / joystick detected. "
                    "Falling back to keyboard control."
                )
                return

            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.is_connected = True
            logger.info(
                "Steering wheel connected: '%s' — %d axes, %d buttons.",
                self.joystick.get_name(),
                self.joystick.get_numaxes(),
                self.joystick.get_numbuttons(),
            )
            logger.info(
                "Axis mapping -> steer: %d, throttle: %d, brake: %d. "
                "If pedals feel inverted, adjust under controllers.steering_wheel "
                "in config.yaml.",
                self.steer_axis, self.throttle_axis, self.brake_axis,
            )
        except pygame.error as exc:
            logger.warning("Steering wheel init failed (%s). Using keyboard.", exc)

    def parse(self, _keys=None) -> dict:
        """Read the current wheel + pedal state and return a control dict.

        Args:
            _keys: Unused. Accepted only to keep the same signature as
                ``KeyboardController.parse`` so the engine can call either.

        Returns:
            dict with keys *throttle*, *steer*, *brake*, *hand_brake*.
        """
        if not self.is_connected or self.joystick is None:
            return {"throttle": 0.0, "steer": 0.0, "brake": 0.0, "hand_brake": False}

        # --- Steering: linear, with a small central dead-zone ---
        steer = self.joystick.get_axis(self.steer_axis)
        if abs(steer) < self.deadzone:
            steer = 0.0

        # --- Pedals: SDL2 pedals rest at -1 (released) and reach +1 (floored) ---
        throttle = (self.joystick.get_axis(self.throttle_axis) + 1.0) / 2.0
        brake = (self.joystick.get_axis(self.brake_axis) + 1.0) / 2.0
        # Clamp to [0, 1] to absorb small calibration overshoots.
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        # --- Hand-brake (optional button) ---
        hand_brake = False
        if self.hand_brake_button is not None:
            try:
                hand_brake = bool(self.joystick.get_button(self.hand_brake_button))
            except pygame.error:
                hand_brake = False

        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "hand_brake": hand_brake,
        }
