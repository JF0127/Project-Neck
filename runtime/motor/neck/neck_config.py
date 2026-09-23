
# ==============================
network_interface = enp4s0 # type: ignore
slave_id = 0
ack_status = 2

# ==============================
# Motor 1
# ==============================
motor1.passage = 1 # pyright: ignore[reportUndefinedVariable]
motor1.id = 1
# Down: -84 deg; top: 41 deg. min/max are numerical bounds.
motor1.min_position_deg = -84
motor1.max_position_deg = 41
motor1.center_position_deg = -7
motor1.max_velocity_deg_s = 25
motor1.speed_param = 50
motor1.current_param = 500

# ==============================
# Motor 2
# ==============================
motor2.passage = 2
motor2.id = 2
# Top: -250 deg; down: -125 deg. min/max are numerical bounds,
# not the top/down labels. Center -200 is provisional pending calibration.
motor2.min_position_deg = -250
motor2.max_position_deg = -125
motor2.center_position_deg = -200
motor2.max_velocity_deg_s = 25
motor2.speed_param = 50
motor2.current_param = 500

# ==============================
# Motor 3
# ==============================
motor3.passage = 3
motor3.id = 3
# Right: -302 deg; left: -120 deg; middle: -200 deg.
motor3.min_position_deg = -302
motor3.max_position_deg = -120
motor3.center_position_deg = -200
motor3.max_velocity_deg_s = 30
motor3.speed_param = 50
motor3.current_param = 500

# ==============================
# Neck RPY Range
# ==============================
pitch_min_deg = -45
pitch_max_deg = 40
roll_min_deg = -40
roll_max_deg = 40
yaw_min_deg = -117
yaw_max_deg = 58

# ==============================
# Kinematics
# ==============================
c11 = 0.2170
c12 = -0.2413
c21 = 0.3218
c22 = 0.2963
k3 = 1.0
pitch_center_deg = 0
roll_center_deg = 0
yaw_center_deg = 0
det_eps = 0.000001

# ==============================
# Feedback
# ==============================
feedback.enabled = true
feedback.socket_path = "/tmp/neck_feedback.sock"
feedback.rate_hz = 30
