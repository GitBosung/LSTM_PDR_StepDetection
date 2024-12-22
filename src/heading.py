import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def update_quaternion_by_gyro(quaternion, gyro, delta_time):
    theta = np.linalg.norm(gyro) * delta_time
    if theta == 0:
        return quaternion
    axis = gyro / np.linalg.norm(gyro)
    delta_q = np.hstack([np.cos(theta / 2), np.sin(theta / 2) * axis])
    new_quaternion = quaternion_multiply(quaternion, delta_q)
    return new_quaternion / np.linalg.norm(new_quaternion)

def extract_yaw_from_quaternion(quaternion):
    w, x, y, z = quaternion
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

def calculate_path(gyro_data, step_points, initial_orientation, stride_length=0.62, delta_time=0.02):
    yaw, pitch, roll = initial_orientation
    quaternion = [1, 0, 0, 0]  # 초기 쿼터니언
    positions = [(0, 0)]
    for i in range(1, len(step_points)):
        step_index = step_points[i]
        for j in range(step_points[i-1], step_index):
            gyro = gyro_data[j]
            quaternion = update_quaternion_by_gyro(quaternion, gyro, delta_time)
        yaw = extract_yaw_from_quaternion(quaternion)
        positions.append((
            positions[-1][0] + stride_length * np.sin(yaw),
            positions[-1][1] + stride_length * np.cos(yaw)
        ))
    return np.array(positions)
