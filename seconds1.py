import math
import os
import numpy as np
import glob
import sys
import cv2
import queue
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def get_speed(vehicle):
    """Get the speed of the vehicle in km/h."""
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

class VehiclePIDController:
    """Vehicle PID controller for both longitudinal and lateral control."""
    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        self.max_brake = max_brake
        self.max_steering = max_steering
        self.max_throttle = max_throttle

        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.long_controller = PIDLongitudinalControl(self.vehicle, **args_longitudinal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """Run one step of control for the vehicle."""
        acceleration = self.long_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()

        if acceleration >= 0.0:
            control.throttle = min(abs(acceleration), self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering rate limiting for smoothness
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        control.steer = np.clip(current_steering, -self.max_steering, self.max_steering)
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = control.steer

        return control

class PIDLongitudinalControl:
    """PID controller for longitudinal control (throttle/brake)."""
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)

    def run_step(self, target_speed):
        """Calculate acceleration or brake based on the target speed."""
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)

    def pid_controller(self, target_speed, current_speed):
        """PID controller logic for speed."""
        error = target_speed - current_speed
        self.errorBuffer.append(error)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0

        return np.clip(self.K_P * error + self.K_D * de + self.K_I * ie, -1.0, 1.0)

class PIDLateralControl:
    """PID controller for lateral control (steering)."""
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)

    def run_step(self, waypoint):
        """Calculate the steering angle based on the waypoint."""
        return self.pid_controller(waypoint, self.vehicle.get_transform())

    def pid_controller(self, waypoint, vehicle_transform):
        """PID controller logic for steering based on the difference between vehicle and waypoint."""
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])

        dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        cross = np.cross(v_vec, w_vec)
        
        if cross[2] < 0:
            dot *= -1

        self.errorBuffer.append(dot)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0

        return np.clip(self.K_P * dot + self.K_I * ie + self.K_D * de, -1.0, 1.0)

def main():
    """Main loop for vehicle spawning and control."""
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        map = world.get_map()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('cybertruck')[0]
        
        # Get valid spawn points from the map
        spawn_points = map.get_spawn_points()
        
        # Use a random valid spawn point from the map
        if len(spawn_points) > 0:
            spawnpoint = np.random.choice(spawn_points)
        else:
            print("No spawn points available")
            return

        vehicle = world.spawn_actor(vehicle_bp, spawnpoint)
        actor_list.append(vehicle)
        print("Vehicle spawned:", vehicle)

        control_vehicle = VehiclePIDController(
            vehicle,
            args_lateral={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0},
            args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0}
        )

        while True:
            vehicle_speed = get_speed(vehicle)
            print(f"Vehicle speed: {vehicle_speed} km/h")

            # Fetch current waypoint and calculate the next waypoint with a distance of 2 meters
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            next_waypoints = list(waypoints.next(2.0))  # Distance provided for next() function

            if len(next_waypoints) > 0:
                waypoint = np.random.choice(next_waypoints[0:3])
                control_signal = control_vehicle.run_step(5, waypoint)  # Target speed set to 5 km/h
                vehicle.apply_control(control_signal)
                print(f"Control signal applied: {control_signal}")
            else:
                print("No next waypoints found.")
            time.sleep(0.1)  # Pause between iterations

            depth_camera_bp = blueprint_library.find('sensor.camera.depth')

            depth_camera_transform = carla.Transform(carla.Location(x=1.5, y=2.4))
            depth_camera = world.spawn_actor(depth_camera_bp,depth_camera_transform)
            depth_camera.listen(lambda image: image.save_to_disk('output/%.6d' % image.frame,carla.ColorConverter.LogarithmicDepth))

    finally:
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == '__main__':
    main()
