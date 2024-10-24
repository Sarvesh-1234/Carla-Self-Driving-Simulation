import sys
import math
import glob
import os
import numpy as np
import cv2
import queue
import logging
import carla

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Append Carla's path (make sure to set the correct path to your CARLA installation)
carla_path = '/path/to/your/carla/dist/carla-*.egg'  # Change this to the correct path
try:
    sys.path.append(glob.glob(carla_path)[0])
except IndexError:
    logging.error("CARLA library not found. Check the path and installation.")

def get_speed(vehicle):
    """Get the current speed of the vehicle in km/h."""
    vel = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # Convert m/s to km/h
    return speed

def process_image(image, vehicle):
    """Process the image from the camera sensor."""
    #try:
        #image.save_to_disk('output/%.6d' % image.frame, carla.ColorConverter.CityScapesPalette)
        #segmentation_image = cv2.imread('output/%.6d.png' % image.frame)
        #cv2.imshow("", segmentation_image)
        #cv2.waitKey(1)
    #except Exception as e:
        #logging.error(f"Image not processed: {e}")

    #return get_speed(vehicle)

    image = np.array(image.raw_data)
    img = image.reshape((600, 800, 4))
    img = img[:, :, :3]

    cv2.imshow("img", img)
    cv2.waitKey(1)


class VehiclePIDController:
    """Controller for vehicle's speed and steering using PID control."""

    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        self.max_brake = max_brake
        self.max_steering = max_steering
        self.max_throttle = max_throttle

        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.long_controller = PIDLongitudinalControl(self.vehicle, **args_longitudinal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)

    def run_step(self, target_speed, waypoints):
        """Calculate the control signals based on the target speed and waypoints."""
        acceleration = self.long_controller.run_step(target_speed)
        
        # Calculate average steering angle from multiple waypoints
        steering = sum(self.lat_controller.run_step(wp) for wp in waypoints) / len(waypoints)
        
        control = carla.VehicleControl()

        # Control logic for throttle and brake
        if acceleration >= 0.0:
            control.throttle = min(abs(acceleration), self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Control logic for steering
        control.steer = max(-self.max_steering, min(self.max_steering, steering))
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = control.steer

        return control

class PIDLongitudinalControl:
    """PID controller for longitudinal control (speed)."""

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)

    def pid_controller(self, target_speed, current_speed):
        """PID control calculation."""
        error = target_speed - current_speed
        self.errorBuffer.append(error)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0

        return np.clip(self.K_P * error + self.K_D * de + self.K_I * ie, -1.0, 1.0)

    def run_step(self, target_speed):
        """Run a control step."""
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)

class PIDLateralControl:
    """PID controller for lateral control (steering)."""

    def __init__(self, vehicle, K_P=1.0, K_D=0.1, K_I=0.01, dt=0.03):
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)

    def run_step(self, waypoint):
        """Run a control step for lateral control."""
        return self.pid_controller(waypoint, self.vehicle.get_transform())

    def pid_controller(self, waypoint, vehicle_transform):
        """PID control calculation for lateral control."""
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
            y=math.sin(math.radians(vehicle_transform.rotation.yaw))
        )
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])

        # Calculate cross track error (CTE)
        cross = np.cross(v_vec, w_vec)[2]

        self.errorBuffer.append(cross)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0

        # PID calculation for steering
        return np.clip((self.K_P * cross) + (self.K_I * ie) + (self.K_D * de), -1.0, 1.0)

# Main script
def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('cybertruck')[0]

        # Get a valid spawn point from the map
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            spawn_point = spawn_points[0]  # Choose the first valid spawn point
        else:
            logging.error("No spawn points available.")
            return

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        control_vehicle = VehiclePIDController(
            vehicle,
            args_lateral={'K_P': 1.2, 'K_D': 0.2, 'K_I': 0.01},  # Adjusted parameters for lateral control
            args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0}
        )

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(lambda image: process_image(image, vehicle))

        while True:
            waypoint = world.get_map().get_waypoint(vehicle.get_location())
            next_waypoints = waypoint.next(3.0)  # Get multiple waypoints
            control_signal = control_vehicle.run_step(30.0, next_waypoints)  # Set target speed to 30
            vehicle.apply_control(control_signal)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        logging.info("Cleaning up actors.")
        for actor in actor_list:
            actor.destroy()

if __name__ == '__main__':
    main()
