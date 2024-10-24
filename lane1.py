import sys
import math
import numpy as np
import queue
import threading
import logging
import carla
import asyncio
import cv2

# Function to display images in a window
def display_img(image):
    """Display the image in a window."""
    cv2.imshow("Lane Detection", image)
    cv2.waitKey(1)  # Short wait for window refresh

def process_frame(frame):
    """Process the image to detect lanes."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest
    height, width = edges.shape
    roi_vertices = np.array([[
        (0, height),
        (width // 2, height // 2),
        (width, height),
    ]], dtype=np.int32)
    
    # Create a mask for the region of interest
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Apply the mask
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform to find lines in the image
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=15, minLineLength=40, maxLineGap=20)

    # Draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green lines for lane detection

    return frame

def get_speed(vehicle):
    """Get the current speed of the vehicle in km/h."""
    vel = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # Convert m/s to km/h
    return speed

# Thread-safe queue for image processing
image_queue = queue.Queue()

def image_processing_thread():
    """Process images from the queue in a separate thread."""
    while True:
        image, vehicle = image_queue.get()  # Block until an image is available
        if image is None:  # Check for exit signal
            break
        image_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        processed_image = process_frame(image_array.copy())
        display_img(processed_image)  # Display the image
        image_queue.task_done()  # Mark task as done

def process_image(image, vehicle):
    """Add the image to the processing queue."""
    image_queue.put((image, vehicle))  # Add to queue for processing

class VehiclePIDController:
    """Controller for vehicle's speed and steering using PID control."""
    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        self.max_brake = max_brake
        self.max_steering = max_steering
        self.max_throttle = max_throttle

        self.vehicle = vehicle
        self.long_controller = PIDLongitudinalControl(self.vehicle, **args_longitudinal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)

    def run_step(self, target_speed, waypoints):
        """Calculate the control signals based on the target speed and waypoints."""
        acceleration = self.long_controller.run_step(target_speed)
        steering = sum(self.lat_controller.run_step(wp) for wp in waypoints) / len(waypoints)

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(abs(acceleration), self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        control.steer = max(-self.max_steering, min(self.max_steering, steering))
        control.hand_brake = False
        control.manual_gear_shift = False

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
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()

        v_vec = np.array([math.cos(math.radians(vehicle_transform.rotation.yaw)),
                          math.sin(math.radians(vehicle_transform.rotation.yaw)), 0.0])
        w_vec = np.array([waypoint.transform.location.x - vehicle_location.x,
                          waypoint.transform.location.y - vehicle_location.y, 0.0])

        cross = np.cross(v_vec, w_vec)[2]
        self.errorBuffer.append(cross)

        if len(self.errorBuffer) >= 2:
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt
            ie = sum(self.errorBuffer) * self.dt
        else:
            de = 0.0
            ie = 0.0

        return np.clip(self.K_P * cross + self.K_D * de + self.K_I * ie, -1.0, 1.0)

async def main():
    """Main asynchronous function to control the vehicle."""
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # Blueprint for vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # Setup the vehicle control
        control_vehicle = VehiclePIDController(
            vehicle,
            args_lateral={'K_P': 1.0, 'K_D': 0.1, 'K_I': 0.01},
            args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}
        )

        # Setup camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '320')  # Further lower resolution for speed
        camera_bp.set_attribute('image_size_y', '240')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Start the image processing thread
        threading.Thread(target=image_processing_thread, daemon=True).start()

        # Use a callback for processing images
        camera.listen(lambda image: process_image(image, vehicle))

        while True:
            waypoint = world.get_map().get_waypoint(vehicle.get_location())
            next_waypoints = waypoint.next(3.0)  # Get multiple waypoints
            control_signal = control_vehicle.run_step(target_speed=30, waypoints=next_waypoints)  # Target speed in km/h
            vehicle.apply_control(control_signal)

            await asyncio.sleep(0.01)  # Yield control to allow other tasks

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        logging.info("Cleaning up actors.")
        for actor in [vehicle, camera]:
            actor.destroy()
        cv2.destroyAllWindows()  # Close any OpenCV windows

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())  # Run the main function in an asyncio loop
