import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaWeatherParameters
import customtkinter as ctk

class WeatherControlGUI(ctk.CTk, Node):
    def __init__(self):
        ctk.CTk.__init__(self)
        Node.__init__(self, 'weather_control_gui')

        self.title("CARLA Weather Control")

        self.params = {
            'cloudiness': ctk.DoubleVar(),
            'precipitation': ctk.DoubleVar(),
            'precipitation_deposits': ctk.DoubleVar(),
            'wind_intensity': ctk.DoubleVar(),
            'fog_density': ctk.DoubleVar(),
            'fog_distance': ctk.DoubleVar(),
            'wetness': ctk.DoubleVar(),
            'sun_azimuth_angle': ctk.DoubleVar(),
            'sun_altitude_angle': ctk.DoubleVar()
        }

        row = 0
        for param, var in self.params.items():
            ctk.CTkLabel(self, text=param.replace('_', ' ').title()).grid(column=0, row=row, padx=10, pady=5)
            ctk.CTkSlider(self, variable=var, from_=0, to=100, orientation='horizontal').grid(column=1, row=row, padx=10, pady=5)
            entry = ctk.CTkEntry(self, textvariable=var, width=50)  # Adjust the width as needed
            entry.grid(column=2, row=row, padx=10, pady=5)
            row += 1

        ctk.CTkButton(self, text="Send", command=self.send_weather_parameters).grid(column=0, row=row, columnspan=3, pady=10)
        self.publisher_ = self.create_publisher(CarlaWeatherParameters, '/carla/weather_control', 10)


    def send_weather_parameters(self):
        msg = CarlaWeatherParameters()
        msg.cloudiness = self.params['cloudiness'].get()
        msg.precipitation = self.params['precipitation'].get()
        msg.precipitation_deposits = self.params['precipitation_deposits'].get()
        msg.wind_intensity = self.params['wind_intensity'].get()
        msg.fog_density = self.params['fog_density'].get()
        msg.fog_distance = self.params['fog_distance'].get()
        msg.wetness = self.params['wetness'].get()
        msg.sun_azimuth_angle = self.params['sun_azimuth_angle'].get()
        msg.sun_altitude_angle = self.params['sun_altitude_angle'].get()
        
        self.publisher_.publish(msg)
        self.get_logger().info('Weather parameters sent')

def main():
    rclpy.init()
    app = WeatherControlGUI()
    app.mainloop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

