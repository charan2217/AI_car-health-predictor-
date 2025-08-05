import obd
import time

class RealOBDSensorData:
    def __init__(self, port_str, interval=1.0):
        self.connection = obd.OBD(port_str)
        self.interval = interval

    def generate(self):
        while True:
            data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'rpm': self._get_value(obd.commands.RPM),
                'coolant_temp': self._get_value(obd.commands.COOLANT_TEMP),
                'intake_pressure': self._get_value(obd.commands.INTAKE_PRESSURE),
                'maf': self._get_value(obd.commands.MAF),
                'throttle_pos': self._get_value(obd.commands.THROTTLE_POS),
                'engine_load': self._get_value(obd.commands.ENGINE_LOAD),
                'vehicle_speed': self._get_value(obd.commands.SPEED),
                'intake_air_temp': self._get_value(obd.commands.INTAKE_TEMP),
                'voltage': self._get_voltage(),
            }
            print(data)
            yield data
            time.sleep(self.interval)

    def _get_value(self, command):
        response = self.connection.query(command)
        return response.value.magnitude if response.value is not None else None

    def _get_voltage(self):
        response = self.connection.query(obd.commands.ELM_VOLTAGE)
        return response.value.magnitude if response.value is not None else None

if __name__ == '__main__':
    sensor = RealOBDSensorData(port_str='COM10', interval=1.0)
    for i, reading in enumerate(sensor.generate()):
        if i > 10:
            break
        print(reading)
